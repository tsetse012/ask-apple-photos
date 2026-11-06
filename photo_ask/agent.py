"""질문 에이전트: 자연어 질문 → Claude(tool use)가 카드 인덱스를 검색 → 후보 묶음.

이 단계는 사진을 보내지 않는다(사진은 캡션 만들 때 Gemini로만 나간다 — index/enrich_caption.py).
Anthropic으로 나가는 것은 질문 문장, 인덱스 요약(시스템 프롬프트), 도구 결과(건수·분포·앞쪽 몇 건의
날짜·장소·라벨·캡션 장면 한 줄)뿐이다. 매 호출을 logs/llm.jsonl 에, 대화 원문 전체를
logs/transcripts/<질의 id>.json 에 남긴다.
"""
import json
import re
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from . import config, llm_log
from .caption import CUISINES, DOC_TYPES
from .cards_query import VISION_FLOOR, VISION_MIN_CONF, CardIndex, FilterError

MAX_TOKENS = 8000
BETAS = ["server-side-fallback-2026-07-01"]
ASK_THRESHOLD = 50          # 최종 후보가 이보다 많으면 한 번 되묻는다(사양 ③)
NUDGE = ("(자동 점검) 최종 후보가 {n}건으로 50건을 넘습니다. 규칙 3대로 breakdown에서 실제로 갈리는 축 하나를 골라 "
         "한 번 되물어 주세요. 같은 답 형식에 QUESTION 줄을 넣으세요.")
HAN = re.compile(r"[\u4e00-\u9fff]")   # CJK 한자 — 답은 한글이어야 한다(모델이 「要약:」처럼 표제를 섞는 경우가 있다)
HAN_NUDGE = ("(자동 점검) 답에 한자가 섞였습니다({chars}). 같은 내용을 한글로만, 같은 답 형식으로 다시 써 주세요. "
             "도구는 다시 부르지 마세요.")
WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]

_FILTER_PROPS = {
    "date_from": {"type": "string", "description": "YYYY-MM-DD, 이 날짜 포함(설정한 시간대 날짜)"},
    "date_to": {"type": "string", "description": "YYYY-MM-DD, 이 날짜 포함"},
    "states": {"type": "array", "items": {"type": "string"},
               "description": "1차 행정구역(도·특별시·광역시·해외 주/도). 여러 개면 OR. 「강원도」처럼 줄여 써도 됨"},
    "cities": {"type": "array", "items": {"type": "string"},
               "description": "시·군·구(또는 동). 여러 개면 OR. 「부산」처럼 줄여 써도 됨"},
    "labels_any": {"type": "array", "items": {"type": "string"},
                   "description": "이 중 하나라도 있는 카드(OR). 장면 라벨·장소 유형·시간대·활동·문서 유형 등 인덱스 낱말"},
    "labels_all": {"type": "array", "items": {"type": "string"}, "description": "전부 있어야 함(AND)"},
    "exclude_labels": {"type": "array", "items": {"type": "string"}, "description": "하나라도 있으면 뺌"},
    "text_like": {"type": "string",
                  "description": "라벨·장소·앨범·제목·설명·사진 속 인식 글자에서 부분 일치. '|'로 여러 개 OR"},
    "is_video": {"type": "boolean"},
    "min_faces": {"type": "integer", "description": "얼굴 수 하한"},
    "max_faces": {"type": "integer", "description": "얼굴 수 상한(0이면 사람 얼굴 없는 사진)"},
    "favorite": {"type": "boolean"},
    "screenshot": {"type": "boolean"},
    "hour_from": {"type": "integer", "description": "찍은 곳 현지 시각 0~23. hour_from > hour_to면 자정을 넘는 구간"},
    "hour_to": {"type": "integer"},
    "include_hidden": {"type": "boolean", "description": "숨김 사진 포함(기본 false)"},
    "cuisine": {"type": "array", "items": {"type": "string", "enum": CUISINES},
                "description": "요리 국적(캡션). 여러 개면 OR. 사진만(영상엔 캡션 없음)"},
    "caption_like": {"type": "string",
                     "description": "캡션의 장면·물체·요리 이름·키워드에서 부분 일치. '|'로 여러 개 OR"},
    "doc_type": {"type": "array", "items": {"type": "string", "enum": [d for d in DOC_TYPES if d != "없음"]},
                 "description": "문서 종류(캡션). 여러 개면 OR"},
    "vision_labels_any": {"type": "array", "items": {"type": "string"},
                          "description": "공개 Vision 분류기 라벨(한국어 이름 또는 영어 id) 중 하나라도(OR)"},
    "vision_min_conf": {"type": "number",
                        "description": f"vision_labels_any 확신도 하한(기본 {VISION_MIN_CONF}, 최저 {VISION_FLOOR})"},
    "content_match": {"type": "string", "enum": ["all", "any"],
                      "description": ("사진 내용 조건(labels_any·vision_labels_any·caption_like)끼리 묶는 법. all(기본)=모두 맞아야, "
                                      "any=하나라도 맞으면. 다른 조건(날짜·장소·cuisine 등)은 언제나 AND")},
    "apple_label_min_conf": {"type": "number",
                             "description": ("labels_any/labels_all/exclude_labels를 사진 앱 장면 확신도 표까지 넓힌다 — "
                                             f"이 확신도 이상이면 그 라벨이 있는 것으로 본다(넓힐 때 0.1, 최저 {VISION_FLOOR})")},
}

TOOLS = [
    {
        "name": "search_cards",
        "description": ("카드 인덱스에서 조건을 모두 만족(AND)하는 사진·영상을 찾는다. 숨김 사진은 기본 제외. "
                        "결과: result_id, total(전체 건수), "
                        "breakdown(date_min·date_max 날짜 범위, 상위 6개월 건수, 장소·라벨·시간대·얼굴 수 분포), "
                        "rows(정렬 순 앞쪽 limit건), resolved(실제 이름으로 푼 장소), "
                        "unknown(인덱스에 없는 이름과 비슷한 후보 — 이 조건은 0건 처리됨)."),
        "input_schema": {"type": "object", "properties": {
            **_FILTER_PROPS,
            "sort": {"type": "string",
                     "enum": ["date_desc", "date_asc", "aesthetic", "faces", "text_len", "relevance"],
                     "description": "기본 date_desc. relevance는 text_like가 있을 때 전문검색 점수 순"},
            "limit": {"type": "integer", "description": "rows로 받을 건수(기본 8, 최대 30)"},
        }, "additionalProperties": False},
    },
    {
        "name": "stats",
        "description": "조건에 맞는 카드 수를 한 축으로 센다(상위 40~60개). 필터 인자는 search_cards와 같다.",
        "input_schema": {"type": "object", "properties": {
            "group_by": {"type": "string", "enum": ["month", "year", "weekday", "hour", "state", "city",
                                                     "label", "venue_type", "time_of_day", "activity"]},
            **_FILTER_PROPS,
        }, "required": ["group_by"], "additionalProperties": False},
    },
    {
        "name": "list_vocab",
        "description": ("인덱스에 실제 있는 낱말과 건수. kind=labels는 장면 라벨·장소 유형·시간대·활동·문서 유형 등 "
                        "모든 태그를 종류별로, states/cities/places는 장소 이름(places=관심 지역·동네·가게 이름), "
                        "cuisine/dishes/doc_types는 캡션의 요리 국적·요리 이름·문서 종류, vision_labels는 Vision 분류기 라벨"
                        "(min_conf 이상, 기본 0.05). contains(포함)·prefix(시작)로 거른다."),
        "input_schema": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["labels", "states", "cities", "places", "cuisine", "dishes",
                                                 "doc_types", "vision_labels"]},
            "prefix": {"type": "string"},
            "contains": {"type": "string"},
            "limit": {"type": "integer", "description": "기본 80, 최대 400"},
            "min_conf": {"type": "number", "description": "kind=vision_labels일 때 확신도 하한(기본 0.05)"},
        }, "required": ["kind"], "additionalProperties": False},
    },
    # strict(문법 강제)는 쓰지 않는다 — ①선택 인자 합계 24개 상한(세 도구 합 37개 → API가 400으로 거절),
    # ②search_cards에만 켜자 모델이 states를 먼저 쓰면 date_from을 끝내 못 넣고 같은 호출을 9번 반복.
    # 대신 서버 쪽에서 모르는 인자 거부 + 쉼표 문자열·공백 정리(_normalize_args).
]
_ALLOWED_KEYS = {t["name"]: set(t["input_schema"]["properties"]) for t in TOOLS}

SYSTEM_TEMPLATE = """너는 한 사람의 개인 사진 보관함에서 사진을 찾아 주는 검색 도우미다. 사진 자체는 볼 수 없다. 사진·영상마다 만들어 둔 '카드'(찍은 날짜·시각, 장소 이름, 사진 앱이 붙인 장면 라벨, 장소 유형, 얼굴 수, 영상 여부 등 글자 메타 + 사진마다 클라우드 모델이 미리보기를 보고 쓴 캡션: 장면 한 줄·물체·음식과 요리 국적·문서 종류·키워드)를 도구로 검색한다. 캡션은 사진에만 있고 영상엔 없다.

## 보관함 요약
{summary}

## 도구 쓰는 법
- search_cards는 조건을 모두 만족(AND)하는 카드를 찾는다. labels_any/labels_all/exclude_labels에는 인덱스에 실제 있는 낱말만 쓴다. 장면 라벨(label)뿐 아니라 장소 유형(venue_type: 카페·식당…), 시간대(time_of_day: 아침·정오·오후·저녁·밤), 활동, 문서 유형, 계절, 공휴일, 미디어 유형(셀피·비디오…)도 같은 칸에 쓴다. 인덱스에 없는 낱말을 넣으면 그 조건은 0건이 되고 unknown에 비슷한 후보가 온다 — 그 후보로 고쳐서 다시 찾는다.
- states/cities에는 「강원도」「부산」처럼 줄여 써도 된다. 실제 이름으로 푼 결과가 resolved에 온다. 서울·광역시는 state에도 시 이름이 들어 있다. 일부 해외 도시 이름은 현지 문자라서 해외는 states(도·주 이름)로 찾는 편이 안전하다.
- 장면 라벨은 사진에 찍힌 것, 장소 유형(venue_type)은 찍은 곳의 가게 종류다.
- stats는 조건에 맞는 카드를 한 축으로 센다. 범위를 어디서 좁힐지 고를 때 쓴다. list_vocab은 인덱스에 실제 있는 낱말 목록이다.
- 캡션 필터: cuisine(요리 국적: 한식·양식·일식·중식·동남아·카페·디저트·기타), doc_type(명함·영수증·서류·메뉴판·간판·스크린샷), caption_like(캡션의 장면·물체·요리 이름·키워드 부분 일치, '|'로 OR — 예: "파스타|피자", "아이|가족", "생일|케이크"). 캡션은 모델이 미리보기를 보고 쓴 것이라 틀릴 수 있다.
- vision_labels_any는 공개 Vision 분류기 라벨(한국어 이름 또는 영어 id, 확신도 vision_min_conf 이상, 기본 0.05·최저 0.02). apple_label_min_conf를 주면 labels_*가 사진 앱 장면 확신도 표(문턱 아래 포함)까지 본다. 둘 다 라벨이 0건일 때 넓히는 용도다.
- content_match: 사진 내용 조건(labels_any·vision_labels_any·caption_like)끼리는 기본이 AND(all)다. "any"를 주면 그 셋 중 하나라도 맞으면 된다(날짜·장소·cuisine 같은 다른 조건은 그대로 AND).
- rows의 scene은 캡션 장면 한 줄이다. text_like는 사진 속 글자(사진 앱 색인·우리 OCR·캡션 글자 조각)까지 찾지만 그 글자 내용은 결과에 싣지 않는다.

## 규칙
1. 날짜가 애매하면(「저번주였나 저번달이었나」「언제쯤」「최근에」) 넓은 창으로 시작한다 — 오늘부터 거꾸로 60일 정도. 「작년 여름」처럼 분명하면 그 기간(작년 6~8월)으로 찾는다.
2. 0건이면 조건 하나를 한 단계만 넓혀 다시 찾는다(최대 4회). 순서: ⓐ 라벨 넓히기(비슷한 장면 라벨 추가, 장소 유형과 장면 라벨을 서로 바꿔 보기 → 그래도 0건이면 같은 뜻을 vision_labels_any 저문턱(0.05, 그래도 0이면 0.02)으로 → 그다음 labels_*에 apple_label_min_conf=0.1을 붙여 사진 앱 저문턱 라벨까지) → ⓑ 장소 넓히기(동·시 → 도) → ⓒ 날짜 창 넓히기(두 배씩, 통째로 없애지 않는다). 사용자가 말한 시기(「저번달」 등)는 가장 나중에 넓힌다. 무엇을 넓혔는지 답에 적는다.
3. 최종 후보가 50건을 넘으면 QUESTION 줄로 사용자에게 한 번 되묻는다(예외: 「가장 잘 나온」「글자 많은」처럼 순서가 답인 질문은 그 순서로 정렬해 답하고 되묻지 않는다). 질문은 인덱스에 있는 축으로만 한다: 사람이 나왔는지(얼굴 수), 낮/밤(시간대), 영상인지, 대략 언제(월), 어느 도시, 요리 국적·문서 종류(breakdown의 cuisine·doc_type). breakdown을 보고 실제로 갈리는 축 하나를 골라 선택지마다 건수를 붙인다. 이미 한 번 되물었으면 다시 묻지 말고 가장 그럴듯한 조건으로 답한다.
4. 「양식」「한식」「일식」처럼 요리 국적은 cuisine 필터로 찾는다(라벨이나 장소 유형으로 대신하지 않는다). 「바다」「케이크」「고양이」「등산」처럼 사진에 무엇이 찍혔는지는 장면 라벨(labels_any)과 caption_like를 **content_match="any"로 함께** 넣어 둘 중 하나라도 맞는 사진을 찾는다 — 목표는 빠짐없이 찾기다. 둘을 기본(all)으로 겹치면 양쪽에 다 걸린 사진만 남아 크게 줄어든다. 캡션은 사진에만 있으니 영상까지 보려면 labels_any가 꼭 들어가야 한다. 사진 앱 라벨에 없는 낱말(「가족」 등)은 caption_like로 찾는다. 그 밖의 범주어(「해산물」 등)는 list_vocab(contains=…)로 실제 라벨을 확인한 뒤 라벨 집합으로 푼다.
   장소 유형(카페·식당 등)으로 찾을 때는 그곳에서 찍혔을 장면 라벨(예: 카페 → 커피·커피잔·음료·디저트)도 labels_any에 함께 넣는다. 장소 유형은 가게 정보가 붙은 사진에만 있고 최근 사진엔 아직 안 붙은 경우가 많다(요약의 월별 비율 참고).
5. 카드에 있는 것(날짜·장소·라벨·캡션·건수)만 말한다. 사진 내용을 지어내지 않는다.
6. 도구는 필요한 만큼만 부른다. 같은 조건을 되풀이하지 않는다. RESULT에는 search_cards가 돌려준 result_id만 쓴다 — stats로 좁힌 조건이 답이면 그 조건으로 search_cards를 한 번 더 불러 id를 받는다.

## 답 형식(마지막 응답, 도구 호출 없이 이 네 줄, 한글로)
요약: 후보 N건 — 날짜 범위·주요 장소·주요 라벨을 2~3문장으로. 날짜 범위는 breakdown의 date_min·date_max로 쓴다(by_month_top6는 건수 상위 6개월일 뿐이라 범위로 쓰면 틀린다).
이유: 어떤 조건으로 찾았고 무엇을 넓혔는지 한 줄.
RESULT: <최종 후보의 result_id, 후보가 없으면 none>
QUESTION: <되물을 때만. 선택지와 건수를 넣은 한 줄. 되묻지 않으면 이 줄을 빼라>

오늘: {today} ({weekday}요일), 시간대 {tz}."""


@dataclass
class AskResult:
    query_id: str
    question: str
    model: str
    effort: str
    answer: str = ""
    why: str = ""
    result_id: str = None
    result_fallback: bool = False
    total: int = 0
    uuids: list = field(default_factory=list)
    clarify_question: str = None
    asked: bool = False
    nudged: bool = False      # 모델이 50건 넘게 답하고 안 되물어서 하네스가 한 번 되묻게 했는지
    user_reply: str = None
    stop: str = ""            # answered | asked | turn_limit | budget | query_budget | refusal | max_tokens | error
    error: str = None
    turns: int = 0
    tool_calls: list = field(default_factory=list)
    search_totals: list = field(default_factory=list)
    widen_count: int = 0
    repeated_calls: int = 0   # 같은 도구·같은 인자를 다시 부른 횟수(하네스가 거절)
    han_retry: bool = False   # 답에 한자가 섞여 한 번 다시 쓰게 했는지
    han_left: int = 0         # 최종 답에 남은 한자 수(다시 쓰게 한 뒤에도 남으면 그대로 기록)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    transcript_path: str = None
    legs: int = 1             # 구간 수: 처음 질문 1 + 답한 뒤 이어 묻기(웹의 「더 넓게/좁게」)마다 1
    followups: list = field(default_factory=list)   # 이어 묻기 말(되물음의 답은 user_reply)


def new_query_id() -> str:
    return datetime.now(config.TZ).strftime("q%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def build_system(index: CardIndex, today: datetime = None) -> str:
    today = today or datetime.now(config.TZ)
    return SYSTEM_TEMPLATE.format(summary=index.summary_text(), today=today.strftime("%Y-%m-%d"),
                                  weekday=WEEKDAY_KO[today.weekday()], tz=config.TZ.key)


def _block_dict(b):
    return b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else b


def _messages_dump(messages):
    out = []
    for m in messages:
        c = m["content"]
        out.append({"role": m["role"], "content": c if isinstance(c, str) else [_block_dict(b) for b in c]})
    return out


def _count_images(blocks) -> int:
    """보낸 내용 속 image 블록 수(tool_result 안쪽까지). 질문 단계에서는 0이어야 한다 — 상수로 적지 않고 센다."""
    n = 0
    for b in blocks or []:
        if isinstance(b, dict):
            if b.get("type") == "image":
                n += 1
            if isinstance(b.get("content"), list):
                n += _count_images(b["content"])
    return n


def _recv_chars(content) -> int:
    n = 0
    for b in content:
        if b.type == "text":
            n += len(b.text)
        elif b.type == "tool_use":
            n += len(json.dumps(b.input, ensure_ascii=False))
        elif b.type == "thinking":
            n += len(getattr(b, "thinking", "") or "")
    return n


_ARRAY_KEYS = {k for k, v in _FILTER_PROPS.items() if v.get("type") == "array"}


def _normalize_args(args: dict) -> dict:
    """앞뒤 공백 제거, 배열 자리에 온 문자열은 쉼표·'|'로 나눈다(실측: labels_any를 "a,b"·"a|b" 문자열로 보낸 적 있음)."""
    out = {}
    for k, v in args.items():
        if isinstance(v, str):
            v = v.strip()
            if k in _ARRAY_KEYS:
                v = [x.strip() for x in re.split(r"[,|]", v) if x.strip()]
        out[k] = v
    return out


def run_tool(index: CardIndex, name: str, args: dict):
    """(결과 문자열, is_error). 잘못된 인자는 모델이 고칠 수 있게 오류 문장으로 돌려준다."""
    if name not in _ALLOWED_KEYS:
        return f"알 수 없는 도구: {name}. 쓸 수 있는 도구: {sorted(_ALLOWED_KEYS)}", True
    bad = sorted(set(args) - _ALLOWED_KEYS[name])
    if bad:
        return f"알 수 없는 인자 {bad}. {name}에 쓸 수 있는 인자: {sorted(_ALLOWED_KEYS[name])}", True
    try:
        args = _normalize_args(args)
        if name == "search_cards":
            out = index.search_cards(**args)
        elif name == "stats":
            out = index.stats(args.pop("group_by", None), **args)
        else:
            out = index.list_vocab(**args)
    except (FilterError, TypeError, ValueError) as e:
        return f"입력 오류: {e}", True
    return json.dumps(out, ensure_ascii=False, separators=(",", ":")), False


_TAG_LINE = re.compile(r"^\s*\**\s*(RESULT|QUESTION)\s*\**\s*[:：]\s*(.*?)\s*$")
_LABEL_LINE = re.compile(r"^\s*\**\s*([^\s:：*]{1,7})\s*\**\s*[:：]\s*(.*?)\s*$")
# 첫 줄의 라벨이 이 집합에 정확히 속할 때만 「요약」 표제로 보고 뗀다 — 「약국:」 같은 본문은 건드리지 않는다.
# 요/要 × 약/約/约 의 한자 섞임 + 摘要 + Summary.
SUMMARY_LABELS = {a + b for a in "요要" for b in "약約约"} | {"摘要", "summary"}


def parse_final(text: str):
    """→ (본문, 이유, RESULT 값(없으면 None, 'none'이면 'none'), QUESTION 또는 None)

    본문의 「요약:」「이유:」 표제는 코드가 붙인다 — 모델이 표제를 「要약:」처럼 한자로 섞어 써도
    첫 줄의 짧은 표제는 떼고 다시 붙인다."""
    rid = q = None
    why, summary = [], []
    for line in text.splitlines():
        m = _TAG_LINE.match(line)
        if m:
            if m.group(1) == "RESULT" and rid is None and m.group(2):
                rid = m.group(2).split()[0]
            elif m.group(1) == "QUESTION" and q is None and m.group(2):
                q = m.group(2)
            continue
        m = _LABEL_LINE.match(line)
        label = m.group(1) if m else ""
        if m and (label == "이유" or (HAN.search(label) and "由" in label)):
            why.append(m.group(2).lstrip("* "))
            continue
        if not summary and m and label.lower() in SUMMARY_LABELS:
            summary.append(m.group(2).lstrip("* "))
            continue
        if line.strip():
            summary.append(line.strip())
    why_text = " ".join(x for x in why if x)
    body = "요약: " + "\n".join(summary).strip() if summary else ""
    if why_text:
        body = (body + "\n" if body else "") + "이유: " + why_text
    rid = rid.strip("`*<>").lower() if rid else None
    return body, why_text, rid, q


DONT_KNOW = "모르겠어요. 지금 후보로 보여 주세요."
# 이어 갈 수 있는 멈춤 — 대화 끝에 사용자 말을 덧붙여도 되는 상태. refusal·max_tokens(잘린 도구 호출이 남을 수 있음)·
# error·budget(누적 상한)은 이어 가지 않고 새로 묻는다.
RESUMABLE = ("answered", "asked", "turn_limit", "query_budget")


@dataclass
class Conversation:
    """멈춘 대화(메모리). 명령줄(ask)과 웹 화면(web/app.py)이 같은 루프를 돈다: Asker.start → (필요하면) Asker.resume.

    되물음의 답은 같은 질문의 연장이고(호출 수·질문당 예산을 이어서 센다), 답한 뒤 이어 묻기는 새 구간이다
    (호출 수·질문당 예산을 새로 센다). 결과 묶음(result_id)은 대화가 끝날 때까지 index.results에 남는다."""
    res: AskResult
    system_text: str
    system: list
    messages: list
    calls: list = field(default_factory=list)
    seen_calls: dict = field(default_factory=dict)
    last_nonempty_rid: str = None
    prev_sent: int = 0
    leg_turn0: int = 0         # 지금 구간이 시작될 때의 호출 수
    leg_cost0: float = 0.0     # 지금 구간이 시작될 때의 비용

    @property
    def resumable(self) -> bool:
        return self.res.stop in RESUMABLE


class Asker:
    def __init__(self, index: CardIndex, model: str = None, effort: str = None, client=None,
                 interactive: bool = False, input_fn=input, print_fn=print, max_turns: int = None):
        self.index = index
        self.model = model or config.MODEL
        self.effort = effort or config.EFFORT
        self.interactive = interactive
        self.input_fn = input_fn
        self.print_fn = print_fn
        self.max_turns = max_turns or config.MAX_TURNS
        if client is None:
            import anthropic
            client = anthropic.Anthropic(timeout=180.0, max_retries=2)
        self.client = client

    def ask(self, question: str, query_id: str = None) -> AskResult:
        """명령줄용. 되물으면 interactive일 때 input_fn으로 답을 받아 같은 대화를 이어 간다(아니면 되물은 자리에서 멈춤)."""
        conv = self.start(question, query_id)
        if conv.res.stop == "asked" and self.interactive:
            self.print_fn(f"\n{conv.res.answer}\n\n되물음: {conv.res.clarify_question}")
            reply = self.input_fn("답 (엔터=모름): ").strip() or DONT_KNOW
            self.resume(conv, reply)
        return conv.res

    def start(self, question: str, query_id: str = None) -> Conversation:
        """새 질문. 되물으면 stop=asked로 멈춘 대화를 돌려준다 — 답은 resume으로."""
        res = AskResult(query_id=query_id or new_query_id(), question=question,
                        model=self.model, effort=self.effort)
        self.index.reset_results()
        system_text = build_system(self.index)
        conv = Conversation(res=res, system_text=system_text,
                            system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
                            messages=[{"role": "user", "content": question}])
        self._run(conv)
        return conv

    def resume(self, conv: Conversation, reply: str) -> Conversation:
        """멈춘 대화에 사용자 말을 덧붙여 이어 간다.
        - stop=asked: 되물음의 답(같은 질문의 연장).
        - answered·turn_limit·query_budget: 이어 묻기(새 구간) — 호출 수·질문당 예산을 새로 세고,
          한 번 되물을 기회·50건 넘을 때의 자동 되묻기·한자 다시 쓰기도 새로 준다."""
        res = conv.res
        if not conv.resumable:
            raise ValueError(f"이어 갈 수 없는 대화(stop={res.stop})")
        if res.stop == "asked":
            res.user_reply = reply
        else:
            res.legs += 1
            res.followups.append(reply)
            conv.leg_turn0, conv.leg_cost0 = res.turns, res.cost_usd
            res.asked = res.nudged = res.han_retry = False
            res.clarify_question = res.user_reply = None
            res.answer = res.why = ""
            res.result_id, res.result_fallback = None, False
        res.stop, res.error = "", None
        conv.messages.append({"role": "user", "content": reply})
        self._run(conv)
        return conv

    def _run(self, conv: Conversation) -> None:
        """모델이 답하거나(answered) 되묻거나(asked) 상한·오류로 멈출 때까지 tool-use 루프를 돈다."""
        res = conv.res
        t0 = time.perf_counter()
        try:
            while True:
                if res.turns - conv.leg_turn0 >= self.max_turns:
                    res.stop = "turn_limit"
                    break
                spent = llm_log.spent_total()
                if spent + 0.05 > config.BUDGET_USD:
                    res.stop, res.error = "budget", f"누적 ${spent:.3f} — 상한 ${config.BUDGET_USD} 근접"
                    break
                if res.cost_usd - conv.leg_cost0 >= config.QUERY_BUDGET_USD:
                    res.stop = "query_budget"
                    break
                dumped = _messages_dump(conv.messages)
                sent_payload = json.dumps({"system": conv.system_text, "tools": TOOLS, "messages": dumped},
                                          ensure_ascii=False)
                resp = self.client.beta.messages.create(
                    model=self.model, max_tokens=MAX_TOKENS, system=conv.system, tools=TOOLS,
                    messages=conv.messages, thinking={"type": "adaptive"},
                    output_config={"effort": self.effort}, cache_control={"type": "ephemeral"},
                    betas=BETAS, fallbacks="default",
                )
                res.turns += 1
                u = resp.usage
                cost = llm_log.usage_cost(u, resp.model)
                res.cost_usd += cost
                res.input_tokens += u.input_tokens or 0
                res.output_tokens += u.output_tokens or 0
                res.cache_read_tokens += u.cache_read_input_tokens or 0
                res.cache_write_tokens += u.cache_creation_input_tokens or 0
                tool_uses = [b for b in resp.content if b.type == "tool_use"]
                record = {
                    "ts": llm_log.now_iso(), "provider": "anthropic", "query_id": res.query_id, "turn": res.turns,
                    "model": self.model, "served_model": resp.model, "effort": self.effort,
                    "sent_chars": len(sent_payload), "sent_chars_new": len(sent_payload) - conv.prev_sent,
                    "recv_chars": _recv_chars(resp.content),
                    "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                    "cache_read_input_tokens": u.cache_read_input_tokens,
                    "cache_creation_input_tokens": u.cache_creation_input_tokens,
                    "cost_usd": round(cost, 6), "stop_reason": resp.stop_reason,
                    "tools": [b.name for b in tool_uses], "request_id": getattr(resp, "_request_id", None),
                    "images_sent": _count_images(dumped),
                }
                llm_log.append(record)
                conv.calls.append(record)
                conv.prev_sent = len(sent_payload)
                conv.messages.append({"role": "assistant", "content": resp.content})

                if resp.stop_reason == "tool_use":
                    results = []
                    for b in tool_uses:
                        key = (b.name, json.dumps(b.input or {}, sort_keys=True, ensure_ascii=False))
                        res.tool_calls.append(b.name)
                        if key in conv.seen_calls:
                            res.repeated_calls += 1
                            results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                            "content": f"같은 도구·같은 인자로 이미 불렀다(그때 결과 요약: {conv.seen_calls[key]}). "
                                                       "빠진 조건이 있으면 인자를 바꿔 다시 부르고, 아니면 답을 써라."})
                            continue
                        out, is_err = run_tool(self.index, b.name, b.input or {})
                        if b.name == "search_cards" and not is_err:
                            total = json.loads(out)["total"]
                            if res.search_totals and res.search_totals[-1] == 0:
                                res.widen_count += 1
                            res.search_totals.append(total)
                            if total:
                                conv.last_nonempty_rid = json.loads(out)["result_id"]
                        conv.seen_calls[key] = ("오류" if is_err else
                                                (f"result_id {json.loads(out)['result_id']}, total {json.loads(out)['total']}"
                                                 if b.name == "search_cards" else "위와 같음"))
                        results.append({"type": "tool_result", "tool_use_id": b.id, "content": out,
                                        **({"is_error": True} if is_err else {})})
                    conv.messages.append({"role": "user", "content": results})
                    continue
                if resp.stop_reason == "pause_turn":
                    continue
                if resp.stop_reason == "refusal":
                    res.stop = "refusal"
                    det = getattr(resp, "stop_details", None)
                    res.error = f"refusal: {getattr(det, 'category', None)}"
                    break
                if resp.stop_reason == "max_tokens":
                    res.stop, res.error = "max_tokens", "응답이 max_tokens에서 잘림"
                    break

                text = "\n".join(b.text for b in resp.content if b.type == "text")
                body, why, rid, clarify = parse_final(text)
                han = HAN.findall(body + why + (clarify or ""))
                if han and not res.han_retry:
                    res.han_retry = True
                    conv.messages.append({"role": "user", "content": HAN_NUDGE.format(
                        chars="".join(dict.fromkeys(han))[:10])})
                    continue
                res.han_left = len(han)
                res.answer, res.why = body, why
                res.result_id, res.result_fallback = None, False
                if rid in self.index.results:
                    res.result_id = rid
                elif rid != "none" and conv.last_nonempty_rid:
                    # RESULT 줄이 없거나 없는 id를 적었으면 마지막으로 0건이 아니었던 검색을 쓴다(기록에 표시).
                    res.result_id, res.result_fallback = conv.last_nonempty_rid, True
                if clarify and not res.asked:
                    # 한 구간에 한 번만 멈춰 되묻는다. 답은 resume(명령줄은 ask가 input_fn으로 받아 넘긴다).
                    res.asked, res.clarify_question = True, clarify
                    res.stop = "asked"
                    break
                if not clarify and not res.asked and not res.nudged and res.result_id:
                    rs = self.index.results[res.result_id]
                    sort = rs["filters"].get("sort", "date_desc")
                    if rs["total"] > ASK_THRESHOLD and sort in ("date_desc", "date_asc"):
                        # 순서가 답인 질문(미학·글자 수 정렬)은 빼고, 사양 ③을 하네스가 한 번 보장한다.
                        res.nudged = True
                        conv.messages.append({"role": "user", "content": NUDGE.format(n=rs["total"])})
                        continue
                res.stop = "answered"
                break
        except Exception as e:  # noqa: BLE001 — 원인은 기록에 그대로 남기고 다시 올린다
            res.stop, res.error = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            if (res.result_id is None and res.stop in ("turn_limit", "query_budget", "budget")
                    and conv.last_nonempty_rid):
                res.result_id, res.result_fallback = conv.last_nonempty_rid, True
            res.uuids = list(self.index.results[res.result_id]["uuids"]) if res.result_id else []
            res.total = len(res.uuids)
            res.seconds = round(res.seconds + time.perf_counter() - t0, 1)   # 사람이 답을 고르는 시간은 빼고 센다
            res.transcript_path = str(self._write_transcript(conv))

    def _write_transcript(self, conv: Conversation):
        """대화 원문 전체를 logs/transcripts/<질의 id>.json 에 (이어 갈 때마다 같은 파일에 통째로) 쓴다."""
        res = conv.res
        config.TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
        path = config.TRANSCRIPT_DIR / f"{res.query_id}.json"
        data = {"result": {k: v for k, v in asdict(res).items() if k != "uuids"},
                "final_uuids": res.uuids, "system": conv.system_text, "tools": TOOLS,
                "messages": _messages_dump(conv.messages), "calls": conv.calls,
                "result_sets": {rid: {"total": r["total"], "filters": r["filters"]}
                                for rid, r in self.index.results.items()}}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return path
