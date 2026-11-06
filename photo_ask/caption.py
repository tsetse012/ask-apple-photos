"""클라우드 VLM 캡션: 미리보기 1장 + 지시문 → Gemini → JSON 한 덩이.

- Google Gemini generateContent + responseSchema. 미리보기 JPEG가 이 단계에서만 밖으로 나간다.
- 요청 시도마다 logs/llm.jsonl 에 provider=gemini·images_sent=1·uuid·이미지 sha1·바이트 수를,
  모델 응답 원문은 logs/captions_raw.jsonl 에 남긴다.
- 함정: Gemini는 사고 토큰이 출력 예산을 먹어 본문이 에러 없이 잘린다 → thinkingBudget 0.
  응답이 MAX_TOKENS에서 잘리면 상한을 두 배로 올려 다시 요청한다.
"""
import base64
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field

from . import config, llm_log

PROMPT_VERSION = "cap-v1"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

CUISINES = ["한식", "양식", "일식", "중식", "동남아", "카페·디저트", "기타", "없음"]
DOC_TYPES = ["명함", "영수증", "서류", "메뉴판", "간판", "스크린샷", "없음"]
INDOOR = ["실내", "실외", "알수없음"]
TIMES = ["아침", "낮", "저녁", "밤", "알수없음"]
LIMITS = {"objects_ko": 8, "keywords_ko": 10, "dishes_ko": 6}
SNIPPET_MAX = 40
SCENE_MAX = 80

INSTRUCTION = """너는 개인 사진첩 검색용 색인을 만든다. 사진 한 장을 보고 정해진 JSON 하나만 낸다. 모든 값은 한국어로(화면에 보이는 글자·고유명사는 보이는 그대로).
- scene_ko: 장면 한 줄(40자 안팎). 무엇이 어디서 어떤 상태인지. 사진에 없는 것은 쓰지 않는다.
- objects_ko: 눈에 띄는 물체·대상 최대 8개(짧은 명사).
- food: 음식이나 음료가 사진의 주된 대상이면 present=true.
  cuisine은 하나만: 한식 | 양식(파스타·피자·스테이크·버거·샐러드·브런치·빵 위주 식사 등 서양 요리) | 일식 | 중식 | 동남아(태국·베트남 등) | 카페·디저트(커피·음료·케이크·디저트·빵이 주인공) | 기타 | 없음(음식이 주 대상이 아님).
  dishes_ko: 알아볼 수 있는 요리·음료 이름(최대 6개, 모르면 빈 목록).
- people_count: 보이는 사람 수(없으면 0, 많으면 대략).
- indoor_outdoor: 실내 | 실외 | 알수없음
- time_of_day_guess: 아침 | 낮 | 저녁 | 밤 | 알수없음 (빛으로 추정)
- text_visible: 읽을 수 있는 글자가 있으면 true. text_snippet: 그중 가장 중요한 글자 40자 이내(없으면 빈 문자열).
- doc_type: 명함 | 영수증 | 서류 | 메뉴판 | 간판 | 스크린샷 | 없음 — 그 물건이 사진의 주된 대상일 때만, 아니면 없음.
- keywords_ko: 나중에 이 사진을 찾을 때 칠 만한 검색어 최대 10개(장면·장소 종류·행사·계절·분위기·음식 종류 등).
사람의 이름·신원·관계는 추측하지 않는다."""
USER_TEXT = "이 사진의 색인 JSON."

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scene_ko": {"type": "STRING"},
        "objects_ko": {"type": "ARRAY", "items": {"type": "STRING"}, "maxItems": LIMITS["objects_ko"]},
        "food": {"type": "OBJECT", "properties": {
            "present": {"type": "BOOLEAN"},
            "cuisine": {"type": "STRING", "enum": CUISINES},
            "dishes_ko": {"type": "ARRAY", "items": {"type": "STRING"}, "maxItems": LIMITS["dishes_ko"]},
        }, "required": ["present", "cuisine", "dishes_ko"], "propertyOrdering": ["present", "cuisine", "dishes_ko"]},
        "people_count": {"type": "INTEGER"},
        "indoor_outdoor": {"type": "STRING", "enum": INDOOR},
        "time_of_day_guess": {"type": "STRING", "enum": TIMES},
        "text_visible": {"type": "BOOLEAN"},
        "text_snippet": {"type": "STRING"},
        "doc_type": {"type": "STRING", "enum": DOC_TYPES},
        "keywords_ko": {"type": "ARRAY", "items": {"type": "STRING"}, "maxItems": LIMITS["keywords_ko"]},
    },
    "required": ["scene_ko", "objects_ko", "food", "people_count", "indoor_outdoor", "time_of_day_guess",
                 "text_visible", "text_snippet", "doc_type", "keywords_ko"],
    "propertyOrdering": ["scene_ko", "objects_ko", "food", "people_count", "indoor_outdoor", "time_of_day_guess",
                         "text_visible", "text_snippet", "doc_type", "keywords_ko"],
}
MAX_OUTPUT_TOKENS = 1024
BLOCK_FINISH = {"SAFETY", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "BLOCKLIST", "SPII", "RECITATION",
                "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "IMAGE_OTHER"}


class CaptionParseError(ValueError):
    """모델 응답이 약속한 JSON이 아님(재시도 대상, 끝내 실패하면 caption_failures에 output으로 남김)."""


def build_request(image: bytes, max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict:
    return {
        "systemInstruction": {"parts": [{"text": INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(image).decode("ascii")}},
            {"text": USER_TEXT}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }


def _str_list(v, key, limit):
    if not isinstance(v, list):
        raise CaptionParseError(f"{key}가 목록이 아님")
    out = []
    for x in v:
        x = str(x).strip()
        if x and x not in out:
            out.append(x)
    return out[:limit]


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def parse_caption(text: str) -> dict:
    """모델 응답 글 → 정리한 캡션 dict. 약속을 크게 어기면 CaptionParseError.
    코드 블록(```json … ```)으로 감싼 응답은 껍데기만 벗긴다."""
    if isinstance(text, str):
        m = _FENCE.match(text)
        if m:
            text = m.group(1)
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        raise CaptionParseError(f"JSON 아님: {e}") from e
    if not isinstance(d, dict):
        raise CaptionParseError("JSON 객체가 아님")
    missing = [k for k in SCHEMA["required"] if k not in d]
    if missing:
        raise CaptionParseError(f"빠진 키: {missing}")
    food = d["food"]
    if not isinstance(food, dict) or not {"present", "cuisine", "dishes_ko"} <= set(food):
        raise CaptionParseError("food 형식이 다름")
    cuisine = food["cuisine"]
    if cuisine not in CUISINES:
        raise CaptionParseError(f"cuisine 값이 목록 밖: {cuisine!r}")
    if d["doc_type"] not in DOC_TYPES:
        raise CaptionParseError(f"doc_type 값이 목록 밖: {d['doc_type']!r}")
    present = bool(food["present"])
    # 정리 규칙: 음식이 주 대상이 아니면 국적은 없음, 주 대상인데 없음이면 기타.
    if not present:
        cuisine = "없음"
    elif cuisine == "없음":
        cuisine = "기타"
    try:
        people = max(0, min(int(d["people_count"]), 999))
    except (TypeError, ValueError) as e:
        raise CaptionParseError(f"people_count가 정수가 아님: {d['people_count']!r}") from e
    scene = " ".join(str(d["scene_ko"]).split())[:SCENE_MAX]
    if not scene:
        raise CaptionParseError("scene_ko가 비었음")
    return {
        "scene_ko": scene,
        "objects_ko": _str_list(d["objects_ko"], "objects_ko", LIMITS["objects_ko"]),
        "food": {"present": present, "cuisine": cuisine,
                 "dishes_ko": _str_list(food["dishes_ko"], "dishes_ko", LIMITS["dishes_ko"])},
        "people_count": people,
        "indoor_outdoor": d["indoor_outdoor"] if d["indoor_outdoor"] in INDOOR else "알수없음",
        "time_of_day_guess": d["time_of_day_guess"] if d["time_of_day_guess"] in TIMES else "알수없음",
        "text_visible": bool(d["text_visible"]),
        "text_snippet": " ".join(str(d["text_snippet"] or "").split())[:SNIPPET_MAX],
        "doc_type": d["doc_type"],
        "keywords_ko": _str_list(d["keywords_ko"], "keywords_ko", LIMITS["keywords_ko"]),
    }


def search_blob(cap: dict) -> str:
    """caption_like 검색 대상: 장면·물체·요리·키워드(소문자)."""
    parts = [cap["scene_ko"], *cap["objects_ko"], *cap["food"]["dishes_ko"], *cap["keywords_ko"]]
    return " ".join(parts).lower()


def gemini_price(model: str) -> dict:
    for key in sorted(config.GEMINI_PRICES, key=len, reverse=True):
        if model == key or model.startswith(key + "-"):
            return config.GEMINI_PRICES[key]
    return config.GEMINI_UNKNOWN_PRICE


def usage_cost(usage: dict, model: str) -> float:
    p = gemini_price(model)
    inp = usage.get("promptTokenCount") or 0
    out = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
    return (inp * p["input"] + out * p["output"]) / 1_000_000


@dataclass
class CaptionOutcome:
    uuid: str
    ok: bool
    caption: dict = None
    raw_text: str = None
    served_model: str = None
    error_kind: str = None      # environment(네트워크·429·5xx) | request(4xx: 모델 이름·키·형식) | output(응답 파싱·차단)
    error: str = None
    attempts: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    records: list = field(default_factory=list)


def _retry_delay(resp) -> float:
    """429 응답의 RetryInfo(예: "17s") 또는 Retry-After 헤더 → 초. 없으면 None."""
    try:
        for det in resp.json().get("error", {}).get("details", []):
            delay = det.get("retryDelay")
            if delay and str(delay).endswith("s"):
                return float(str(delay)[:-1])
    except ValueError:
        pass
    ra = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
    try:
        return float(ra) if ra else None
    except ValueError:
        return None


def _record_usage(rec: dict, out: "CaptionOutcome", data, model: str) -> None:
    """응답에서 읽을 수 있는 사용량을 기록과 합계에 넣는다. 구조 검사보다 먼저 불러 본문이 깨져도 비용을 잃지 않는다."""
    if not isinstance(data, dict):
        return
    served = data.get("modelVersion") or model
    usage = data.get("usageMetadata")
    if not isinstance(usage, dict):
        usage = {}
    try:
        cost = usage_cost(usage, served)
        input_tokens = usage.get("promptTokenCount")
        output_tokens = usage.get("candidatesTokenCount")
        thoughts_tokens = usage.get("thoughtsTokenCount")
    except (TypeError, ValueError, AttributeError):
        cost, input_tokens, output_tokens, thoughts_tokens = 0.0, None, None, None
    try:
        image_tokens = sum(d.get("tokenCount", 0) for d in usage.get("promptTokensDetails") or []
                           if d.get("modality") == "IMAGE")
    except (TypeError, AttributeError):
        image_tokens = None
    rec.update(served_model=served, cost_usd=round(cost, 7), input_tokens=input_tokens,
               output_tokens=output_tokens, thoughts_tokens=thoughts_tokens, image_tokens=image_tokens,
               response_id=data.get("responseId"))
    out.cost_usd += cost


def _response_shape(data):
    """HTTP 200 본문의 구조 검사 → candidates[0].
    JSON이 아니거나 모양이 다르면 ValueError — 부르는 쪽이 그 사진의 실패로 기록하고 배치를 계속한다."""
    if not isinstance(data, dict):
        raise ValueError("JSON 객체가 아님")
    usage = data.get("usageMetadata") or {}
    if not isinstance(usage, dict):
        raise ValueError("usageMetadata가 객체가 아님")
    cands = data.get("candidates")
    if cands is None:
        cands = [{}]
    if not isinstance(cands, list) or not cands:
        raise ValueError("candidates가 비었거나 목록이 아님")
    cand = cands[0]
    if not isinstance(cand, dict):
        raise ValueError("candidates[0]이 객체가 아님")
    if cand.get("finishReason") is not None and not isinstance(cand.get("finishReason"), str):
        raise ValueError("finishReason이 문자열이 아님")
    feedback = data.get("promptFeedback")
    if feedback is not None and not isinstance(feedback, dict):
        raise ValueError("promptFeedback이 객체가 아님")
    content = cand.get("content")
    if content is not None and not isinstance(content, dict):
        raise ValueError("content가 객체가 아님")
    parts = (content or {}).get("parts")
    if parts is not None and (not isinstance(parts, list) or not all(isinstance(p, dict) for p in parts)):
        raise ValueError("content.parts가 객체 목록이 아님")
    return cand


def caption_one(post, uuid: str, image: bytes, *, model: str, run_id: str, log_fn,
                max_retries: int = 2, max_429_waits: int = 8, sleep=time.sleep, timeout: float = 90.0) -> CaptionOutcome:
    """사진 1장 캡션. post(url, headers=, data=, timeout=) → requests.Response 같은 객체.
    재시도: 네트워크·5xx·응답 파싱 실패·MAX_TOKENS는 최대 max_retries번 다시, 429는 따로 기다리며 최대 max_429_waits번.
    요청 시도마다 log_fn(기록 dict, 원문 글)을 부른다(기록과 비용 합산은 호출한 쪽이 한 곳에서)."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY 없음 — 환경변수로 키를 지정하세요")
    url = API_URL.format(model=model)
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    sha1 = hashlib.sha1(image).hexdigest()
    out = CaptionOutcome(uuid=uuid, ok=False)
    failures = waits = 0
    max_tokens = MAX_OUTPUT_TOKENS
    t_start = time.perf_counter()
    while True:
        out.attempts += 1
        body = json.dumps(build_request(image, max_tokens))
        rec = {"ts": llm_log.now_iso(), "provider": "gemini", "step": "caption", "query_id": run_id, "uuid": uuid,
               "attempt": out.attempts, "model": model, "prompt_version": PROMPT_VERSION,
               "images_sent": 1, "image_bytes": len(image), "image_sha1": sha1,
               "sent_chars": len(INSTRUCTION) + len(USER_TEXT), "cost_usd": 0.0}
        t0 = time.perf_counter()
        raw = None
        try:
            resp = post(url, headers=headers, data=body, timeout=timeout)
        except Exception as e:  # noqa: BLE001 — 네트워크 계열(재시도 대상). 원인은 기록에 남긴다
            rec.update(http_status=None, error=f"network: {type(e).__name__}: {str(e)[:200]}",
                       seconds=round(time.perf_counter() - t0, 2))
            log_fn(rec, None)
            out.records.append(rec)
            failures += 1
            out.error_kind, out.error = "environment", rec["error"]
            if failures > max_retries:
                break
            sleep(min(2 ** failures + random.random(), 30))
            continue
        rec["http_status"] = resp.status_code
        rec["seconds"] = round(time.perf_counter() - t0, 2)
        if resp.status_code == 429:
            delay = _retry_delay(resp)
            rec["error"] = f"429 rate limit (retryDelay={delay})"
            log_fn(rec, None)
            out.records.append(rec)
            waits += 1
            out.error_kind, out.error = "environment", rec["error"]
            if waits > max_429_waits:
                break
            sleep(min(delay if delay else 2 ** waits, 90) + random.random())
            continue
        if resp.status_code >= 500 or resp.status_code == 408:
            rec["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            log_fn(rec, None)
            out.records.append(rec)
            failures += 1
            out.error_kind, out.error = "environment", rec["error"]
            if failures > max_retries:
                break
            sleep(min(2 ** failures + random.random(), 30))
            continue
        if resp.status_code != 200:
            # 400·403·404 등 요청·설정의 문제(모델 이름·키·형식) — 같은 요청을 다시 보내도 같다. 고친 뒤 다시 돌리면 재시도된다.
            rec["error"] = f"HTTP {resp.status_code}: {resp.text[:300]}"
            log_fn(rec, None)
            out.records.append(rec)
            out.error_kind, out.error = "request", rec["error"]
            break
        try:
            data = resp.json()
            _record_usage(rec, out, data, model)
            cand = _response_shape(data)
        except (ValueError, TypeError, AttributeError) as e:
            # 200인데 JSON이 아니거나 구조가 다름 — 읽을 수 있었던 사용량의 비용은 이미 합산됐다.
            rec["error"] = f"응답 형식: {str(e)[:200]}"
            log_fn(rec, None)
            out.records.append(rec)
            failures += 1
            out.error_kind, out.error = "output", rec["error"]
            if failures > max_retries:
                break
            continue
        served = rec["served_model"]
        try:
            finish = cand.get("finishReason")
            parts = (cand.get("content") or {}).get("parts") or []
            raw = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (ValueError, TypeError, AttributeError) as e:
            # 사용량은 멀쩡한데 본문 part 모양이 깨진 경우 — 비용은 위에서 이미 기록했다.
            rec["error"] = f"본문 형식: {str(e)[:200]}"
            log_fn(rec, None)
            out.records.append(rec)
            failures += 1
            out.error_kind, out.error = "output", rec["error"]
            if failures > max_retries:
                break
            continue
        rec["finish_reason"] = finish
        rec["recv_chars"] = len(raw)
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block or finish in BLOCK_FINISH:
            rec["error"] = f"blocked: {block or finish}"
            log_fn(rec, raw)
            out.records.append(rec)
            out.error_kind, out.error, out.raw_text = "output", rec["error"], raw
            break
        if finish == "MAX_TOKENS":
            rec["error"] = "MAX_TOKENS(응답 잘림)"
            log_fn(rec, raw)
            out.records.append(rec)
            failures += 1
            out.error_kind, out.error, out.raw_text = "output", rec["error"], raw
            max_tokens = min(max_tokens * 2, 4096)
            if failures > max_retries:
                break
            continue
        try:
            cap = parse_caption(raw)
        except CaptionParseError as e:
            rec["error"] = f"parse: {e}"
            log_fn(rec, raw)
            out.records.append(rec)
            failures += 1
            out.error_kind, out.error, out.raw_text = "output", rec["error"], raw
            if failures > max_retries:
                break
            continue
        log_fn(rec, raw)
        out.records.append(rec)
        out.ok, out.caption, out.raw_text, out.served_model = True, cap, raw, served
        out.error_kind = out.error = None
        break
    out.seconds = round(time.perf_counter() - t_start, 2)
    return out
