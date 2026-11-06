"""카드 인덱스(SQLite) 스키마. 사진·영상 1개 = cards 1줄."""
import sqlite3
from pathlib import Path

CARD_COLUMNS = [
    # (이름, SQL 타입)
    ("uuid", "TEXT PRIMARY KEY"),
    ("date_local", "TEXT"),        # ISO8601, 설정 시간대(config.TZ)
    ("date_capture", "TEXT"),      # ISO8601, 찍은 곳 시간대
    ("year", "INTEGER"), ("month", "INTEGER"), ("day", "INTEGER"),
    ("weekday", "INTEGER"),        # 0=월 … 6=일 (설정 시간대)
    ("hour", "INTEGER"),           # 설정 시간대
    ("hour_capture", "INTEGER"),   # 찍은 곳 현지 시각
    ("lat", "REAL"), ("lon", "REAL"),
    ("country", "TEXT"), ("state", "TEXT"), ("sub_admin", "TEXT"),
    ("city", "TEXT"), ("district", "TEXT"),
    ("place_names", "TEXT"),       # JSON 목록: 관심 지역·랜드마크·물가 이름 등
    ("neighborhoods", "TEXT"),     # JSON
    ("venues", "TEXT"),            # JSON: 가게·장소 이름
    ("venue_types", "TEXT"),       # JSON: 카페·식당 같은 장소 유형
    ("apple_labels", "TEXT"),      # JSON: 사진 앱 장면 라벨(한국어 그대로)
    ("season", "TEXT"), ("holidays", "TEXT"), ("activities", "TEXT"),
    ("times_of_day", "TEXT"),      # JSON: 아침·정오·오후·저녁·밤
    ("document_types", "TEXT"), ("id_document_types", "TEXT"),
    ("text_found", "INTEGER"),
    ("detected_text", "TEXT"),     # 사진 앱 검색 색인의 인식 글자(로컬 매칭 전용, LLM에 안 보냄)
    ("text_len", "INTEGER"),
    ("media_types", "TEXT"), ("age_groups", "TEXT"), ("sounds", "TEXT"),
    ("human_actions", "TEXT"), ("source_app", "TEXT"),
    ("favorite", "INTEGER"), ("hidden", "INTEGER"), ("screenshot", "INTEGER"),
    ("is_video", "INTEGER"), ("duration_s", "REAL"), ("live_photo", "INTEGER"),
    ("face_count", "INTEGER"), ("aesthetic", "REAL"),
    ("albums", "TEXT"), ("keywords", "TEXT"), ("title", "TEXT"), ("description", "TEXT"),
    ("preview_path", "TEXT"), ("preview_w", "INTEGER"), ("preview_h", "INTEGER"),
    ("original_local", "INTEGER"), ("burst", "INTEGER"),
    ("exif_make", "TEXT"), ("exif_model", "TEXT"),
    ("search_text", "TEXT"),       # 소문자 덩어리: 라벨·태그·장소·앨범·제목·설명 (LIKE 검색용)
    ("card_hash", "TEXT"),
    ("indexed_at", "TEXT"),
]

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS cards (
  {", ".join(f"{n} {t}" for n, t in CARD_COLUMNS)}
);
CREATE INDEX IF NOT EXISTS cards_date ON cards(date_local);
CREATE INDEX IF NOT EXISTS cards_state ON cards(state);
CREATE INDEX IF NOT EXISTS cards_city ON cards(city);

-- 라벨·장소 유형·활동·시간대 등 '고를 수 있는 낱말'을 한 표로. kind로 출처를 구분.
CREATE TABLE IF NOT EXISTS card_tags (
  uuid TEXT NOT NULL,
  tag TEXT NOT NULL,
  kind TEXT NOT NULL,
  PRIMARY KEY (uuid, tag, kind)
);
CREATE INDEX IF NOT EXISTS card_tags_tag ON card_tags(tag);

-- 보조 전문검색. unicode61 토크나이저: trigram은 한국어 2음절 낱말(예: 바다)을 0건으로 놓친다.
CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(uuid UNINDEXED, body, tokenize='unicode61');

CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT);

-- ===== 2단계 보강(enrich). build_index.py --rebuild 는 아래 표를 지우지 않는다(캡션은 돈을 내고 만든 것). =====

-- 사진 앱(Apple) 자체 장면 확신도: Photos.sqlite ZSCENECLASSIFICATION → macOS 분류표로 한국어 이름.
-- 사진 앱 검색 라벨(card_tags kind=label)은 문턱 이상만이라, 그 아래까지 넓힐 때 쓴다. build_index.py가 채운다.
CREATE TABLE IF NOT EXISTS label_scores (
  uuid TEXT NOT NULL,
  label_ko TEXT NOT NULL,
  confidence REAL NOT NULL,
  PRIMARY KEY (uuid, label_ko)
);
CREATE INDEX IF NOT EXISTS label_scores_label ON label_scores(label_ko, confidence);

-- 공개 Vision 분류기(VNClassifyImageRequest) 라벨, 확신도 0.02 이상. enrich_vision.py가 채운다.
CREATE TABLE IF NOT EXISTS vision_labels (
  uuid TEXT NOT NULL,
  label_en TEXT NOT NULL,
  label_ko TEXT,
  confidence REAL NOT NULL,
  PRIMARY KEY (uuid, label_en)
);
CREATE INDEX IF NOT EXISTS vision_labels_ko ON vision_labels(label_ko, confidence);

-- 한국어 OCR(VNRecognizeTextRequest accurate, ko-KR+en-US) 결과. 글자 내용은 로컬 매칭 전용(LLM에 안 보냄).
CREATE TABLE IF NOT EXISTS ocr_text (
  uuid TEXT PRIMARY KEY,
  text TEXT,
  line_count INTEGER,
  mean_conf REAL
);

-- 보강 단계별 처리 상태(재개용). step = vision | ocr. status = ok | error.
CREATE TABLE IF NOT EXISTS enrich_status (
  uuid TEXT NOT NULL,
  step TEXT NOT NULL,
  source_path TEXT,
  status TEXT NOT NULL,
  error TEXT,
  ms REAL,
  run_id TEXT,
  updated_at TEXT,
  PRIMARY KEY (uuid, step)
);

-- 클라우드 VLM 캡션(미리보기 1장 → JSON). json = 정리한 결과, raw_text = 모델 응답 원문.
-- cuisine·doc_type·scene_ko·search_blob 은 json에서 뽑아 둔 검색용 값(--reparse 로 다시 만들 수 있다).
CREATE TABLE IF NOT EXISTS captions (
  uuid TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  prompt_version TEXT,
  source_path TEXT,
  raw_text TEXT,
  scene_ko TEXT,
  food_present INTEGER,
  cuisine TEXT,
  doc_type TEXT,
  people_count INTEGER,
  text_snippet TEXT,
  search_blob TEXT
);
CREATE INDEX IF NOT EXISTS captions_cuisine ON captions(cuisine);
CREATE INDEX IF NOT EXISTS captions_doc_type ON captions(doc_type);

-- 캡션 실패(건너뛴 사진). error_kind = environment(네트워크·429·5xx: 다시 돌리면 재시도) | output(응답 파싱·검증 실패·차단).
CREATE TABLE IF NOT EXISTS caption_failures (
  uuid TEXT PRIMARY KEY,
  model TEXT,
  error_kind TEXT,
  error TEXT,
  attempts INTEGER,
  raw_text TEXT,
  last_at TEXT
);
"""

ENRICH_TABLES = ("label_scores", "vision_labels", "ocr_text", "enrich_status", "captions", "caption_failures")


def open_index_rw(path: Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60)
    conn.executescript(SCHEMA_SQL)
    return conn


def open_index_ro(path: Path, check_same_thread: bool = True) -> sqlite3.Connection:
    """읽기 전용 연결. check_same_thread=False는 웹 서버용 — 세션마다 연결 하나를 잠금으로 한 번에 한 스레드만 쓴다."""
    if not Path(path).exists():
        raise FileNotFoundError(f"인덱스 없음: {path} — 먼저 index/build_index.py 를 실행하세요")
    conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, timeout=60,
                           check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    return conn
