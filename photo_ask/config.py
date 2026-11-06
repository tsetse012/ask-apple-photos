"""경로·모델·가격·예산 설정. 값은 환경변수로 덮어쓸 수 있다."""
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
INDEX_DB = Path(os.environ.get("PHOTO_ASK_INDEX", ROOT / "index" / "cards.sqlite"))
LOG_DIR = Path(os.environ.get("PHOTO_ASK_LOG_DIR", ROOT / "logs"))
LLM_LOG = LOG_DIR / "llm.jsonl"
TRANSCRIPT_DIR = LOG_DIR / "transcripts"
OUT_DIR = ROOT / "out"
# 웹 화면용 미리보기 캐시(보관함 미리보기의 APFS 복제). 웹 서버는 보관함 대신 여기만 읽는다 — photo_ask/preview_cache.py
PREVIEW_CACHE_DIR = Path(os.environ.get("PHOTO_ASK_PREVIEW_CACHE", ROOT / "index" / "previews"))
LIBRARY = Path(os.environ.get(
    "PHOTO_ASK_LIBRARY", Path.home() / "Pictures" / "Photos Library.photoslibrary"))

TZ = ZoneInfo(os.environ.get("PHOTO_ASK_TZ", "Asia/Seoul"))

MODEL = os.environ.get("PHOTO_ASK_MODEL", "claude-sonnet-5-5")
EFFORT = os.environ.get("PHOTO_ASK_EFFORT", "low")
# 전체 누적 상한(logs/llm.jsonl 합계 기준). 넘으면 새 호출을 거부한다.
BUDGET_USD = float(os.environ.get("PHOTO_ASK_BUDGET_USD", "3.0"))
# 질문 하나에 쓸 수 있는 상한. 넘으면 그때까지의 결과로 끝낸다.
QUERY_BUDGET_USD = float(os.environ.get("PHOTO_ASK_QUERY_BUDGET_USD", "0.25"))
MAX_TURNS = int(os.environ.get("PHOTO_ASK_MAX_TURNS", "10"))

# USD per 1M tokens. Anthropic 공개 가격표 기준. Sonnet 5.5 = Sonnet 5 가격.
PRICES = {
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_write_5m": 2.5,
                          "cache_write_1h": 4.0, "cache_read": 0.2},
    "claude-sonnet-5": {"input": 2.0, "output": 10.0, "cache_write_5m": 2.5,
                        "cache_write_1h": 4.0, "cache_read": 0.2},
}
# 모르는 모델(서버 폴백 등)이 응답하면 과소추정하지 않도록 가장 비싼 표값으로 센다.
UNKNOWN_MODEL_PRICE = {"input": 10.0, "output": 50.0, "cache_write_5m": 12.5,
                       "cache_write_1h": 20.0, "cache_read": 1.0}

# ===== 2단계: 클라우드 VLM 캡션(Google Gemini) =====
CAPTION_MODEL = os.environ.get("PHOTO_ASK_CAPTION_MODEL", "gemini-3.1-flash-lite")
# 캡션 누적 상한(logs/llm.jsonl 의 provider=gemini 합계 기준). 넘기 전에 새 요청을 멈춘다.
CAPTION_BUDGET_USD = float(os.environ.get("PHOTO_ASK_CAPTION_BUDGET_USD", "6.0"))
CAPTION_RAW_LOG = LOG_DIR / "captions_raw.jsonl"
# USD per 1M tokens. Gemini API 유료 등급 공개 가격표 기준. 출력 단가에 사고 토큰 포함.
# 암묵 캐시 할인은 계산에 넣지 않는다(과소추정 방지).
GEMINI_PRICES = {
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
}
# 모르는 모델이 응답하면(이름 변경 등) 과소추정하지 않도록 비싼 값으로 센다.
GEMINI_UNKNOWN_PRICE = {"input": 2.50, "output": 15.0}
