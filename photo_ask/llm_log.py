"""LLM 호출 기록(logs/llm.jsonl)과 비용 계산·예산 확인.

한 줄 = API 호출 1회. 전체 대화 원문은 logs/transcripts/<질의 id>.json 에 따로 남는다.
기록 쓰기가 실패하면 예외를 그대로 올린다(조용히 넘기지 않는다).
"""
import json
import threading
from datetime import datetime
from pathlib import Path

from . import config

_APPEND_LOCK = threading.Lock()   # 캡션 작업은 여러 스레드가 같은 파일에 덧붙인다


def price_for(model: str) -> dict:
    for key in sorted(config.PRICES, key=len, reverse=True):
        if model == key or model.startswith(key + "-"):
            return config.PRICES[key]
    return config.UNKNOWN_MODEL_PRICE


def usage_cost(usage, model: str) -> float:
    p = price_for(model)
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    detail = getattr(usage, "cache_creation", None)
    write_1h = (getattr(detail, "ephemeral_1h_input_tokens", 0) or 0) if detail else 0
    write_5m = max(cache_write - write_1h, 0)
    return (inp * p["input"] + out * p["output"] + cache_read * p["cache_read"]
            + write_5m * p["cache_write_5m"] + write_1h * p["cache_write_1h"]) / 1_000_000


def append(record: dict, path: Path = None) -> None:
    path = Path(path or config.LLM_LOG)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _APPEND_LOCK, open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


def read_all(path: Path = None) -> list:
    path = Path(path or config.LLM_LOG)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    # 기록이 깨졌으면 지출을 과소계산할 수 있으니 멈춘다.
                    raise RuntimeError(f"{path}:{n} 기록 줄을 읽을 수 없음 — 예산 계산 불가: {e}") from e
    return out


def spent_total(path: Path = None, provider="anthropic") -> float:
    """provider별 누적 추정 비용. provider 필드가 없는 줄은 Anthropic 호출이다.
    provider = 문자열 하나 · 여러 개(tuple/list/set) · None(전부).

    gemini·anthropic 모두 공식 가격표로 곱한 추정이다. 예산 상한 비교에는 쓰되 청구액으로 읽지 않는다."""
    want = None if provider is None else ({provider} if isinstance(provider, str) else set(provider))
    return sum(float(r.get("cost_usd") or 0) for r in read_all(path)
               if want is None or r.get("provider", "anthropic") in want)


def now_iso() -> str:
    return datetime.now(config.TZ).isoformat(timespec="seconds")
