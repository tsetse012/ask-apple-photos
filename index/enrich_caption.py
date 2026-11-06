#!/usr/bin/env python3
"""클라우드 VLM 캡션: 사진 미리보기 → captions 표 (Google Gemini).

- ⚠️ 미리보기 사진(대부분 360×480 JPEG)이 Google로 나간다. --send-previews 없이는 사진을 보내지 않고
  무엇이 어디로 나가는지 안내한 뒤 종료 코드 2로 끝난다. --report·--reparse는 네트워크를 쓰지 않는다.
- 요청 시도마다 logs/llm.jsonl 에 provider=gemini · images_sent=1 · uuid · 이미지 sha1·바이트 수를 남기고,
  모델 응답 원문은 logs/captions_raw.jsonl 에 남긴다.
- 대상: 사진(영상 제외), 미리보기 있는 것. 숨김은 기본 제외(--no-skip-hidden 이면 포함), 스크린샷은 기본 포함
  (--skip-screenshots 이면 제외). captions에 이미 있는 uuid는 건너뛴다(재개).
  caption_failures 중 output(응답 문제)은 --retry-failed 일 때만 다시, environment(네트워크·429·5xx)·
  request(4xx — 모델 이름·키·형식을 고친 뒤)는 매번 다시. 4xx가 연달아 3번 나면 그 자리에서 멈춘다.
- 동시 요청 4. 누적 상한(--budget, 기본 $6)을 넘기 전에 새 요청을 멈춘다. 1,000장마다 누적 비용·실패 수 출력.
  멈추려면 logs/caption.stop 파일을 만든다.
- 요청·원문·비용 기록(logs/llm.jsonl)을 남기지 못하면 누적 지출을 잃은 채 유료 요청을 이어가게 되므로
  그 자리에서 새 제출을 멈추고 종료 코드 5로 끝난다(사진 한 장의 응답 오류와 구분한다).

사용:
  export GEMINI_API_KEY=...
  .venv/bin/python index/enrich_caption.py --report                     # 통계만(API 호출 0)
  .venv/bin/python index/enrich_caption.py --send-previews --limit 20   # 시험
  .venv/bin/python index/enrich_caption.py --send-previews              # 전부(재개 가능)
  .venv/bin/python index/enrich_caption.py --reparse                    # 저장된 원문으로 정리 값 다시 만들기(API 호출 0)
"""
import argparse
import json
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from photo_ask import caption, config, llm_log  # noqa: E402
from photo_ask.cards_schema import open_index_rw  # noqa: E402

STOP_FILE = config.LOG_DIR / "caption.stop"
EST_DEFAULT = 0.001    # 첫 성공 전 1장 추정 비용(USD) — 실측 평균이 생기면 그것을 쓴다
REQUEST_FAIL_STOP = 3  # 요청·설정 오류(4xx)가 연달아 이만큼 나면 멈춘다 — 거절될 요청에 사진을 계속 보내지 않으려고


class LedgerWriteError(RuntimeError):
    """요청·원문·비용 기록을 남기지 못했다. 기록 없이 유료 요청을 계속하면 누적 지출을 잃으므로
    사진 한 장의 응답 오류(계속 처리)와 달리 배치를 세우는 치명 오류로 다룬다."""


class Ledger:
    """요청 기록·원문 기록·누적 비용을 한 곳에서(여러 스레드가 부른다)."""

    def __init__(self, spent: float, raw_log: Path = None):
        self.lock = threading.Lock()
        self.spent = spent
        self.calls = 0
        self.ok_costs = []
        self.raw_log = raw_log or config.CAPTION_RAW_LOG

    def log(self, rec: dict, raw):
        with self.lock:
            try:
                llm_log.append(rec)
            except Exception as e:  # noqa: BLE001 — 기록 실패는 원인과 무관하게 치명(비용을 잃는다)
                raise LedgerWriteError(f"요청 기록 실패({config.LLM_LOG}): {e}") from e
            self.spent += float(rec.get("cost_usd") or 0)
            self.calls += 1
            if raw is not None:
                try:
                    llm_log.append({"ts": rec["ts"], "query_id": rec["query_id"], "uuid": rec["uuid"],
                                    "attempt": rec["attempt"], "model": rec.get("served_model") or rec["model"],
                                    "finish_reason": rec.get("finish_reason"), "error": rec.get("error"),
                                    "raw_text": raw}, self.raw_log)
                except Exception as e:  # noqa: BLE001 — 원문 기록 실패도 치명
                    raise LedgerWriteError(f"응답 원문 기록 실패({self.raw_log}): {e}") from e

    def estimate(self) -> float:
        with self.lock:
            recent = self.ok_costs[-200:]
        return max(statistics.mean(recent), 0.00005) if recent else EST_DEFAULT


def targets(conn, retry_failed=False, skip_hidden=True, skip_screenshots=False):
    """캡션 대상 사진. 숨김은 기본 제외, 스크린샷은 옵션."""
    sql = ("SELECT c.uuid, c.preview_path FROM cards c WHERE c.is_video=0 AND c.preview_path IS NOT NULL "
           "AND NOT EXISTS (SELECT 1 FROM captions p WHERE p.uuid=c.uuid)")
    if skip_hidden:
        sql += " AND coalesce(c.hidden, 0) = 0"
    if skip_screenshots:
        sql += " AND coalesce(c.screenshot, 0) = 0"
    if not retry_failed:
        sql += " AND NOT EXISTS (SELECT 1 FROM caption_failures f WHERE f.uuid=c.uuid AND f.error_kind='output')"
    return conn.execute(sql + " ORDER BY c.date_local DESC").fetchall()


def save_success(conn, o: caption.CaptionOutcome, path: str):
    cap = o.caption
    conn.execute(
        "INSERT INTO captions(uuid, model, json, created_at, prompt_version, source_path, raw_text, scene_ko, "
        "food_present, cuisine, doc_type, people_count, text_snippet, search_blob) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(uuid) DO UPDATE SET model=excluded.model, json=excluded.json, created_at=excluded.created_at, "
        "prompt_version=excluded.prompt_version, source_path=excluded.source_path, raw_text=excluded.raw_text, "
        "scene_ko=excluded.scene_ko, food_present=excluded.food_present, cuisine=excluded.cuisine, "
        "doc_type=excluded.doc_type, people_count=excluded.people_count, text_snippet=excluded.text_snippet, "
        "search_blob=excluded.search_blob",
        (o.uuid, o.served_model, json.dumps(cap, ensure_ascii=False), llm_log.now_iso(), caption.PROMPT_VERSION,
         path, o.raw_text, cap["scene_ko"], int(cap["food"]["present"]), cap["food"]["cuisine"], cap["doc_type"],
         cap["people_count"], cap["text_snippet"], caption.search_blob(cap)))
    conn.execute("DELETE FROM caption_failures WHERE uuid=?", (o.uuid,))


def save_failure(conn, o: caption.CaptionOutcome, model: str):
    conn.execute(
        "INSERT INTO caption_failures(uuid, model, error_kind, error, attempts, raw_text, last_at) VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(uuid) DO UPDATE SET model=excluded.model, error_kind=excluded.error_kind, error=excluded.error, "
        "attempts=caption_failures.attempts + excluded.attempts, raw_text=excluded.raw_text, last_at=excluded.last_at",
        (o.uuid, model, o.error_kind, (o.error or "")[:500], o.attempts, o.raw_text, llm_log.now_iso()))


def run(conn, items, *, model, budget, concurrency, caption_fn, run_id, progress_every=1000, ledger=None,
        print_fn=print, stop_file=STOP_FILE) -> dict:
    """items [(uuid, 미리보기 경로)]를 캡션. caption_fn(uuid, 이미지 바이트, log_fn) → CaptionOutcome.
    예산·정지 파일을 매 제출 전에 보고, 결과에 오류가 오면 새 제출을 멈춘다. → 실행 통계."""
    ledger = ledger or Ledger(llm_log.spent_total(provider="gemini"))
    spent_start = ledger.spent
    stats = Counter()
    stop_reason = None
    ledger_error = None
    request_fail_streak = 0
    t0 = time.perf_counter()
    it = iter(items)
    pending = {}
    paths = dict(items)

    def progress(tag=""):
        el = time.perf_counter() - t0
        print_fn(f"[{tag or '진행'}] 처리 {stats['done']:,}/{len(items):,} · 성공 {stats['ok']:,} · 실패 {stats['fail']:,}"
                 f"(output {stats['fail_output']}·environment {stats['fail_environment']}·request {stats['fail_request']})"
                 f" · 요청 {ledger.calls:,}회 · "
                 f"이번 ${ledger.spent - spent_start:.4f} · 누적 ${ledger.spent:.4f}/${budget} · {el:.0f}초", flush=True)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        while True:
            while len(pending) < concurrency and stop_reason is None:
                if stop_file and Path(stop_file).exists():
                    stop_reason = f"정지 파일 {stop_file}"
                    break
                est = ledger.estimate()
                if ledger.spent + (len(pending) + 1) * est * 1.5 > budget:
                    stop_reason = f"예산 상한 근접(누적 ${ledger.spent:.4f}, 장당 추정 ${est:.5f}, 상한 ${budget})"
                    break
                nxt = next(it, None)
                if nxt is None:
                    break
                uuid, path = nxt
                try:
                    image = Path(path).read_bytes()
                except OSError as e:
                    o = caption.CaptionOutcome(uuid=uuid, ok=False, error_kind="environment",
                                               error=f"미리보기 읽기 실패: {e}")
                    save_failure(conn, o, model)
                    stats["done"] += 1
                    stats["fail"] += 1
                    stats["fail_environment"] += 1
                    continue
                fut = ex.submit(caption_fn, uuid, image, ledger.log)
                pending[fut] = uuid
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                uuid = pending.pop(fut)
                try:
                    o = fut.result()
                except LedgerWriteError as e:
                    # 기록·원문·누적 비용을 남기지 못했다 → 누적 지출을 모른 채 다음 유료 요청을 보내면 안 된다.
                    # 사진 한 장의 실패로 남기지 않고 새 제출을 멈춘다(main이 0이 아닌 코드로 끝난다).
                    ledger_error = str(e)
                    stop_reason = f"기록 실패 — {e}"
                    continue
                except Exception as e:  # noqa: BLE001 — 한 장의 예외가 배치 전체를 멈추지 않게 그 사진의 실패로 남긴다
                    o = caption.CaptionOutcome(uuid=uuid, ok=False, error_kind="environment",
                                               error=f"{type(e).__name__}: {str(e)[:300]}")
                stats["done"] += 1
                if o.ok:
                    save_success(conn, o, paths[o.uuid])
                    stats["ok"] += 1
                    with ledger.lock:
                        ledger.ok_costs.append(o.cost_usd)
                else:
                    save_failure(conn, o, model)
                    stats["fail"] += 1
                    stats[f"fail_{o.error_kind}"] += 1
                if not o.ok and o.error_kind == "request":
                    request_fail_streak += 1
                    if request_fail_streak >= REQUEST_FAIL_STOP and stop_reason is None:
                        stop_reason = f"요청·설정 오류 {request_fail_streak}회 연속 — {(o.error or '')[:160]}"
                else:
                    request_fail_streak = 0
                stats["attempts"] += o.attempts
                if stats["done"] % 50 == 0:
                    conn.commit()
                if progress_every and stats["done"] % progress_every == 0:
                    progress()
    conn.commit()
    progress("끝")
    return {"stop_reason": stop_reason, "ledger_error": ledger_error,
            "seconds": round(time.perf_counter() - t0, 1),
            "spent_run": round(ledger.spent - spent_start, 6), "spent_total": round(ledger.spent, 6),
            "calls": ledger.calls, **stats}


def reparse(conn) -> dict:
    """저장된 raw_text로 json·검색용 값을 다시 만든다(정리 규칙을 바꿨을 때, API 호출 0)."""
    n = Counter()
    for uuid, raw, model, path in conn.execute("SELECT uuid, raw_text, model, source_path FROM captions").fetchall():
        try:
            cap = caption.parse_caption(raw)
        except caption.CaptionParseError as e:
            n["now_invalid"] += 1
            print(f"  다시 읽기 실패 {uuid[:8]}: {e}", file=sys.stderr)
            continue
        o = caption.CaptionOutcome(uuid=uuid, ok=True, caption=cap, raw_text=raw, served_model=model)
        save_success(conn, o, path)
        n["updated"] += 1
    conn.commit()
    return dict(n)


def report(conn) -> str:
    q = lambda sql, *p: conn.execute(sql, p).fetchall()  # noqa: E731
    out = []
    (n_photos,), = q("SELECT COUNT(*) FROM cards WHERE is_video=0 AND preview_path IS NOT NULL")
    (n_cap,), = q("SELECT COUNT(*) FROM captions")
    (n_cap_hidden,), = q("SELECT COUNT(*) FROM captions p JOIN cards c ON c.uuid=p.uuid WHERE c.hidden=1")
    fails = q("SELECT error_kind, COUNT(*) FROM caption_failures GROUP BY 1")
    out.append(f"캡션: {n_cap:,}/{n_photos:,}장(사진, 숨김 {n_cap_hidden} 포함) · 실패로 남은 것 "
               + (" · ".join(f"{k} {v}" for k, v in fails) or "0"))
    errs = q("SELECT substr(error, 1, 50), COUNT(*) FROM caption_failures GROUP BY 1 ORDER BY 2 DESC LIMIT 5")
    if errs:
        out.append("  실패 사유 상위: " + " · ".join(f"{e} {n}" for e, n in errs))
    out.append("  cuisine 분포: " + " · ".join(f"{k} {v}" for k, v in q(
        "SELECT cuisine, COUNT(*) FROM captions GROUP BY 1 ORDER BY 2 DESC")))
    out.append("  doc_type 분포: " + " · ".join(f"{k} {v}" for k, v in q(
        "SELECT doc_type, COUNT(*) FROM captions GROUP BY 1 ORDER BY 2 DESC")))
    out.append("  사람 수: " + " · ".join(f"{k} {v}" for k, v in q(
        "SELECT CASE WHEN people_count=0 THEN '0' WHEN people_count<=2 THEN '1-2' WHEN people_count<=5 THEN '3-5' "
        "ELSE '6+' END b, COUNT(*) FROM captions GROUP BY b ORDER BY b")))
    io = Counter(json.loads(j)["indoor_outdoor"] for (j,) in q("SELECT json FROM captions"))
    out.append("  실내·실외: " + " · ".join(f"{k} {v}" for k, v in io.most_common()))
    # 독립 신호와의 일치(정확도 아님 — 서로 다른 두 판정기가 같은 말을 하는 비율)
    rows = q("SELECT p.food_present, EXISTS(SELECT 1 FROM card_tags t WHERE t.uuid=p.uuid AND t.kind='label' "
             "AND t.tag='음식'), p.doc_type='스크린샷', c.screenshot, p.people_count>0, coalesce(c.face_count,0)>0 "
             "FROM captions p JOIN cards c ON c.uuid=p.uuid")
    if rows:
        def agree(i, j, name_a, name_b):
            tab = Counter((bool(r[i]), bool(r[j])) for r in rows)
            return (f"{name_a}×{name_b}: 둘 다 {tab[(True, True)]} · 캡션만 {tab[(True, False)]} · "
                    f"상대만 {tab[(False, True)]} · 둘 다 아님 {tab[(False, False)]}")
        out.append("  독립 신호 대조 — " + agree(0, 1, "캡션 음식", "사진 앱 라벨 음식"))
        out.append("  독립 신호 대조 — " + agree(2, 3, "캡션 스크린샷", "사진 앱 스크린샷 표시"))
        out.append("  독립 신호 대조 — " + agree(4, 5, "캡션 사람>0", "사진 앱 얼굴>0"))
    out.append("  모델별 캡션: " + " · ".join(f"{m} {n:,}" for m, n in q(
        "SELECT model, COUNT(*) FROM captions GROUP BY 1 ORDER BY 2 DESC")))
    recs = [r for r in llm_log.read_all() if r.get("provider") == "gemini" and r.get("step") == "caption"]
    out += _gemini_report(recs)
    return "\n".join(out)


def _gemini_report(recs) -> list:
    out = []
    if recs:
        ok = [r for r in recs if r.get("http_status") == 200 and not r.get("error")]
        cost = sum(float(r.get("cost_usd") or 0) for r in recs)
        photos = len({r["uuid"] for r in recs})
        out.append(f"  [gemini] 요청 기록: {len(recs):,}회(images_sent 합 {sum(r.get('images_sent', 0) for r in recs):,}, "
                   f"서로 다른 사진 {photos:,}장) · HTTP "
                   + " · ".join(f"{k} {v}" for k, v in Counter(r.get('http_status') for r in recs).most_common())
                   + f" · 비용 합 ${cost:.6f} (공식 단가로 추정)")
        errs = Counter(" ".join((r.get("error") or "").split())[:60] for r in recs if r.get("error"))
        if errs:
            out.append("  [gemini] 실패 시도 사유: " + " · ".join(f"{k} {v}" for k, v in errs.most_common(5)))
        if ok:
            med = lambda k: statistics.median([r.get(k) or 0 for r in ok])  # noqa: E731
            img = f"(이미지 {med('image_tokens')})" if any(r.get("image_tokens") for r in ok) else "(이미지 토큰 내역 없음)"
            out.append(f"  [gemini] 성공 요청 중앙값: 입력 {med('input_tokens')}토큰{img} · 출력 "
                       f"{med('output_tokens')} · 사고 {med('thoughts_tokens')} · {round(med('seconds'), 2)}초 · "
                       f"${statistics.mean([float(r['cost_usd']) for r in ok]):.7f}/회(평균)")
            out.append("  [gemini] 응답 모델: " + " · ".join(
                f"{k} {v}" for k, v in Counter(r.get('served_model') for r in ok).items()))
    return out


def make_post():
    import requests
    local = threading.local()

    def post(url, **kw):
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return local.s.post(url, **kw)
    return post


def make_caption_fn(model: str, run_id: str):
    """사진 1장 캡션 함수: (uuid, 이미지 바이트, log_fn) → CaptionOutcome."""
    post = make_post()
    return lambda uuid, image, log_fn: caption.caption_one(post, uuid, image, model=model, run_id=run_id,
                                                           log_fn=log_fn)


def exit_code(res: dict) -> int:
    """run() 결과 → 종료 코드. 기록·비용 실패는 5(치명), 예산·연속 요청 오류 정지는 4, 정지 파일은 0."""
    if res.get("ledger_error"):
        return 5
    return 0 if not res.get("stop_reason") or res["stop_reason"].startswith("정지 파일") else 4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(config.INDEX_DB))
    ap.add_argument("--model", default=config.CAPTION_MODEL)
    ap.add_argument("--budget", type=float, default=config.CAPTION_BUDGET_USD,
                    help="캡션 누적 상한(USD, logs/llm.jsonl 의 provider=gemini 합계)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retry-failed", action="store_true", help="응답 문제로 실패한 사진도 다시")
    ap.add_argument("--skip-hidden", action=argparse.BooleanOptionalAction, default=True,
                    help="사진 앱 「숨김」 사진은 클라우드로 보내지 않는다(기본 켜짐, 보내려면 --no-skip-hidden)")
    ap.add_argument("--skip-screenshots", action=argparse.BooleanOptionalAction, default=False,
                    help="스크린샷은 클라우드로 보내지 않는다(기본 꺼짐)")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--report", action="store_true", help="통계만(API 호출 0)")
    ap.add_argument("--reparse", action="store_true", help="저장된 원문으로 정리 값 다시 만들기(API 호출 0)")
    ap.add_argument("--send-previews", action="store_true",
                    help="사진 미리보기를 실제로 Google에 보낸다(없으면 안내만 하고 종료 코드 2)")
    args = ap.parse_args()

    conn = open_index_rw(Path(args.index))
    if args.report:
        print(report(conn))
        return 0
    if args.reparse:
        print(reparse(conn))
        print(report(conn))
        return 0
    items = targets(conn, args.retry_failed, skip_hidden=args.skip_hidden, skip_screenshots=args.skip_screenshots)
    if args.limit:
        items = items[: args.limit]
    if not args.send_previews:
        print("이 단계는 사진 미리보기(JPEG)를 Google Gemini API로 보냅니다.")
        print(f"  - 모델: {args.model}")
        print(f"  - 대상: {len(items):,}장 (숨김 {'제외' if args.skip_hidden else '포함'}, "
              f"스크린샷 {'제외' if args.skip_screenshots else '포함'})")
        print(f"  - 함께 보내는 것: 장면 설명을 요청하는 고정 지시문 + 사진 1장. 저장 위치: {config.LLM_LOG}, "
              f"{config.CAPTION_RAW_LOG}")
        print("  - 누적 상한: " + f"${args.budget} (logs/llm.jsonl 의 gemini 비용 합)")
        print("  - 무료 등급 키는 입력이 제품 개선에 쓰일 수 있습니다. 유료 등급 키 사용을 권장합니다.")
        print("실제로 보내려면 --send-previews 를 붙여 다시 실행하세요. "
              "통계만 보려면 --report, 저장된 원문으로 다시 정리하려면 --reparse 를 쓰세요.")
        return 2
    if not items:
        print("보낼 사진이 없습니다(이미 캡션 있음 또는 대상 없음).")
        return 0
    run_id = datetime.now(config.TZ).strftime("cap%Y%m%d-%H%M%S")
    spent = llm_log.spent_total(provider="gemini")
    print(f"[시작] {run_id} · 모델 {args.model} · 대상 {len(items):,}장 · 동시 {args.concurrency} · "
          f"캡션 누적 ${spent:.4f}/상한 ${args.budget} · 정지 파일 {STOP_FILE}", flush=True)
    res = run(conn, items, model=args.model, budget=args.budget, concurrency=args.concurrency,
              caption_fn=make_caption_fn(args.model, run_id), run_id=run_id,
              progress_every=args.progress_every, ledger=Ledger(spent))
    print(f"[결과] {json.dumps(res, ensure_ascii=False)}")
    print(report(conn))
    conn.close()
    return exit_code(res)


if __name__ == "__main__":
    sys.exit(main())
