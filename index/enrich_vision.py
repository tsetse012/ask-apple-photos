#!/usr/bin/env python3
"""로컬 미리보기 → 공개 Vision 분류 라벨(vision_labels) + 한국어 OCR(ocr_text). 네트워크 0.

- 보관함 파일은 읽기만 한다: 미리보기 JPEG 경로를 Swift 헬퍼(tools/vision_helper)에 넘기고, 헬퍼가 열어 읽는다.
- 헬퍼는 별도 프로세스로 띄운다. 실행 파일이 없거나 소스보다 오래되면 swiftc로 다시 만든다(없으면 안내 후 종료).
- 분류: 미리보기가 있는 모든 카드(사진·영상 대표 프레임, 숨김 포함), 확신도 0.02 이상 라벨을 저장.
  영어 라벨 → 한국어는 macOS 장면 분류표(scenetaxonomy.loctable 'ko'). 분류표가 없으면 경고 한 번 뒤 영어 id를 이름으로 쓴다.
- OCR: 글자가 있을 법한 사진만(사진 앱 색인 text_found · 스크린샷 · 문서/스크린샷/간판/영수증/손글씨 라벨 · 문서 유형 태그 ·
  Vision 문서류 라벨). 미리보기(대부분 360×480)라 작은 글자는 놓친다.
- 재개: enrich_status에 ok로 남은 사진(같은 미리보기 경로)은 건너뛴다. 실패는 error로 남고 다음 실행에서 다시 시도한다.
- ⚠️ OCR은 샌드박스(파일 접근이 제한된 실행 환경) 안에서 첫 장 뒤부터 e5rtError 13으로 실패할 수 있다.
  e5rt 실패가 연달아 나오면 헬퍼를 멈추고 그 사실을 출력한 뒤 종료 코드 3으로 끝낸다(샌드박스 밖에서 다시 실행하면 이어진다).

사용:
  .venv/bin/python index/enrich_vision.py                 # 분류 → OCR
  .venv/bin/python index/enrich_vision.py --step classify
  .venv/bin/python index/enrich_vision.py --step ocr
  .venv/bin/python index/enrich_vision.py --report        # 통계만(처리 안 함)
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from photo_ask import config, scene_taxonomy  # noqa: E402
from photo_ask.cards_schema import open_index_rw  # noqa: E402

HELPER_SRC = ROOT / "tools" / "vision_helper.swift"
HELPER_BIN = ROOT / "tools" / "vision_helper"
MIN_CONF = 0.02
# OCR 대상 고르기
OCR_APPLE_LABELS = ("문서", "스크린샷", "간판", "영수증", "손글씨")
OCR_VISION_LABELS = ("document", "screenshot", "sign", "receipt", "handwriting")
OCR_VISION_CONF = 0.2
E5RT_STREAK_STOP = 5
COMMIT_EVERY = 200


def ensure_helper() -> Path:
    if HELPER_BIN.exists() and HELPER_BIN.stat().st_mtime >= HELPER_SRC.stat().st_mtime:
        return HELPER_BIN
    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise SystemExit(
            "swiftc를 찾지 못했습니다 — 이 단계는 Xcode 명령줄 도구가 필요합니다.\n"
            "  - 터미널에서 `xcode-select --install` 로 설치한 뒤 다시 실행하세요.\n"
            "  - 이미 설치했다면 `xcode-select -p` 로 경로를 확인하고, `swiftc --version` 이 되는지 보세요."
        )
    print(f"헬퍼 컴파일: swiftc -O {HELPER_SRC.name}", flush=True)
    subprocess.run([swiftc, "-O", str(HELPER_SRC), "-o", str(HELPER_BIN)], check=True)
    return HELPER_BIN


def run_helper(mode: str, items, on_result, env_extra=None) -> dict:
    """items [(uuid, 경로)]를 헬퍼 프로세스 하나로 처리하며 결과 줄마다 on_result(dict)를 부른다.
    on_result가 False를 돌려주면 헬퍼를 멈춘다. → {"meta", "returncode", "stderr_tail", "stopped"}"""
    helper = ensure_helper()
    with tempfile.TemporaryDirectory(prefix="photo-ask-vision-") as tmp:
        listfile = Path(tmp) / "items.tsv"
        listfile.write_text("".join(f"{u}\t{p}\n" for u, p in items), encoding="utf-8")
        errfile = Path(tmp) / "stderr.txt"
        env = {**os.environ, "MIN_CONF": str(MIN_CONF), **(env_extra or {})}
        meta, stopped = None, False
        with open(errfile, "w") as err:
            proc = subprocess.Popen([str(helper), mode, str(listfile)], stdout=subprocess.PIPE,
                                    stderr=err, text=True, env=env)
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "meta" in obj:
                    meta = obj["meta"]
                    continue
                if on_result(obj) is False:
                    stopped = True
                    proc.terminate()
                    break
            rc = proc.wait()
        tail = errfile.read_text(errors="replace").strip().splitlines()[-3:]
    return {"meta": meta, "returncode": rc, "stderr_tail": tail, "stopped": stopped}


def classify_targets(conn, force=False):
    sql = ("SELECT c.uuid, c.preview_path FROM cards c WHERE c.preview_path IS NOT NULL")
    if not force:
        sql += (" AND NOT EXISTS (SELECT 1 FROM enrich_status s WHERE s.uuid=c.uuid AND s.step='vision' "
                "AND s.status='ok' AND s.source_path=c.preview_path)")
    return conn.execute(sql + " ORDER BY c.date_local DESC").fetchall()


def ocr_target_sources(conn) -> dict:
    """{uuid: [대상이 된 이유…]} — 사진만(영상 제외), 숨김 포함."""
    reasons = {}

    def add(sql, params, why):
        for (u,) in conn.execute(sql, params):
            reasons.setdefault(u, []).append(why)

    base = "SELECT c.uuid FROM cards c WHERE c.is_video=0 AND c.preview_path IS NOT NULL AND "
    add(base + "c.text_found=1", (), "apple_text_found")
    add(base + "c.screenshot=1", (), "screenshot_flag")
    q = ",".join("?" * len(OCR_APPLE_LABELS))
    add(base + f"EXISTS (SELECT 1 FROM card_tags t WHERE t.uuid=c.uuid AND t.kind='label' AND t.tag IN ({q}))",
        OCR_APPLE_LABELS, "apple_label")
    add(base + "EXISTS (SELECT 1 FROM card_tags t WHERE t.uuid=c.uuid AND t.kind IN ('document_type','id_document_type'))",
        (), "apple_document_type")
    q = ",".join("?" * len(OCR_VISION_LABELS))
    add(base + f"EXISTS (SELECT 1 FROM vision_labels v WHERE v.uuid=c.uuid AND v.label_en IN ({q}) AND v.confidence >= ?)",
        (*OCR_VISION_LABELS, OCR_VISION_CONF), "vision_label")
    return reasons


def ocr_targets(conn, force=False):
    reasons = ocr_target_sources(conn)
    rows = []
    for u, path, status_path, status in conn.execute(
            "SELECT c.uuid, c.preview_path, s.source_path, s.status FROM cards c LEFT JOIN enrich_status s "
            "ON s.uuid=c.uuid AND s.step='ocr' ORDER BY c.date_local DESC"):
        if u not in reasons:
            continue
        if not force and status == "ok" and status_path == path:
            continue
        rows.append((u, path))
    return rows, reasons


class Writer:
    """헬퍼 결과 줄 → DB. 일정 건수마다 커밋."""

    def __init__(self, conn, step, run_id, paths):
        self.conn, self.step, self.run_id, self.paths = conn, step, run_id, paths
        # 이름표는 분류(Vision 라벨 → 한국어)에만 쓴다. OCR은 이름표 없이 돈다.
        # 분류표가 없는 macOS에서도 로컬 보강 전체가 죽지 않게, 없으면 경고 한 번 뒤 영어 라벨 id를 이름으로 쓴다.
        self.ko, self.ko_fallback = {}, False
        if step == "vision":
            try:
                self.ko = scene_taxonomy.names("ko")
            except scene_taxonomy.TaxonomyUnavailable as e:
                self.ko_fallback = True
                print(f"⚠ {e} — Vision 한국어 이름 없이 영어 라벨 id를 이름으로 저장합니다.", file=sys.stderr)
        self.n = Counter()
        self.e5rt_streak = 0
        self.e5rt_stopped = False
        self.pending = 0

    def status(self, uuid, status, error=None, ms=None):
        self.conn.execute(
            "INSERT INTO enrich_status(uuid, step, source_path, status, error, ms, run_id, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(uuid, step) DO UPDATE SET source_path=excluded.source_path, "
            "status=excluded.status, error=excluded.error, ms=excluded.ms, run_id=excluded.run_id, "
            "updated_at=excluded.updated_at",
            (uuid, self.step, self.paths.get(uuid), status, error, ms, self.run_id,
             datetime.now(config.TZ).isoformat(timespec="seconds")))

    def __call__(self, obj):
        uuid = obj.get("uuid")
        if not uuid:
            self.n["bad_line"] += 1
            return True
        if "error" in obj:
            self.n["error"] += 1
            self.status(uuid, "error", obj["error"][:500])
            if "e5rtError" in obj["error"]:
                self.e5rt_streak += 1
                if self.e5rt_streak >= E5RT_STREAK_STOP:
                    self.e5rt_stopped = True
                    self._tick(force=True)
                    return False
            self._tick()
            return True
        self.e5rt_streak = 0
        if self.step == "vision":
            labels = obj.get("labels") or []
            self.conn.execute("DELETE FROM vision_labels WHERE uuid=?", (uuid,))
            self.conn.executemany(
                "INSERT INTO vision_labels(uuid, label_en, label_ko, confidence) VALUES (?,?,?,?)",
                [(uuid, en, self.ko.get(en) or (en if self.ko_fallback else None), conf) for en, conf in labels])
            self.n["label_rows"] += len(labels)
            self.n["no_ko_name"] += sum(1 for en, _ in labels if not self.ko.get(en))
        else:
            lines = obj.get("lines") or []
            confs = [c for _, c in lines]
            self.conn.execute(
                "INSERT INTO ocr_text(uuid, text, line_count, mean_conf) VALUES (?,?,?,?) ON CONFLICT(uuid) "
                "DO UPDATE SET text=excluded.text, line_count=excluded.line_count, mean_conf=excluded.mean_conf",
                (uuid, "\n".join(t for t, _ in lines), len(lines),
                 round(sum(confs) / len(confs), 4) if confs else None))
        self.n["ok"] += 1
        self.status(uuid, "ok", None, obj.get("ms"))
        self._tick()
        return True

    def _tick(self, force=False):
        self.pending += 1
        if force or self.pending >= COMMIT_EVERY:
            self.conn.commit()
            self.pending = 0
            done = self.n["ok"] + self.n["error"]
            print(f"  [{self.step}] {done}/{len(self.paths)} ok={self.n['ok']} error={self.n['error']}", flush=True)


def run_step(conn, step, items, run_id) -> Writer:
    paths = dict(items)
    w = Writer(conn, step, run_id, paths)
    if not items:
        print(f"[{step}] 할 일 없음(이미 처리됨)")
        return w
    t0 = time.perf_counter()
    info = run_helper("classify" if step == "vision" else "ocr", items, w)
    conn.commit()
    secs = time.perf_counter() - t0
    print(f"[{step}] {len(items)}장 · ok={w.n['ok']} error={w.n['error']} · {secs:.1f}초 · 헬퍼 rc={info['returncode']}"
          f" · revision={info['meta'] and info['meta'].get('classify_revision' if step == 'vision' else 'ocr_revision')}"
          f" · {info['stderr_tail'][-1] if info['stderr_tail'] else ''}")
    if step == "vision":
        print(f"  라벨 행 {w.n['label_rows']:,} · 한국어 이름 없는 라벨 행 {w.n['no_ko_name']}")
    if w.e5rt_stopped:
        print(f"⚠ OCR이 e5rtError 13으로 {E5RT_STREAK_STOP}장 연달아 실패해 멈췄다 — 샌드박스(파일 접근이 제한된 실행 환경) 안에서 "
              "돌린 경우의 알려진 증상(샌드박스 밖 정상). 실패한 사진은 error로 남았고, 샌드박스 밖에서 다시 실행하면 이어서 처리한다.",
              file=sys.stderr)
    return w


def _quantiles(vals, n=10):
    vals = sorted(vals)
    if len(vals) < 2:
        return vals
    return [round(x, 3) for x in statistics.quantiles(vals, n=n)]


def report(conn) -> str:
    q = lambda sql, *p: conn.execute(sql, p).fetchall()  # noqa: E731
    out = []
    # 카드가 0장이면 SUM(...)이 NULL이다 — 합계를 0으로 두고 안내만 한 뒤 정상 종료한다.
    (n_cards, n_prev), = q("SELECT COUNT(*), COALESCE(SUM(preview_path IS NOT NULL), 0) FROM cards")
    if not n_cards:
        return "색인이 비어 있다(카드 0장) — index/build_index.py 를 먼저 실행해야 한다. 지금은 보강할 사진이 없다."
    st = dict(((s, st_), n) for s, st_, n in q(
        "SELECT step, status, COUNT(*) FROM enrich_status GROUP BY step, status"))
    (vl_rows, vl_photos, vl_names), = q("SELECT COUNT(*), COUNT(DISTINCT uuid), COUNT(DISTINCT label_en) FROM vision_labels")
    per = [n for (n,) in q("SELECT COUNT(*) FROM vision_labels GROUP BY uuid")]
    out.append(f"Vision 분류: 미리보기 {n_prev:,}/{n_cards:,}장 중 ok {st.get(('vision', 'ok'), 0):,} · error "
               f"{st.get(('vision', 'error'), 0)} · 라벨 행 {vl_rows:,}(사진 {vl_photos:,}장, 라벨 종류 {vl_names}) · "
               f"사진당 라벨 수(≥{MIN_CONF}) 중앙 {statistics.median(per) if per else 0}")
    for th in (0.05, 0.1, 0.3, 0.5):
        (n,), = q("SELECT COUNT(*) FROM vision_labels WHERE confidence >= ?", th)
        out.append(f"  확신도 ≥{th}: {n:,}행")
    reasons = ocr_target_sources(conn)
    by_reason = Counter(r for rs in reasons.values() for r in rs)
    out.append(f"OCR 대상 {len(reasons):,}장(사진만) — 이유별(겹침 있음): "
               + " · ".join(f"{k} {v}" for k, v in by_reason.most_common()))
    out.append(f"OCR 상태: ok {st.get(('ocr', 'ok'), 0)} · error {st.get(('ocr', 'error'), 0)}")
    errs = Counter(e.split("(")[0][:60] for (e,) in q(
        "SELECT error FROM enrich_status WHERE step='ocr' AND status='error'"))
    if errs:
        out.append("  OCR 오류 종류: " + " · ".join(f"{k} {v}" for k, v in errs.most_common(5)))
    rows = q("SELECT o.uuid, o.line_count, o.mean_conf, length(o.text), c.text_found, coalesce(c.text_len, 0) "
             "FROM ocr_text o JOIN cards c ON c.uuid=o.uuid")
    if rows:
        lc = Counter("0" if r[1] == 0 else "1-2" if r[1] <= 2 else "3-9" if r[1] <= 9 else "10-29" if r[1] <= 29 else "30+"
                     for r in rows)
        out.append("  줄 수 분포: " + " · ".join(f"{k}줄 {lc.get(k, 0)}" for k in ("0", "1-2", "3-9", "10-29", "30+")))
        confs = [r[2] for r in rows if r[2] is not None]
        out.append(f"  사진별 평균 신뢰도 십분위(줄 있는 {len(confs)}장): {_quantiles(confs)}")
        for th in (0.3, 0.5, 0.8):
            out.append(f"    평균 신뢰도 ≥{th}: {sum(1 for c in confs if c >= th)}장")
        (line_n,), = q("SELECT SUM(line_count) FROM ocr_text")
        out.append(f"  전체 줄 {line_n or 0:,} · 글자 {sum(r[3] or 0 for r in rows):,}")
        both = [(r[3] or 0, r[5]) for r in rows if r[4] == 1 and r[5] > 0]
        if both:
            ratios = [ours / apple for ours, apple in both]
            zero = sum(1 for ours, _ in both if ours == 0)
            out.append(f"  사진 앱 색인 글자와 대조(text_found {len(both)}장): 우리 글자 수 / 사진 앱 글자 수 십분위 {_quantiles(ratios)}"
                       f" · 우리가 0자인 사진 {zero}장 · 우리 ≥ 사진 앱의 절반 {sum(1 for x in ratios if x >= 0.5)}장")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(config.INDEX_DB))
    ap.add_argument("--step", choices=["classify", "ocr", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="이미 처리한 사진도 다시")
    ap.add_argument("--limit", type=int, help="이번 실행에서 처리할 최대 장수(시험용)")
    ap.add_argument("--report", action="store_true", help="통계만 출력")
    args = ap.parse_args()

    conn = open_index_rw(Path(args.index))
    if args.report:
        print(report(conn))
        return 0
    run_id = datetime.now(config.TZ).strftime("vis%Y%m%d-%H%M%S")
    rc = 0
    if args.step in ("classify", "all"):
        items = classify_targets(conn, args.force)[: args.limit]
        run_step(conn, "vision", items, run_id)
    if args.step in ("ocr", "all"):
        items, reasons = ocr_targets(conn, args.force)
        items = items[: args.limit]
        print(f"[ocr] 대상 {len(reasons):,}장 중 이번에 처리 {len(items):,}장")
        w = run_step(conn, "ocr", items, run_id)
        if w.e5rt_stopped:
            rc = 3
    print(report(conn))
    conn.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
