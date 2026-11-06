#!/usr/bin/env python3
"""웹 화면용 미리보기 캐시 동기화: 보관함 미리보기 → index/previews/<uuid>.jpeg (APFS 복제, 디스크 거의 0).

웹 서버는 보관함 안 파일을 열면 권한 문제로 멈출 수 있어 이 캐시만 읽는다 — photo_ask/preview_cache.py.
보관함을 읽을 수 있는 문맥(터미널에서 직접 실행)에서 돌린다. index/build_index.py 끝에서도 자동으로 돈다.

  .venv/bin/python index/sync_previews.py              # 증분(바뀐 것만 복제, 빠진 것 삭제). 숨김 사진 제외
  .venv/bin/python index/sync_previews.py --dry-run    # 할 일만 센다
  .venv/bin/python index/sync_previews.py --include-hidden
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photo_ask import config, preview_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(config.INDEX_DB))
    ap.add_argument("--dir", default=str(config.PREVIEW_CACHE_DIR))
    ap.add_argument("--include-hidden", action="store_true", help="숨김 사진도 캐시에 넣는다(기본 제외)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--copy", action="store_true", help="복제(clonefile) 대신 보통 복사")
    args = ap.parse_args()
    t0 = time.perf_counter()
    counts = preview_cache.sync(Path(args.index), Path(args.dir), include_hidden=args.include_hidden,
                                dry_run=args.dry_run, clone=not args.copy)
    print(("[dry-run] " if args.dry_run else "") + "미리보기 캐시: "
          + ", ".join(f"{k}={v}" for k, v in counts.items()) + f" · {time.perf_counter() - t0:.1f}s")
    print(f"캐시: {args.dir} (지금 {preview_cache.cached_count(Path(args.dir))}장)")
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
