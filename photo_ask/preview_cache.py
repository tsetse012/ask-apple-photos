"""웹 화면용 미리보기 캐시: 보관함 미리보기 JPEG → index/previews/<uuid>.jpeg (APFS 복제).

왜 캐시인가: 백그라운드로 띄운 웹 서버(파이썬)가 사진 보관함(…/Photos Library.photoslibrary) 안 파일을 열면
macOS 파일 접근 권한에 막혀 **에러 없이 멈출 수 있다**(open()이 한없이 반환하지 않는다). 그래서 보관함을 읽을 수
있는 문맥(터미널에서 직접 실행)에서 이 동기화를 돌려 두고, 웹 서버는 이 캐시만 읽는다. 동기화는
index/build_index.py 끝에서 자동으로 돌고, 따로는 index/sync_previews.py.

- 복제(clonefile)는 같은 APFS 볼륨에서 데이터 블록을 공유한다 → 디스크를 거의 더 쓰지 않는다. 안 되면 보통 복사.
- 보관함은 읽기만 한다: 원본은 stat과 clonefile(원본을 바꾸지 않음)·읽기 복사만.
- 숨김 사진은 기본 제외(include_hidden=True로 포함) — 웹 화면은 숨김 사진을 보여 주지 않는다(캡션 단계와 같은 기본값).
- manifest.json = {uuid: [원본 경로, 크기, mtime_ns]} — 원본이 바뀌면 다시 복제하고, 인덱스에서 빠진 uuid는 지운다.
- ⚠️ 권한 없는 문맥(launchd 같은 백그라운드 실행)에서 sync를 돌리면 원본 stat에서 멈출 수 있다 — 터미널에서 직접 돌린다.
"""
import ctypes
import ctypes.util
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path

from . import config

# 카드 uuid 모양(영숫자와 '-'만 — 점·슬래시가 없어 경로 조작이 안 된다). 실제 값은 대문자 16진 UUID.
UUID_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z-]{0,63}")
MANIFEST_NAME = "manifest.json"
_libc = None


def cache_dir() -> Path:
    return Path(config.PREVIEW_CACHE_DIR)


def cache_file(uuid: str, root: Path = None) -> Path:
    """uuid → 캐시 파일 경로. uuid 모양이 아니면 ValueError(경로 조작 방지 — 경로는 uuid로만 만든다)."""
    if not isinstance(uuid, str) or not UUID_RE.fullmatch(uuid):
        raise ValueError("uuid 모양이 아님")
    return Path(root or cache_dir()) / f"{uuid}.jpeg"


def _clone(src: str, dst: str) -> bool:
    """APFS clonefile(2). 성공 True. 함수가 없거나 실패하면(다른 볼륨 등) False — 호출 쪽이 보통 복사로 넘어간다."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    fn = getattr(_libc, "clonefile", None)
    if fn is None:
        return False
    return fn(os.fsencode(src), os.fsencode(dst), ctypes.c_int(0)) == 0


def _read_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}   # 깨졌으면 전부 다시 대조한다(복제라 비용이 작다)


def sync(index_db: Path = None, root: Path = None, include_hidden: bool = False, dry_run: bool = False,
         clone: bool = True) -> dict:
    """인덱스의 preview_path → 캐시. 건수 dict를 돌려준다(dry_run이면 할 일만 센다)."""
    index_db = Path(index_db or config.INDEX_DB)
    root = Path(root or cache_dir())
    conn = sqlite3.connect(f"file:{index_db.resolve()}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT uuid, preview_path, coalesce(hidden, 0) FROM cards").fetchall()
    finally:
        conn.close()
    counts = dict(cards=len(rows), wanted=0, unchanged=0, cloned=0, copied=0, removed=0,
                  hidden_skipped=0, no_preview=0, missing_source=0, errors=0)
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(root, 0o700)
    manifest_path = root / MANIFEST_NAME
    manifest = _read_manifest(manifest_path)
    new_manifest, wanted, error_samples = {}, set(), []
    for uuid, src, hidden in rows:
        if not isinstance(uuid, str) or not UUID_RE.fullmatch(uuid):
            counts["errors"] += 1
            continue
        if not src:
            counts["no_preview"] += 1
            continue
        if hidden and not include_hidden:
            counts["hidden_skipped"] += 1
            continue
        dst = cache_file(uuid, root)
        try:
            st = os.stat(src)
        except FileNotFoundError:
            counts["missing_source"] += 1
            if dst.is_file() and uuid in manifest:      # 원본이 잠시 없으면 예전 캐시는 남긴다
                wanted.add(uuid)
                new_manifest[uuid] = manifest[uuid]
            continue
        wanted.add(uuid)
        counts["wanted"] += 1
        sig = [src, st.st_size, st.st_mtime_ns]
        if manifest.get(uuid) == sig and dst.is_file() and dst.stat().st_size == st.st_size:
            counts["unchanged"] += 1
            new_manifest[uuid] = sig
            continue
        if dry_run:
            counts["cloned"] += 1        # 할 일로 센다
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        try:
            if tmp.exists():
                tmp.unlink()
            if clone and _clone(src, str(tmp)):
                counts["cloned"] += 1
            else:
                shutil.copyfile(src, tmp)
                counts["copied"] += 1
            os.chmod(tmp, 0o600)
            os.replace(tmp, dst)
            new_manifest[uuid] = sig
        except OSError as e:
            counts["errors"] += 1
            if len(error_samples) < 3:
                error_samples.append(f"{type(e).__name__} errno={e.errno}")
            if tmp.exists():
                tmp.unlink()
    if root.is_dir():
        for f in root.glob("*.jpeg"):
            if f.stem not in wanted:       # 인덱스에서 빠졌거나 숨김으로 바뀐 사진
                counts["removed"] += 1
                if not dry_run:
                    f.unlink()
        if not dry_run:
            for f in root.glob("*.jpeg.tmp"):
                f.unlink()
    if not dry_run:
        tmpm = manifest_path.with_name(MANIFEST_NAME + ".tmp")
        tmpm.write_text(json.dumps(new_manifest, ensure_ascii=False), encoding="utf-8")
        os.chmod(tmpm, 0o600)  # 원본 경로·크기 목록 — 캐시 파일과 같은 600
        os.replace(tmpm, manifest_path)
    if error_samples:
        counts["error_samples"] = error_samples
    return counts


def cached_count(root: Path = None) -> int:
    root = Path(root or cache_dir())
    return sum(1 for _ in root.glob("*.jpeg")) if root.is_dir() else 0
