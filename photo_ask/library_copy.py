"""사진 보관함 DB를 임시 폴더로 복사한다.

보관함 원본은 읽기만 한다. SQLite 연결·잠금·-shm 갱신이 원본 파일에 닿지 않도록
osxphotos와 우리 쿼리는 전부 이 복사본을 연다(osxphotos 자체는 DB가 잠겨 있지 않으면
원본을 바로 열기 때문에 복사를 우리가 먼저 한다).
"""
import shutil
from pathlib import Path

DB_FILES = ["Photos.sqlite", "photos.db"]
SEARCH_FILES = ["leo.sqlite", "psi.sqlite"]
SUFFIXES = ["", "-wal", "-shm"]


def copy_library_database(library: Path, dest_dir: Path) -> Path:
    """dest_dir/database/ 에 Photos.sqlite·photos.db·search/*.sqlite(+wal·shm)를 복사하고
    복사된 Photos.sqlite 경로를 돌려준다."""
    src = Path(library) / "database"
    if not (src / "Photos.sqlite").exists():
        raise FileNotFoundError(f"Photos.sqlite 없음: {src}")
    dst = Path(dest_dir) / "database"
    (dst / "search").mkdir(parents=True, exist_ok=True)
    for name in DB_FILES:
        for suf in SUFFIXES:
            s = src / f"{name}{suf}"
            if s.exists():
                shutil.copyfile(s, dst / f"{name}{suf}")
    for name in SEARCH_FILES:
        for suf in SUFFIXES:
            s = src / "search" / f"{name}{suf}"
            if s.exists():
                shutil.copyfile(s, dst / "search" / f"{name}{suf}")
    return dst / "Photos.sqlite"
