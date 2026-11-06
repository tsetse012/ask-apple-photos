"""macOS 사진 앱 장면 분류표: 장면 식별자(정수) → 키 → 한국어 이름.

- `PFSceneTaxonomyData_<버전>.bz2`: `n:<식별자>,…,<키>` 줄이 식별자 → 키. 키는 공개 Vision 분류기 라벨과 같은
  영어 식별자(dog·food …)이거나 Q로 시작하는 번호다.
- `scenetaxonomy.loctable`: plist {언어: {키: 이름}}. 공개 Vision 분류기 라벨도 여기 들어 있다.
- macOS 버전마다 분류표 파일 이름의 버전 번호가 다르다. 있는 파일 중 가장 높은 번호를 쓴다.

읽기만 한다. 분류표가 없으면 TaxonomyUnavailable을 올리고, 부르는 쪽(build_index.py)이 경고 후
label_scores 없이 진행한다.
"""
import bz2
import re
import plistlib
from functools import lru_cache
from pathlib import Path

RES_DIR = Path("/System/Library/PrivateFrameworks/PhotosFormats.framework/Versions/A/Resources")
DATA_NAME = re.compile(r"PFSceneTaxonomyData_(\d+)\.bz2")
LOCTABLE = "scenetaxonomy.loctable"


class TaxonomyUnavailable(RuntimeError):
    """장면 분류표 파일이 없거나 형식이 다름 — 호출 쪽이 경고 후 label_scores 없이 진행한다."""


def taxonomy_version(res_dir: Path = RES_DIR) -> int:
    """있는 분류표 중 가장 높은 버전 번호. 없으면 TaxonomyUnavailable."""
    found = []
    for p in Path(res_dir).glob("PFSceneTaxonomyData_*.bz2"):
        m = DATA_NAME.fullmatch(p.name)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        raise TaxonomyUnavailable(f"장면 분류표 없음: {Path(res_dir) / 'PFSceneTaxonomyData_*.bz2'}")
    return max(found)[0]


def taxonomy_path(res_dir: Path = RES_DIR) -> Path:
    version = taxonomy_version(res_dir)
    return Path(res_dir) / f"PFSceneTaxonomyData_{version}.bz2"


@lru_cache(maxsize=None)
def scene_keys(res_dir: Path = RES_DIR) -> dict:
    """{장면 식별자: 키}. 있는 파일 중 가장 높은 버전을 읽는다."""
    path = taxonomy_path(res_dir)
    out = {}
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("n:"):
                parts = line[2:].strip().split(",")
                out[int(parts[0])] = parts[-1]
    if not out:
        raise TaxonomyUnavailable(f"장면 분류표에 n: 줄이 없음: {path}")
    return out


@lru_cache(maxsize=None)
def names(lang: str = "ko", res_dir: Path = RES_DIR) -> dict:
    """{키: 이름} — 키는 scene_keys()의 값 또는 Vision 분류기 라벨(영어 식별자)."""
    path = Path(res_dir) / LOCTABLE
    try:
        with open(path, "rb") as fh:
            table = plistlib.load(fh)
    except (OSError, plistlib.InvalidFileException) as e:
        raise TaxonomyUnavailable(f"이름표를 읽지 못함: {path}") from e
    if lang not in table or not table[lang]:
        raise TaxonomyUnavailable(f"{path}에 '{lang}' 이름표가 없음")
    return table[lang]
