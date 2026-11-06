#!/usr/bin/env python3
"""사진 보관함 → 카드 인덱스(index/cards.sqlite).

- 보관함은 읽기만 한다: DB 파일을 임시 폴더로 복사한 뒤 그 복사본을 osxphotos로 연다.
- 보관함이 없으면 경로와 해결 방법을 알려 주고 종료 코드 2로 끝난다.
- 증분: uuid 기준 upsert. 카드 내용 해시가 같으면 건드리지 않고, 보관함에서 사라진 uuid는 지운다.
  단 지울 카드가 기존의 5%를 넘으면(복사가 불완전했을 수 있음) 지우지 않고 경고만 한다(--allow-mass-delete로 허용).
- 사진 앱 자체 장면 확신도(문턱 아래 포함)를 label_scores 표로 매번 새로 쓴다(macOS 장면 분류표로 한국어 이름).
  분류표가 없는 macOS면 경고 후 label_scores 없이 진행한다.
- 끝나면 웹 화면용 미리보기 캐시(index/previews/, APFS 복제)를 맞춘다(--no-preview-cache로 건너뜀) — photo_ask/preview_cache.py
- 사용: .venv/bin/python index/build_index.py [--rebuild] [--allow-mass-delete] [--no-preview-cache]
"""
import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photo_ask import config, scene_taxonomy  # noqa: E402
from photo_ask.cards_schema import CARD_COLUMNS, ENRICH_TABLES, open_index_rw  # noqa: E402
from photo_ask.library_copy import copy_library_database  # noqa: E402

COLS = [n for n, _ in CARD_COLUMNS]
# 해시에서 빼는 값: 미리보기 크기는 파일을 열어야 알 수 있어 따로 채우고, 나머지는 기록용.
NOT_HASHED = {"preview_w", "preview_h", "card_hash", "indexed_at"}
KOREAN_METRO_SUFFIXES = ("특별시", "광역시", "특별자치시")
LABEL_SCORE_FLOOR = 0.02      # label_scores에 넣는 사진 앱 장면 확신도 하한
MASS_DELETE_RATIO = 0.05      # 한 번에 이 비율보다 많이 지워야 하면 멈춘다
SEARCH_DB_FILES = ("psi.sqlite", "leo.sqlite")


def dedupe(items):
    seen, out = set(), []
    for x in items or []:
        if x is None:
            continue
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def jlist(items):
    return json.dumps(dedupe(items), ensure_ascii=False)


def first(items):
    items = dedupe(items)
    return items[0] if items else None


def preview_size(path):
    """미리보기 JPEG 헤더만 읽어 (가로, 세로). 회전 정보가 있으면 반영."""
    from Foundation import NSURL
    from Quartz import CGImageSourceCopyPropertiesAtIndex, CGImageSourceCreateWithURL
    src = CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(path), None)
    if src is None:
        return None, None
    props = CGImageSourceCopyPropertiesAtIndex(src, 0, None)
    if not props:
        return None, None
    w, h = props.get("PixelWidth"), props.get("PixelHeight")
    if w is None or h is None:
        return None, None
    if props.get("Orientation", 1) in (5, 6, 7, 8):
        w, h = h, w
    return int(w), int(h)


def read_asset_extras(photos_sqlite: Path) -> dict:
    """osxphotos가 내주지 않는 값(영상 길이)을 DB 복사본에서 직접 읽는다."""
    conn = sqlite3.connect(f"file:{photos_sqlite}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT ZUUID, ZKIND, ZDURATION FROM ZASSET").fetchall()
    finally:
        conn.close()
    return {u: {"kind": k, "duration": d} for u, k, d in rows if u}


def read_label_scores(photos_sqlite: Path, floor: float = LABEL_SCORE_FLOOR):
    """Photos.sqlite ZSCENECLASSIFICATION(분류 종류 0 = 장면) → ({(uuid, 한국어 이름): 최대 확신도}, 통계).
    여러 식별자가 같은 한국어 이름이면 가장 높은 확신도를 쓴다. 이름이 없는 식별자(사진 앱 내부용)는 센다.
    macOS 장면 분류표가 없으면 예외 대신 빈 표와 unavailable 사유를 돌려준다."""
    try:
        keys = scene_taxonomy.scene_keys()
        ko = scene_taxonomy.names("ko")
    except scene_taxonomy.TaxonomyUnavailable as e:
        return {}, {"scene_rows": 0, "unmapped_rows": 0, "below_floor_rows": 0, "unavailable": str(e)}
    conn = sqlite3.connect(f"file:{photos_sqlite}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT a.ZUUID, s.ZSCENEIDENTIFIER, s.ZCONFIDENCE FROM ZSCENECLASSIFICATION s "
            "JOIN ZADDITIONALASSETATTRIBUTES aa ON s.ZASSETATTRIBUTES = aa.Z_PK "
            "JOIN ZASSET a ON aa.ZASSET = a.Z_PK WHERE s.ZCLASSIFICATIONTYPE = 0").fetchall()
    finally:
        conn.close()
    best = {}
    stats = {"scene_rows": len(rows), "unmapped_rows": 0, "below_floor_rows": 0}
    for uuid, sid, conf in rows:
        name = ko.get(keys[sid]) if sid in keys else None
        if not name or name == "NULL":
            stats["unmapped_rows"] += 1
            continue
        if conf is None or conf < floor:
            stats["below_floor_rows"] += 1
            continue
        k = (uuid, name)
        if conf > best.get(k, -1.0):
            best[k] = conf
    return best, stats


def removal_plan(existing, seen, allow: bool):
    """보관함에서 안 보이는 카드 → (지울 uuid 목록, 막혔나). 기존의 MASS_DELETE_RATIO를 넘게 지워야 하면
    allow 없이는 막는다(사진 앱이 쓰는 중에 복사해 DB가 불완전하면 대량 삭제가 날 수 있어서)."""
    gone = [u for u in existing if u not in seen]
    blocked = bool(existing) and len(gone) > MASS_DELETE_RATIO * len(existing) and not allow
    return gone, blocked


def enrich_orphan_plan(conn, allow: bool):
    """보강 표에 행이 있는 uuid 중 현재 cards에 없는 것. 삭제 전 목록(prior)이 아니라 표 자체를 보므로,
    5% 가드로 남긴 행도 --allow-mass-delete 재실행에서 정리된다.
    가드는 일반 삭제와 같다: 보강 표 uuid 중 이 비율을 넘게 지워야 하면 allow 없이는 막는다.
    → (지울 uuid 목록, 막혔나, 보강 표 uuid 수)"""
    enrich = sorted({u for table in ENRICH_TABLES for (u,) in conn.execute(f"SELECT DISTINCT uuid FROM {table}")})
    card_uuids = {u for (u,) in conn.execute("SELECT uuid FROM cards")}
    gone, blocked = removal_plan(enrich, card_uuids, allow)
    return gone, blocked, len(enrich)


def cleanup_enrich_orphans(conn, allow: bool):
    """--rebuild 로 카드를 새로 만든 뒤, 카드에 없는 uuid의 보강 행을 지운다.
    판정은 enrich_orphan_plan(5% 가드). 막히면 아무것도 지우지 않는다.
    → (대상 uuid 목록, 막혔나, 보강 표 uuid 수)"""
    gone, blocked, n_enrich = enrich_orphan_plan(conn, allow)
    if not blocked:
        for u in gone:
            for table in ENRICH_TABLES:
                conn.execute(f"DELETE FROM {table} WHERE uuid=?", (u,))
    return gone, blocked, n_enrich


def delete_cards(conn, uuids) -> None:
    """카드와 그 uuid의 보강 행(label_scores·vision_labels·ocr_text·enrich_status·captions·caption_failures)을 함께 지운다.
    카드만 지우면 고아 행이 남아 stats·list_vocab 이 센다. cards_fts 는 호출 쪽이 통째로 다시 쓴다."""
    for u in uuids:
        conn.execute("DELETE FROM cards WHERE uuid=?", (u,))
        conn.execute("DELETE FROM card_tags WHERE uuid=?", (u,))
        for table in ENRICH_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE uuid=?", (u,))


class _EmptySearchInfo:
    """검색 색인(psi.sqlite/leo.sqlite)을 못 읽었거나 아주 오래된 보관함일 때 쓰는 빈 값."""
    country = state = city = source = season = None
    text_found = None
    labels = venue_types = activities = times_of_day = holidays = []
    document_types = id_document_types = media_types = age_groups = []
    sounds = human_actions = event_types = week_part = []
    trip = captured_by_me = detected_text = place_names = []
    landmarks = bodies_of_water = areas_of_interest = neighborhoods = venues = []


def card_tags(si) -> list:
    """사진 앱 검색 색인에서 '고를 수 있는 낱말'을 (낱말, 종류)로."""
    out = []
    def add(values, kind):
        for v in dedupe(values if isinstance(values, list) else [values]):
            out.append((v, kind))
    add(si.labels, "label")
    add(si.venue_types, "venue_type")
    add(si.activities, "activity")
    add(si.times_of_day, "time_of_day")
    add(si.season, "season")
    add(si.holidays, "holiday")
    add(si.document_types, "document_type")
    add(si.id_document_types, "id_document_type")
    add(si.media_types, "media_type")
    add(si.age_groups, "age_group")
    add(si.sounds, "sound")
    add(si.human_actions, "human_action")
    add(si.event_types, "event_type")
    add(si.week_part, "week_part")
    add(si.trip, "flag")
    add(si.captured_by_me, "flag")
    add(si.text_found, "flag")
    add(si.source, "source_app")
    return out


def build_card(p, extras: dict):
    d = p.date
    if d.tzinfo is None:
        d = d.replace(tzinfo=config.TZ)
    dk = d.astimezone(config.TZ)
    si = p.search_info or _EmptySearchInfo()
    names = p.place.names if p.place else None

    def pn(field):
        return list(getattr(names, field) or []) if names else []

    country = first(pn("country")) or si.country or None
    state = first(pn("state_province")) or si.state or None
    city = first(pn("city")) or si.city or None
    sub_admin = first(pn("sub_administrative_area"))
    district = first(pn("additional_city_info"))
    # 서울·광역시·세종은 1차 행정구역인데 사진 앱은 state를 비워 둔다 → 시 이름을 state로 쓴다.
    if not state and city and country == "대한민국" and city.endswith(KOREAN_METRO_SUFFIXES):
        state = city

    lat, lon = (p.location or (None, None))
    if lat is not None and lon is not None and lat == 0.0 and lon == 0.0:
        lat = lon = None

    labels = dedupe(si.labels)
    tags = card_tags(si)
    detected = dedupe(si.detected_text)
    place_names = dedupe(list(si.place_names) + pn("area_of_interest") + list(si.landmarks)
                         + list(si.bodies_of_water) + pn("body_of_water") + pn("ocean")
                         + pn("inland_water") + list(si.areas_of_interest))
    neighborhoods = dedupe(si.neighborhoods)
    venues = dedupe(si.venues)
    albums = dedupe(p.albums)
    keywords = dedupe(p.keywords)
    title = p.title or None
    description = p.description or None
    ex = extras.get(p.uuid, {})
    exif = p.exif_info
    make, model = (exif.camera_make or None, exif.camera_model or None) if exif else (None, None)
    derivs = p.path_derivatives or []
    score = p.score.overall if p.score is not None else None

    search_parts = ([t for t, _ in tags] + [country, state, sub_admin, city, district]
                    + neighborhoods + place_names + venues + albums + keywords
                    + [title, description])
    card = {
        "uuid": p.uuid,
        "date_local": dk.isoformat(timespec="seconds"),
        "date_capture": d.isoformat(timespec="seconds"),
        "year": dk.year, "month": dk.month, "day": dk.day,
        "weekday": dk.weekday(), "hour": dk.hour, "hour_capture": d.hour,
        "lat": lat, "lon": lon,
        "country": country, "state": state, "sub_admin": sub_admin,
        "city": city, "district": district,
        "place_names": json.dumps(place_names, ensure_ascii=False),
        "neighborhoods": json.dumps(neighborhoods, ensure_ascii=False),
        "venues": json.dumps(venues, ensure_ascii=False),
        "venue_types": jlist(si.venue_types),
        "apple_labels": json.dumps(labels, ensure_ascii=False),
        "season": si.season or None,
        "holidays": jlist(si.holidays),
        "activities": jlist(si.activities),
        "times_of_day": jlist(si.times_of_day),
        "document_types": jlist(si.document_types),
        "id_document_types": jlist(si.id_document_types),
        "text_found": 1 if si.text_found else 0,
        "detected_text": " ".join(detected) or None,
        "text_len": sum(len(t) for t in detected),
        "media_types": jlist(si.media_types),
        "age_groups": jlist(si.age_groups),
        "sounds": jlist(si.sounds),
        "human_actions": jlist(si.human_actions),
        "source_app": si.source or None,
        "favorite": int(bool(p.favorite)),
        "hidden": int(bool(p.hidden)),
        "screenshot": int(bool(p.screenshot)),
        "is_video": int(bool(p.ismovie)),
        "duration_s": round(float(ex["duration"]), 1) if p.ismovie and ex.get("duration") else None,
        "live_photo": int(bool(p.live_photo)),
        "face_count": len(p.face_info or []),
        "aesthetic": round(score, 4) if score is not None else None,
        "albums": json.dumps(albums, ensure_ascii=False),
        "keywords": json.dumps(keywords, ensure_ascii=False),
        "title": title, "description": description,
        "preview_path": derivs[0] if derivs else None,
        "preview_w": None, "preview_h": None,
        "original_local": int(not p.ismissing),
        "burst": int(bool(p.burst)),
        "exif_make": make, "exif_model": model,
        "search_text": " ".join(dedupe(search_parts)).lower(),
    }
    hashed = {k: v for k, v in card.items() if k not in NOT_HASHED}
    card["card_hash"] = hashlib.sha1(
        json.dumps(hashed, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return card, tags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", default=str(config.LIBRARY))
    ap.add_argument("--index", default=str(config.INDEX_DB))
    ap.add_argument("--rebuild", action="store_true",
                    help="기존 카드를 모두 지우고 새로 만든다(현재 카드에 없는 보강 행은 정리한다)")
    ap.add_argument("--allow-mass-delete", action="store_true",
                    # argparse가 도움말을 % 포매팅하므로 리터럴 %는 %%로 둬야 한다(f-string의 %는 이스케이프가 아님).
                    help=f"보관함에서 사라진 카드가 기존의 {MASS_DELETE_RATIO:.0%}%를 넘어도 지운다")
    ap.add_argument("--no-preview-cache", action="store_true",
                    help="웹 화면용 미리보기 캐시(index/previews/) 동기화를 건너뛴다")
    args = ap.parse_args()

    library = Path(args.library)
    if not library.exists():
        print(f"사진 보관함을 찾지 못했습니다: {library}\n"
              "  - 사진 앱 보관함 경로가 다르면 PHOTO_ASK_LIBRARY 환경변수나 --library 로 지정하세요.\n"
              "  - 예: PHOTO_ASK_LIBRARY=\"$HOME/Pictures/Photos Library.photoslibrary\" "
              ".venv/bin/python index/build_index.py\n", file=sys.stderr)
        return 2
    search_dir = library / "database" / "search"
    if not any((search_dir / name).exists() for name in SEARCH_DB_FILES):
        print(f"⚠ 검색 색인 파일이 없습니다: {search_dir}/({'|'.join(SEARCH_DB_FILES)}) — "
              "장소·라벨·검색어 정보가 비어 있을 수 있습니다. 인덱스는 계속 만듭니다.", file=sys.stderr)

    import osxphotos  # 무거워서 여기서 import

    t0 = time.perf_counter()
    timings = {}
    with tempfile.TemporaryDirectory(prefix="photo-ask-db-") as tmp:
        t = time.perf_counter()
        photos_copy = copy_library_database(Path(args.library), Path(tmp))
        timings["복사"] = time.perf_counter() - t

        t = time.perf_counter()
        db = osxphotos.PhotosDB(dbfile=str(photos_copy), library_path=args.library)
        library_db_dir = (Path(args.library) / "database").resolve()
        for opened in (db._tmp_db, db._dbfile):  # osxphotos가 실제로 연 파일 — 원본이면 즉시 중단
            if Path(opened).resolve().is_relative_to(library_db_dir):
                raise RuntimeError(f"osxphotos가 보관함 원본 DB를 열었다: {opened}")
        photos = db.photos(intrash=False)
        extras = read_asset_extras(photos_copy)
        timings["osxphotos 읽기"] = time.perf_counter() - t

        t = time.perf_counter()
        scores, score_stats = read_label_scores(photos_copy)
        timings["장면 확신도 읽기"] = time.perf_counter() - t

        t = time.perf_counter()
        built = [build_card(p, extras) for p in photos]
        timings["카드 만들기"] = time.perf_counter() - t

    t = time.perf_counter()
    conn = open_index_rw(Path(args.index))
    # --rebuild는 카드를 지우므로, 지우기 전 카드 값을 보존해 증분 판정이 아닌 삭제 판정(removal_plan)에 쓴다.
    # 보강 행 정리는 이 목록이 아니라 보강 표 자체를 본다(enrich_orphan_plan).
    prior = {r[0]: r[1:] for r in conn.execute(
        "SELECT uuid, card_hash, preview_path, preview_w, preview_h FROM cards")}
    if args.rebuild:
        conn.executescript("DELETE FROM cards; DELETE FROM card_tags; DELETE FROM cards_fts;")
    existing = {} if args.rebuild else prior
    now = datetime.now(config.TZ).isoformat(timespec="seconds")
    counts = {"inserted": 0, "updated": 0, "unchanged": 0, "removed": 0, "preview_measured": 0}
    upsert_sql = (f"INSERT INTO cards ({', '.join(COLS)}) VALUES ({', '.join('?' for _ in COLS)}) "
                  f"ON CONFLICT(uuid) DO UPDATE SET "
                  + ", ".join(f"{c}=excluded.{c}" for c in COLS if c != "uuid"))
    seen = set()
    for card, tags in built:
        uuid = card["uuid"]
        seen.add(uuid)
        old = existing.get(uuid)
        if old and old[0] == card["card_hash"] and (old[2] is not None or not card["preview_path"]):
            counts["unchanged"] += 1
            continue
        if old and old[1] == card["preview_path"] and old[2] is not None:
            card["preview_w"], card["preview_h"] = old[2], old[3]
        elif card["preview_path"]:
            card["preview_w"], card["preview_h"] = preview_size(card["preview_path"])
            counts["preview_measured"] += 1
        card["indexed_at"] = now
        conn.execute(upsert_sql, [card[c] for c in COLS])
        conn.execute("DELETE FROM card_tags WHERE uuid=?", (uuid,))
        conn.executemany("INSERT OR IGNORE INTO card_tags(uuid, tag, kind) VALUES (?,?,?)",
                         [(uuid, tag, kind) for tag, kind in tags])
        counts["updated" if old else "inserted"] += 1
    gone, mass_delete_blocked = removal_plan(existing, seen, args.allow_mass_delete)
    if mass_delete_blocked:
        print(f"⚠ 보관함에서 안 보이는 카드 {len(gone)}건(기존 {len(existing)}건의 {len(gone) / len(existing):.1%}) — "
              f"{MASS_DELETE_RATIO:.0%} 초과라 지우지 않음. DB 복사가 불완전했을 수 있다. "
              "정말 지워졌으면 --allow-mass-delete 로 다시 실행.", file=sys.stderr)
    else:
        delete_cards(conn, gone)
        counts["removed"] = len(gone)
    counts["removal_blocked"] = len(gone) if mass_delete_blocked else 0

    orphan_blocked = False
    if args.rebuild:
        # 정리 대상은 prior가 아니라 보강 표에 남은 uuid 중 현재 cards에 없는 것이다 —
        # 가드로 남긴 행을 다음 실행(특히 --allow-mass-delete)이 다시 볼 수 있어야 한다.
        orphan_gone, orphan_blocked, n_enrich = cleanup_enrich_orphans(conn, args.allow_mass_delete)
        counts["enrich_orphans_removed"] = 0 if orphan_blocked else len(orphan_gone)
        if orphan_blocked:
            print(f"⚠ 카드에 없는 사진의 보강 행 {len(orphan_gone)}건(보강 행이 있는 사진 {n_enrich:,}건의 "
                  f"{len(orphan_gone) / n_enrich:.1%}) — {MASS_DELETE_RATIO:.0%} 초과라 지우지 않음. "
                  "정말 지워졌으면 --allow-mass-delete 로 다시 실행.", file=sys.stderr)

    # label_scores: 매번 통째로 새로 쓴다(장면 확신도는 카드 해시에 안 들어가서 증분 판정이 안 된다).
    # 지우기가 막힌 실행(복사 불완전 의심)이나 분류표가 없는 macOS에선 기존 표를 그대로 둔다.
    if mass_delete_blocked or orphan_blocked:
        score_stats["written"] = "건너뜀(지우기 막힘)"
    elif "unavailable" in score_stats:
        score_stats["written"] = "건너뜀(장면 분류표 없음)"
        print(f"⚠ {score_stats['unavailable']} — label_scores 없이 인덱스를 만듭니다.", file=sys.stderr)
    else:
        conn.execute("DELETE FROM label_scores")
        conn.executemany("INSERT INTO label_scores(uuid, label_ko, confidence) VALUES (?,?,?)",
                         [(u, name, round(conf, 4)) for (u, name), conf in scores.items() if u in seen])
        score_stats["written"] = conn.execute("SELECT COUNT(*) FROM label_scores").fetchone()[0]
    changed = counts["inserted"] + counts["updated"] + counts["removed"]
    if changed or args.rebuild:
        conn.execute("DELETE FROM cards_fts")
        conn.execute("INSERT INTO cards_fts(uuid, body) SELECT uuid, search_text FROM cards")
    conn.commit()
    timings["인덱스 쓰기"] = time.perf_counter() - t

    stats = {
        "cards": conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0],
        "videos": conn.execute("SELECT COUNT(*) FROM cards WHERE is_video=1").fetchone()[0],
        "hidden": conn.execute("SELECT COUNT(*) FROM cards WHERE hidden=1").fetchone()[0],
        "with_gps": conn.execute("SELECT COUNT(*) FROM cards WHERE lat IS NOT NULL").fetchone()[0],
        "apple_label_kinds": conn.execute(
            "SELECT COUNT(DISTINCT tag) FROM card_tags WHERE kind='label'").fetchone()[0],
        "tag_kinds_all": conn.execute("SELECT COUNT(DISTINCT tag) FROM card_tags").fetchone()[0],
        "states": conn.execute("SELECT COUNT(DISTINCT state) FROM cards WHERE state IS NOT NULL").fetchone()[0],
        "cities": conn.execute("SELECT COUNT(DISTINCT city) FROM cards WHERE city IS NOT NULL").fetchone()[0],
        "fts_rows": conn.execute("SELECT COUNT(*) FROM cards_fts").fetchone()[0],
        "label_scores_rows": conn.execute("SELECT COUNT(*) FROM label_scores").fetchone()[0],
        "label_scores_names": conn.execute("SELECT COUNT(DISTINCT label_ko) FROM label_scores").fetchone()[0],
    }
    total = time.perf_counter() - t0
    meta = {"built_at": now, "library": args.library, "osxphotos": osxphotos.__version__,
            "build_seconds": f"{total:.1f}", **{k: str(v) for k, v in stats.items()},
            **{f"last_{k}": str(v) for k, v in counts.items()}}
    conn.executemany("INSERT INTO index_meta(key, value) VALUES (?,?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", list(meta.items()))
    conn.commit()
    conn.close()

    print(f"보관함: {args.library} (읽기 전용, DB 복사본 사용)")
    print(f"인덱스: {args.index}")
    print("변경: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print("통계: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    print("장면 확신도: " + ", ".join(f"{k}={v}" for k, v in score_stats.items())
          + f" (하한 {LABEL_SCORE_FLOOR})")
    print("단계 시간: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    print(f"총 실행 시간: {total:.1f}s")

    if not args.no_preview_cache:
        # 웹 서버는 보관함 파일을 권한 문제로 못 열 수 있어 이 캐시만 읽는다. 이 명령은 보관함을 읽을 수 있는 문맥에서 돈다.
        from photo_ask import preview_cache
        t = time.perf_counter()
        pc = preview_cache.sync(Path(args.index))
        print("미리보기 캐시: " + ", ".join(f"{k}={v}" for k, v in pc.items())
              + f" · {time.perf_counter() - t:.1f}s")
        if pc["errors"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
