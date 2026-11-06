"""인덱스 빌드·로컬 보강 단위 테스트 — 가짜 Photos.sqlite·가짜 헬퍼 출력으로만(보관함·Vision 실행 없음)."""
import bz2
import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "index"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from photo_ask import scene_taxonomy  # noqa: E402
from photo_ask.cards_schema import open_index_rw  # noqa: E402
import build_index  # noqa: E402
import enrich_vision  # noqa: E402


class RemovalPlanTest(unittest.TestCase):
    """보관함에서 안 보이는 카드가 기존의 5%를 넘으면 지우지 않는다."""

    def test_small_removal_goes_through(self):
        existing = {f"U{i}": None for i in range(100)}
        seen = set(list(existing)[:95])
        gone, blocked = build_index.removal_plan(existing, seen, allow=False)
        self.assertEqual(len(gone), 5)
        self.assertFalse(blocked)

    def test_mass_removal_is_blocked_unless_allowed(self):
        existing = {f"U{i}": None for i in range(100)}
        seen = set(list(existing)[:94])
        gone, blocked = build_index.removal_plan(existing, seen, allow=False)
        self.assertEqual(len(gone), 6)
        self.assertTrue(blocked)
        _, blocked = build_index.removal_plan(existing, seen, allow=True)
        self.assertFalse(blocked)

    def test_empty_index(self):
        self.assertEqual(build_index.removal_plan({}, set(), allow=False), ([], False))


class DeleteCardsOrphanTest(unittest.TestCase):
    """보관함에서 사라진 카드를 지울 때 보강 표의 같은 uuid 행도 함께 지워야 고아 행이 안 남는다."""

    def test_delete_removes_enrich_rows_and_keeps_others(self):
        from test_cards_query import make_card
        from photo_ask.cards_schema import CARD_COLUMNS, ENRICH_TABLES
        cols = [n for n, _ in CARD_COLUMNS]
        with tempfile.TemporaryDirectory() as tmp:
            conn = open_index_rw(Path(tmp) / "cards.sqlite")
            for u in ("GONE", "KEEP"):
                card, tags = make_card(u, "2026-09-01T10:00:00+09:00", 10, labels=["바다"])
                conn.execute(f"INSERT INTO cards ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                             [card[c] for c in cols])
                conn.executemany("INSERT INTO card_tags VALUES (?,?,?)", [(u, t, k) for t, k in tags])
                conn.execute("INSERT INTO label_scores VALUES (?, '바다', 0.5)", (u,))
                conn.execute("INSERT INTO vision_labels VALUES (?, 'sea', '바다', 0.5)", (u,))
                conn.execute("INSERT INTO ocr_text VALUES (?, '글자', 1, 0.9)", (u,))
                conn.execute("INSERT INTO enrich_status(uuid, step, status) VALUES (?, 'vision', 'ok')", (u,))
                conn.execute("INSERT INTO captions(uuid, model, json, created_at) VALUES (?, 'm', '{}', 't')", (u,))
                conn.execute("INSERT INTO caption_failures(uuid, error_kind) VALUES (?, 'output')", (u,))
            conn.commit()
            build_index.delete_cards(conn, ["GONE"])
            conn.commit()
            count = lambda table, u: conn.execute(  # noqa: E731
                f"SELECT COUNT(*) FROM {table} WHERE uuid=?", (u,)).fetchone()[0]
            for table in ("cards", "card_tags") + tuple(ENRICH_TABLES):
                self.assertEqual(count(table, "GONE"), 0, table)
                self.assertGreater(count(table, "KEEP"), 0, table)       # 다른 사진 행은 그대로
            conn.close()


class LabelScoresTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "Photos.sqlite"
        c = sqlite3.connect(self.db)
        c.executescript("""
            CREATE TABLE ZASSET (Z_PK INTEGER PRIMARY KEY, ZUUID TEXT);
            CREATE TABLE ZADDITIONALASSETATTRIBUTES (Z_PK INTEGER PRIMARY KEY, ZASSET INTEGER);
            CREATE TABLE ZSCENECLASSIFICATION (Z_PK INTEGER PRIMARY KEY, ZASSETATTRIBUTES INTEGER,
                ZSCENEIDENTIFIER INTEGER, ZCONFIDENCE REAL, ZCLASSIFICATIONTYPE INTEGER);
            INSERT INTO ZASSET VALUES (1, 'U1'), (2, 'U2');
            INSERT INTO ZADDITIONALASSETATTRIBUTES VALUES (10, 1), (20, 2);
            INSERT INTO ZSCENECLASSIFICATION(ZASSETATTRIBUTES, ZSCENEIDENTIFIER, ZCONFIDENCE, ZCLASSIFICATIONTYPE) VALUES
              (10, 100, 0.8, 0), (10, 101, 0.3, 0), (10, 102, 0.01, 0), (10, 999, 0.9, 0),
              (20, 100, 0.2, 0), (20, 103, 0.6, 0), (20, 100, 0.9, 8);
        """)
        c.commit()
        c.close()
        # 100·101 → 같은 한국어 이름(최대값을 쓴다), 102 → 하한 아래, 999 → 분류표에 없음(사진 앱 내부용)
        self.keys = {100: "dog", 101: "Q1", 102: "cat", 103: "Q2"}
        self.names = {"dog": "개", "Q1": "개", "cat": "고양이", "Q2": "바다"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_mapping_max_floor_and_unmapped(self):
        with mock.patch.object(scene_taxonomy, "scene_keys", return_value=self.keys), \
             mock.patch.object(scene_taxonomy, "names", return_value=self.names):
            best, stats = build_index.read_label_scores(self.db, floor=0.02)
        self.assertEqual(best, {("U1", "개"): 0.8, ("U2", "개"): 0.2, ("U2", "바다"): 0.6})
        self.assertEqual(stats, {"scene_rows": 6, "unmapped_rows": 1, "below_floor_rows": 1})   # 종류 8은 안 읽는다


class TaxonomyLookupTest(unittest.TestCase):
    """분류표 파일 이름의 버전 번호가 macOS마다 다르다 — 있는 것 중 가장 높은 번호를 쓰고, 없으면 예외로 알린다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, version):
        with bz2.open(self.dir / f"PFSceneTaxonomyData_{version}.bz2", "wt", encoding="utf-8") as fh:
            fh.write(f"n:1,key{version}\n")

    def test_highest_version_wins(self):
        self.write(99)
        self.write(101)
        self.assertEqual(scene_taxonomy.taxonomy_version(self.dir), 101)
        self.assertEqual(scene_taxonomy.taxonomy_path(self.dir).name, "PFSceneTaxonomyData_101.bz2")
        self.assertEqual(scene_taxonomy.scene_keys(self.dir), {1: "key101"})

    def test_no_taxonomy_file_raises(self):
        (self.dir / "unrelated.txt").write_text("x")
        with self.assertRaises(scene_taxonomy.TaxonomyUnavailable):
            scene_taxonomy.taxonomy_version(self.dir)

    def test_read_label_scores_without_taxonomy_is_empty_with_reason(self):
        with mock.patch.object(scene_taxonomy, "scene_keys",
                               side_effect=scene_taxonomy.TaxonomyUnavailable("분류표 없음")):
            best, stats = build_index.read_label_scores(Path(self.dir) / "Photos.sqlite")
        self.assertEqual(best, {})
        self.assertEqual(stats["scene_rows"], 0)
        self.assertEqual(stats["unavailable"], "분류표 없음")


class MissingLibraryTest(unittest.TestCase):
    """보관함 경로가 없으면 스택 트레이스 대신 해결 방법을 알려 주고 종료 코드 2."""

    def test_exit_code_2_with_guidance(self):
        with tempfile.TemporaryDirectory() as tmp:
            err = io.StringIO()
            argv = ["build_index.py", "--library", str(Path(tmp) / "nope.photoslibrary"),
                    "--index", str(Path(tmp) / "cards.sqlite")]
            with mock.patch.object(sys, "argv", argv), redirect_stderr(err):
                rc = build_index.main()
        self.assertEqual(rc, 2)
        text = err.getvalue()
        self.assertIn("사진 보관함을 찾지 못했습니다", text)
        self.assertIn("PHOTO_ASK_LIBRARY", text)


class HelperCompileTest(unittest.TestCase):
    """첫 실행에 swiftc로 헬퍼를 자동 컴파일하고, swiftc가 없으면 설치 안내로 끝난다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.src = base / "vision_helper.swift"
        self.src.write_text("// test")
        self.bin = base / "vision_helper"
        self.patches = [mock.patch.object(enrich_vision, "HELPER_SRC", self.src),
                        mock.patch.object(enrich_vision, "HELPER_BIN", self.bin)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_missing_swiftc_prints_install_guidance(self):
        with mock.patch.object(enrich_vision.shutil, "which", return_value=None), \
             self.assertRaises(SystemExit) as cm:
            enrich_vision.ensure_helper()
        self.assertIn("xcode-select --install", str(cm.exception.code))

    def test_first_run_compiles_with_swiftc(self):
        def fake_run(args, check):
            self.assertEqual(args[0], "/usr/bin/swiftc")
            self.bin.write_bytes(b"bin")
            self.compiled = args

        with mock.patch.object(enrich_vision.shutil, "which", return_value="/usr/bin/swiftc"), \
             mock.patch.object(enrich_vision.subprocess, "run", side_effect=fake_run):
            out = enrich_vision.ensure_helper()
        self.assertEqual(out, self.bin)
        self.assertIn(str(self.src), self.compiled)
        self.assertIn(str(self.bin), self.compiled)


@unittest.skipUnless((scene_taxonomy.RES_DIR / "scenetaxonomy.loctable").exists(), "macOS 분류표 없음")
class RealTaxonomyTest(unittest.TestCase):
    def test_vision_ids_have_korean_names(self):
        ko = scene_taxonomy.names("ko")
        self.assertEqual(ko["dog"], "개")
        self.assertEqual(ko["food"], "음식")
        self.assertTrue(len(scene_taxonomy.scene_keys()) > 1000)


class VisionWriterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = open_index_rw(Path(self.tmp.name) / "cards.sqlite")
        names = {"dog": "개", "food": "음식"}
        self.patch = mock.patch.object(scene_taxonomy, "names", return_value=names)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.conn.close()
        self.tmp.cleanup()

    def test_classify_lines_write_labels_and_status(self):
        w = enrich_vision.Writer(self.conn, "vision", "run1", {"U1": "/p/1.jpeg", "U2": "/p/2.jpeg"})
        self.assertTrue(w({"uuid": "U1", "labels": [["dog", 0.9], ["unknown_id", 0.05]], "ms": 7.0}))
        self.assertTrue(w({"uuid": "U2", "error": "LOADFAIL"}))
        self.conn.commit()
        rows = self.conn.execute("SELECT uuid, label_en, label_ko, confidence FROM vision_labels ORDER BY 2").fetchall()
        self.assertEqual(rows, [("U1", "dog", "개", 0.9), ("U1", "unknown_id", None, 0.05)])
        st = dict(self.conn.execute("SELECT uuid, status FROM enrich_status").fetchall())
        self.assertEqual(st, {"U1": "ok", "U2": "error"})
        self.assertEqual(w.n["no_ko_name"], 1)

    def test_ocr_lines_and_e5rt_streak_stop(self):
        paths = {f"U{i}": f"/p/{i}.jpeg" for i in range(8)}
        w = enrich_vision.Writer(self.conn, "ocr", "run1", paths)
        self.assertTrue(w({"uuid": "U0", "lines": [["합계", 0.5], ["12,000원", 1.0]], "ms": 120}))
        e5 = 'VNFAIL e5rtError("e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options call failed", 13)'
        results = [w({"uuid": f"U{i}", "error": e5}) for i in range(1, 6)]
        self.assertEqual(results, [True, True, True, True, False])   # 5번째 연속 실패에서 멈춘다
        self.assertTrue(w.e5rt_stopped)
        self.conn.commit()
        self.assertEqual(self.conn.execute("SELECT text, line_count, mean_conf FROM ocr_text").fetchall(),
                         [("합계\n12,000원", 2, 0.75)])
        self.assertEqual(self.conn.execute(
            "SELECT COUNT(*) FROM enrich_status WHERE step='ocr' AND status='error'").fetchone()[0], 5)

    def test_ocr_target_reasons(self):
        from test_cards_query import make_card
        from photo_ask.cards_schema import CARD_COLUMNS
        cols = [n for n, _ in CARD_COLUMNS]
        cards = [make_card("T1", "2026-09-01T10:00:00+09:00", 10, labels=["간판"]),
                 make_card("T2", "2026-09-02T10:00:00+09:00", 10, labels=["바다"]),
                 make_card("T3", "2026-09-03T10:00:00+09:00", 10, labels=["바다"]),
                 make_card("T4", "2026-09-04T10:00:00+09:00", 10, labels=["문서"], video=1)]
        for card, tags in cards:
            card["preview_path"] = f"/p/{card['uuid']}.jpeg"
            self.conn.execute(f"INSERT INTO cards ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                              [card[c] for c in cols])
            self.conn.executemany("INSERT INTO card_tags VALUES (?,?,?)", [(card["uuid"], t, k) for t, k in tags])
        self.conn.execute("UPDATE cards SET text_found=1 WHERE uuid='T2'")
        self.conn.execute("INSERT INTO vision_labels VALUES ('T3', 'document', '문서', 0.25)")
        reasons = enrich_vision.ocr_target_sources(self.conn)
        self.assertEqual(reasons, {"T1": ["apple_label"], "T2": ["apple_text_found"], "T3": ["vision_label"]})  # 영상 제외
        items, _ = enrich_vision.ocr_targets(self.conn)
        self.assertEqual(sorted(u for u, _ in items), ["T1", "T2", "T3"])


class VisionWriterWithoutTaxonomyTest(unittest.TestCase):
    """분류표가 없는 macOS: 이름표는 classify만 읽고, 없으면 경고 한 번 뒤 영어 라벨 id를 이름으로 쓴다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = open_index_rw(Path(self.tmp.name) / "cards.sqlite")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_classify_uses_english_id_when_taxonomy_is_missing(self):
        err = io.StringIO()
        with mock.patch.object(enrich_vision.scene_taxonomy, "names",
                               side_effect=scene_taxonomy.TaxonomyUnavailable("분류표 없음")), \
             redirect_stderr(err):
            w = enrich_vision.Writer(self.conn, "vision", "run1", {"U1": "/p/1.jpeg"})
            w({"uuid": "U1", "labels": [["dog", 0.9], ["food", 0.4]], "ms": 5.0})
        rows = self.conn.execute("SELECT uuid, label_en, label_ko, confidence FROM vision_labels ORDER BY 2").fetchall()
        self.assertEqual(rows, [("U1", "dog", "dog", 0.9), ("U1", "food", "food", 0.4)])
        self.assertEqual(err.getvalue().count("⚠"), 1)          # 경고는 한 번만

    def test_ocr_never_reads_the_name_table(self):
        with mock.patch.object(enrich_vision.scene_taxonomy, "names",
                               side_effect=AssertionError("OCR이 이름표를 읽었다")):
            w = enrich_vision.Writer(self.conn, "ocr", "run1", {"U1": "/p/1.jpeg"})
            self.assertTrue(w({"uuid": "U1", "lines": [["합계", 0.9]], "ms": 3.0}))
        self.assertEqual(self.conn.execute("SELECT text FROM ocr_text").fetchone()[0], "합계")


class RebuildOrphanCleanupTest(unittest.TestCase):
    """--rebuild 뒤 카드에 없는 보강 행은 정리하고, 현재 카드의 행은 남긴다(5% 가드 동일).
    정리 대상은 삭제 전 목록(prior)이 아니라 보강 표 자체의 uuid라서, 가드로 남긴 행도 재실행에서 지워진다."""

    def setUp(self):
        from test_cards_query import make_card
        from photo_ask.cards_schema import CARD_COLUMNS
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = open_index_rw(Path(self.tmp.name) / "cards.sqlite")
        self.cols = [n for n, _ in CARD_COLUMNS]
        self.make_card = make_card

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def seed(self, uuid):
        card, tags = self.make_card(uuid, "2026-09-01T10:00:00+09:00", 10, labels=["바다"])
        self.conn.execute(f"INSERT INTO cards ({','.join(self.cols)}) VALUES ({','.join('?' * len(self.cols))})",
                          [card[c] for c in self.cols])
        self.conn.executemany("INSERT INTO card_tags VALUES (?,?,?)", [(uuid, t, k) for t, k in tags])
        self.conn.execute("INSERT INTO label_scores VALUES (?, '바다', 0.5)", (uuid,))
        self.conn.execute("INSERT INTO vision_labels VALUES (?, 'sea', '바다', 0.5)", (uuid,))
        self.conn.execute("INSERT INTO ocr_text VALUES (?, '글자', 1, 0.9)", (uuid,))
        self.conn.execute("INSERT INTO enrich_status(uuid, step, status) VALUES (?, 'vision', 'ok')", (uuid,))
        self.conn.execute("INSERT INTO captions(uuid, model, json, created_at) VALUES (?, 'm', '{}', 't')", (uuid,))
        self.conn.execute("INSERT INTO caption_failures(uuid, error_kind) VALUES (?, 'output')", (uuid,))

    def count(self, table, uuid):
        return self.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE uuid=?", (uuid,)).fetchone()[0]

    def test_removes_orphans_and_keeps_current_cards_rows(self):
        from photo_ask.cards_schema import ENRICH_TABLES
        for u in ("KEEP", "GONE"):
            self.seed(u)
        self.conn.execute("DELETE FROM cards WHERE uuid='GONE'")    # 카드만 지워지고 보강 행은 남은 상태
        gone, blocked, n_enrich = build_index.cleanup_enrich_orphans(self.conn, allow=True)
        self.assertEqual(gone, ["GONE"])
        self.assertFalse(blocked)
        self.assertEqual(n_enrich, 2)
        for table in ENRICH_TABLES:
            self.assertEqual(self.count(table, "GONE"), 0, table)
            self.assertGreater(self.count(table, "KEEP"), 0, table)

    def test_mass_delete_guard_blocks_orphan_cleanup(self):
        self.seed("KEEP")                                            # 현재 카드 1장
        for i in range(100):
            self.conn.execute("INSERT INTO label_scores VALUES (?, '바다', 0.5)", (f"U{i}",))
        gone, blocked, n_enrich = build_index.cleanup_enrich_orphans(self.conn, allow=False)
        self.assertEqual(len(gone), 100)
        self.assertTrue(blocked)
        self.assertEqual(n_enrich, 101)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM label_scores").fetchone()[0], 101)

    def test_allow_mass_delete_rerun_cleans_guarded_orphans(self):
        """가드로 남긴 행은 다음 실행의 prior에 없어도 정리된다 — 99/100 보호 뒤 --allow-mass-delete 재실행."""
        for i in range(100):
            self.conn.execute("INSERT INTO label_scores VALUES (?, '바다', 0.5)", (f"U{i}",))
        self.seed("KEEP")                                            # 현재 카드
        _, blocked, _ = build_index.cleanup_enrich_orphans(self.conn, allow=False)
        self.assertTrue(blocked)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM label_scores").fetchone()[0], 101)
        gone, blocked, _ = build_index.cleanup_enrich_orphans(self.conn, allow=True)
        self.assertFalse(blocked)
        self.assertEqual(sorted(gone), sorted(f"U{i}" for i in range(100)))
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM label_scores").fetchone()[0], 1)
        self.assertEqual(self.count("label_scores", "KEEP"), 1)


class RebuildMainTest(unittest.TestCase):
    """build_index.main --rebuild: 카드에 없는 보강 행은 정리하고, 가드로 막힌 행은 재실행(--allow-mass-delete)에서 지운다."""

    def seed(self, db, uuids):
        from photo_ask.cards_schema import CARD_COLUMNS
        cols = [n for n, _ in CARD_COLUMNS]
        conn = open_index_rw(db)
        for u in uuids:
            card, _ = __import__("test_cards_query", fromlist=["make_card"]).make_card(
                u, "2026-09-01T10:00:00+09:00", 10)
            conn.execute(f"INSERT INTO cards ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                         [card[c] for c in cols])
            conn.execute("INSERT INTO vision_labels VALUES (?, 'sea', '바다', 0.5)", (u,))
        conn.commit()
        conn.close()

    def run_main(self, base, db, built_uuids, allow=False):
        from types import SimpleNamespace
        err = io.StringIO()

        def fake_build_card(photo, extras):
            card, _ = __import__("test_cards_query", fromlist=["make_card"]).make_card(
                photo, "2026-09-01T10:00:00+09:00", 10)
            return card, []

        fake_db = SimpleNamespace(_tmp_db=str(base / "copy" / "Photos.sqlite"),
                                  _dbfile=str(base / "copy" / "Photos.sqlite"),
                                  photos=lambda **kw: built_uuids)
        fake_osxphotos = SimpleNamespace(PhotosDB=lambda **kw: fake_db, __version__="test")
        argv = ["build_index.py", "--library", str(base / "Fake.photoslibrary"), "--index", str(db),
                "--rebuild", "--no-preview-cache"] + (["--allow-mass-delete"] if allow else [])
        with mock.patch.dict(sys.modules, {"osxphotos": fake_osxphotos}), \
             mock.patch.object(build_index, "copy_library_database", return_value=base / "copy" / "Photos.sqlite"), \
             mock.patch.object(build_index, "read_asset_extras", return_value={}), \
             mock.patch.object(build_index, "read_label_scores", return_value=({}, {"scene_rows": 0})), \
             mock.patch.object(build_index, "build_card", side_effect=fake_build_card), \
             mock.patch.object(sys, "argv", argv), redirect_stderr(err):
            rc = build_index.main()
        self.assertEqual(rc, 0, err.getvalue())
        conn = open_index_rw(db)
        rows = {
            "cards": conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0],
            "keep_vision": conn.execute("SELECT COUNT(*) FROM vision_labels WHERE uuid='KEEP'").fetchone()[0],
            "gone_vision": conn.execute("SELECT COUNT(*) FROM vision_labels WHERE uuid='GONE'").fetchone()[0],
            "all_vision": conn.execute("SELECT COUNT(*) FROM vision_labels").fetchone()[0],
        }
        conn.close()
        return rows, err.getvalue()

    def make_library(self, base):
        library = base / "Fake.photoslibrary"
        (library / "database" / "search").mkdir(parents=True)
        (library / "database" / "search" / "psi.sqlite").write_bytes(b"")
        return library

    def test_rebuild_removes_enrich_rows_for_photos_no_longer_in_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.make_library(base)
            db = base / "cards.sqlite"
            self.seed(db, ["KEEP", "GONE"])
            rows, _ = self.run_main(base, db, ["KEEP", "NEW"], allow=True)
        self.assertEqual(rows["cards"], 2)
        self.assertEqual(rows["keep_vision"], 1)      # 현재 카드의 보강 행은 유지
        self.assertEqual(rows["gone_vision"], 0)      # 사라진 사진의 보강 행은 정리

    def test_rebuild_still_applies_mass_delete_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.make_library(base)
            db = base / "cards.sqlite"
            self.seed(db, [f"U{i}" for i in range(100)])
            rows, err = self.run_main(base, db, ["U0"], allow=False)
        self.assertEqual(rows["all_vision"], 100)     # 99/100 삭제는 막혀 보강 행이 남는다
        self.assertIn("초과라 지우지 않음", err)

    def test_rerun_with_allow_mass_delete_removes_guarded_orphans(self):
        """보호로 남은 고아 행은 삭제 전 목록(prior)에 없어도 다음 실행에서 정리된다."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.make_library(base)
            db = base / "cards.sqlite"
            self.seed(db, [f"U{i}" for i in range(100)])
            first, err1 = self.run_main(base, db, ["U0"], allow=False)
            self.assertEqual(first["all_vision"], 100)
            self.assertIn("초과라 지우지 않음", err1)
            second, err2 = self.run_main(base, db, ["U0"], allow=True)
        self.assertEqual(second["all_vision"], 1)     # U0의 행만 남고 고아 99건은 정리
        self.assertEqual(second["keep_vision"], 0)    # U0 uuid는 'KEEP'이 아니다
        self.assertNotIn("초과라 지우지 않음", err2)


class EmptyIndexVisionReportTest(unittest.TestCase):
    """빈 색인(카드 0장)에서 SUM(...)이 NULL이 되어도 예외 없이 안내하고 정상 종료한다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "cards.sqlite"
        conn = open_index_rw(self.db)          # 스키마만 만들고 카드는 0장
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_report_says_index_is_empty(self):
        conn = open_index_rw(self.db)
        text = enrich_vision.report(conn)      # TypeError가 나면 실패
        conn.close()
        self.assertIn("색인이 비어 있다", text)

    def test_report_flag_exits_zero(self):
        out = io.StringIO()
        from contextlib import redirect_stdout
        with mock.patch.object(sys, "argv", ["enrich_vision.py", "--index", str(self.db), "--report"]), \
             redirect_stdout(out):
            rc = enrich_vision.main()
        self.assertEqual(rc, 0)
        self.assertIn("색인이 비어 있다", out.getvalue())

    def test_normal_run_on_empty_index_exits_zero(self):
        out = io.StringIO()
        from contextlib import redirect_stdout
        with mock.patch.object(sys, "argv", ["enrich_vision.py", "--index", str(self.db)]), \
             redirect_stdout(out):
            rc = enrich_vision.main()
        self.assertEqual(rc, 0)
        self.assertIn("색인이 비어 있다", out.getvalue())


if __name__ == "__main__":
    unittest.main()
