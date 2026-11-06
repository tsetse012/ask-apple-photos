"""미리보기 캐시 동기화(photo_ask/preview_cache.py) — 가짜 보관함 폴더로 검사."""
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from photo_ask import preview_cache  # noqa: E402
from photo_ask.cards_schema import CARD_COLUMNS, open_index_rw  # noqa: E402
from test_cards_query import make_card  # noqa: E402

COLS = [n for n, _ in CARD_COLUMNS]


class PreviewCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = Path(self.tmp.name)
        self.lib = t / "library"
        self.lib.mkdir()
        self.src1 = self.lib / "u1.jpeg"
        self.src1.write_bytes(b"\xff\xd8\xff" + b"one" * 50)
        self.src2 = self.lib / "u2.jpeg"
        self.src2.write_bytes(b"\xff\xd8\xff" + b"two" * 50)
        self.root = t / "previews"
        self.db = t / "cards.sqlite"
        conn = open_index_rw(self.db)
        rows = [("U1", str(self.src1), 0), ("U2", str(self.src2), 1), ("U3", None, 0),
                ("U4", str(self.lib / "missing.jpeg"), 0), ("../evil", str(self.src1), 0)]
        for uuid, path, hidden in rows:
            card, _ = make_card(uuid, "2026-09-20T13:00:00+09:00", 13, hidden=hidden)
            card["preview_path"] = path
            conn.execute(f"INSERT INTO cards ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})",
                         [card[c] for c in COLS])
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_sync_counts_and_read_only_source(self):
        before = (self.src1.read_bytes(), self.src1.stat().st_mtime_ns)
        c = preview_cache.sync(self.db, self.root)
        self.assertEqual((c["wanted"], c["hidden_skipped"], c["no_preview"], c["missing_source"], c["errors"]),
                         (1, 1, 1, 1, 1))
        self.assertEqual(c["cloned"] + c["copied"], 1)
        dst = self.root / "U1.jpeg"
        self.assertEqual(dst.read_bytes(), self.src1.read_bytes())
        self.assertEqual((self.src1.read_bytes(), self.src1.stat().st_mtime_ns), before)   # 원본은 그대로
        self.assertEqual(stat.S_IMODE(dst.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        self.assertFalse((self.root / "..").resolve().joinpath("evil.jpeg").exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["U1.jpeg", "manifest.json"])
        self.assertEqual(preview_cache.sync(self.db, self.root)["unchanged"], 1)

    def test_changed_source_is_recopied_and_hidden_toggle(self):
        preview_cache.sync(self.db, self.root)
        self.src1.write_bytes(b"\xff\xd8\xff" + b"new!" * 80)
        c = preview_cache.sync(self.db, self.root)
        self.assertEqual(c["cloned"] + c["copied"], 1)
        self.assertEqual((self.root / "U1.jpeg").read_bytes(), self.src1.read_bytes())
        c = preview_cache.sync(self.db, self.root, include_hidden=True)
        self.assertTrue((self.root / "U2.jpeg").exists())
        c = preview_cache.sync(self.db, self.root)                 # 기본값(숨김 제외)으로 돌리면 지운다
        self.assertEqual(c["removed"], 1)
        self.assertFalse((self.root / "U2.jpeg").exists())

    def test_missing_source_keeps_old_cache(self):
        preview_cache.sync(self.db, self.root)
        os.rename(self.src1, self.lib / "moved.jpeg")
        c = preview_cache.sync(self.db, self.root)
        self.assertEqual(c["missing_source"], 2)
        self.assertTrue((self.root / "U1.jpeg").exists())

    def test_dry_run_writes_nothing(self):
        c = preview_cache.sync(self.db, self.root, dry_run=True)
        self.assertEqual(c["cloned"], 1)
        self.assertFalse(self.root.exists())

    def test_copy_mode_and_cache_file_guard(self):
        c = preview_cache.sync(self.db, self.root, clone=False)
        self.assertEqual(c["copied"], 1)
        for bad in ("../x", "a/b", "", ".hidden", "x" * 65, None):
            with self.assertRaises(ValueError):
                preview_cache.cache_file(bad, self.root)
        self.assertEqual(preview_cache.cache_file("C0000001-0000", self.root).name, "C0000001-0000.jpeg")


if __name__ == "__main__":
    unittest.main()
