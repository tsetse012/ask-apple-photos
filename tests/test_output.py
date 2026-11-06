"""결과 출력: 미리보기는 out/으로 복사한 사본만 열고, 시트도 사본만 가리킨다(보관함 파일을 앱으로 직접 열지 않음)."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photo_ask import config, output  # noqa: E402


class OutputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.fake_library = base / "Fake.photoslibrary" / "resources"
        self.fake_library.mkdir(parents=True)
        self.saved_out = config.OUT_DIR
        config.OUT_DIR = base / "out"
        self.rows = []
        for i in range(3):
            src = self.fake_library / f"P{i}.jpeg"
            src.write_bytes(b"\xff\xd8\xff\xd9")
            self.rows.append({"uuid": f"U{i}000000-0000", "date": "2026-01-01 10:00", "place": "가상시",
                              "labels": "바다", "video": False, "preview": str(src)})
        self.rows.append({"uuid": "U9000000-0000", "date": "2026-01-01 10:00", "place": "-",
                          "labels": "-", "video": True, "preview": None})

    def tearDown(self):
        config.OUT_DIR = self.saved_out
        self.tmp.cleanup()

    def test_open_uses_copies_only(self):
        copied = output.copy_previews(self.rows, "qtest")
        self.assertEqual(sum(p is not None for p in copied), 3)
        with mock.patch("photo_ask.output.subprocess.run") as run:
            output.open_paths(copied)
        args = run.call_args[0][0]
        self.assertEqual(args[0], "open")
        self.assertEqual(len(args), 4)
        for a in args[1:]:
            self.assertTrue(a.startswith(str(config.OUT_DIR)), a)
            self.assertNotIn("photoslibrary", a)

    def test_sheet_points_to_copies(self):
        copied = output.copy_previews(self.rows, "qtest")
        sheet = output.write_sheet(self.rows, copied, "qtest", "질문", "요약")
        html = sheet.read_text()
        self.assertEqual(html.count("<img "), 3)
        self.assertIn("미리보기 없음", html)
        self.assertNotIn("photoslibrary", html)

    def test_table_hides_library_paths_by_default(self):
        """표에 보관함 원본 경로를 기본으로 찍지 않는다."""
        rows = [dict(r, caption="[양식] 가상 장면 | 줄") for r in self.rows]
        table = output.format_table(rows)
        self.assertIn("| # | 날짜 | 장소 | 라벨 3개 | 캡션 | uuid |", table)
        self.assertIn("U0000000", table)
        self.assertIn("[양식] 가상 장면 / 줄", table)  # 캡션 속 '|'는 표를 깨지 않게 바꾼다
        self.assertNotIn("photoslibrary", table)

    def test_table_shows_copies_and_paths_only_when_asked(self):
        copied = output.copy_previews(self.rows, "qtest")
        table = output.format_table(self.rows, copied=copied)
        self.assertIn("사본(out/)", table)
        self.assertIn("01_U0000000.jpeg", table)
        self.assertNotIn("photoslibrary", table)
        table = output.format_table(self.rows, show_paths=True)
        self.assertIn("원본 경로", table)
        self.assertIn("photoslibrary", table)


if __name__ == "__main__":
    unittest.main()
