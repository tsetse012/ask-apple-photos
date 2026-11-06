"""명령줄 진입점의 --help 회귀 테스트 — 보관함을 열지 않고 종료 코드 0(가짜 데이터·API 키 없음)."""
import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from photo_ask.cards_schema import open_index_rw  # noqa: E402

SCRIPTS = ["ask.py", "index/build_index.py", "index/enrich_vision.py", "index/enrich_caption.py",
           "index/sync_previews.py"]


class HelpTest(unittest.TestCase):
    """`--help`는 argparse가 인자를 읽기 전에 끝내므로 보관함·API에 닿지 않는다."""

    def test_every_script_help_exits_zero(self):
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY")}
        for script in SCRIPTS:
            with self.subTest(script=script):
                r = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=ROOT, env=env,
                                   capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 0, f"{script}\nstdout={r.stdout}\nstderr={r.stderr}")


class EmptyIndexAskTest(unittest.TestCase):
    """카드 0장 색인이면 ask.py가 예외 없이 build_index.py 실행을 안내하고 끝난다(모델 호출 0)."""

    def test_ask_prints_guidance_without_model_call(self):
        import ask
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "cards.sqlite"
            open_index_rw(db).close()
            err = io.StringIO()
            argv = ["ask.py", "고양이", "--index", str(db)]
            with mock.patch.object(sys, "argv", argv), redirect_stderr(err):
                rc = ask.main()
        self.assertEqual(rc, 2)
        text = err.getvalue()
        self.assertIn("색인이 비어 있다", text)
        self.assertIn("build_index.py", text)


if __name__ == "__main__":
    unittest.main()
