"""캡션 단계 단위 테스트 — 가짜 HTTP 응답으로만(과금 0, 사진 전송 0).

- 응답 정리(parse_caption): 약속한 JSON이 아니면 CaptionParseError
- 사진 1장(caption_one): 재시도 2회·429 대기·4xx·MAX_TOKENS·차단, 요청 시도마다 기록(images_sent=1)
- 배치(run): 파싱 실패는 건너뛰고 caption_failures에 기록, 예산·정지 파일·연속 4xx에서 멈춤, 재개 대상
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "index"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from photo_ask import caption, config, llm_log  # noqa: E402
from photo_ask.cards_schema import open_index_rw  # noqa: E402
from photo_ask.jpeg_meta import JpegMetadataError, strip_metadata  # noqa: E402
import enrich_caption  # noqa: E402

GOOD = {"scene_ko": "가상 식탁", "objects_ko": ["접시", "포크", "접시"], "food": {"present": True, "cuisine": "양식",
        "dishes_ko": ["파스타"]}, "people_count": 0, "indoor_outdoor": "실내", "time_of_day_guess": "낮",
        "text_visible": False, "text_snippet": "", "doc_type": "없음", "keywords_ko": ["점심", "식사"]}


class Resp:
    def __init__(self, code, body, headers=None):
        self.status_code = code
        self._body = body
        self.text = json.dumps(body, ensure_ascii=False)
        self.headers = headers or {}

    def json(self):
        return self._body


def ok_resp(text, finish="STOP", prompt=1535, out=250):
    return Resp(200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
                      "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": out,
                                        "promptTokensDetails": [{"modality": "IMAGE", "tokenCount": 1064}]},
                      "modelVersion": "gemini-3.1-flash-lite", "responseId": "resp-test"})


class FakePost:
    def __init__(self, script):
        self.script = list(script)
        self.bodies = []

    def __call__(self, url, headers=None, data=None, timeout=None):
        self.bodies.append(json.loads(data))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ParseCaptionTest(unittest.TestCase):
    def test_valid_is_normalized(self):
        cap = caption.parse_caption(json.dumps(GOOD, ensure_ascii=False))
        self.assertEqual(cap["objects_ko"], ["접시", "포크"])       # 중복 제거
        self.assertEqual(cap["food"]["cuisine"], "양식")
        self.assertIn("파스타", caption.search_blob(cap))

    def test_not_json_or_missing_keys(self):
        with self.assertRaises(caption.CaptionParseError):
            caption.parse_caption("음식 사진입니다")
        with self.assertRaises(caption.CaptionParseError):
            caption.parse_caption(json.dumps([1, 2]))
        bad = dict(GOOD)
        bad.pop("doc_type")
        with self.assertRaises(caption.CaptionParseError):
            caption.parse_caption(json.dumps(bad, ensure_ascii=False))

    def test_enum_outside_list_fails(self):
        bad = json.loads(json.dumps(GOOD))
        bad["food"]["cuisine"] = "프랑스식"
        with self.assertRaises(caption.CaptionParseError):
            caption.parse_caption(json.dumps(bad, ensure_ascii=False))
        bad = dict(GOOD, doc_type="여권")
        with self.assertRaises(caption.CaptionParseError):
            caption.parse_caption(json.dumps(bad, ensure_ascii=False))

    def test_cuisine_consistency_rules_and_limits(self):
        d = json.loads(json.dumps(GOOD))
        d["food"]["present"] = False
        d["text_snippet"] = "가" * 60
        d["keywords_ko"] = [f"k{i}" for i in range(15)]
        d["people_count"] = "3"
        cap = caption.parse_caption(json.dumps(d, ensure_ascii=False))
        self.assertEqual(cap["food"]["cuisine"], "없음")           # 음식이 주 대상이 아니면 국적 없음
        self.assertEqual(len(cap["text_snippet"]), caption.SNIPPET_MAX)
        self.assertEqual(len(cap["keywords_ko"]), 10)
        self.assertEqual(cap["people_count"], 3)
        d = json.loads(json.dumps(GOOD))
        d["food"]["cuisine"] = "없음"
        self.assertEqual(caption.parse_caption(json.dumps(d, ensure_ascii=False))["food"]["cuisine"], "기타")
        d["people_count"] = "많음"
        with self.assertRaises(caption.CaptionParseError):
            caption.parse_caption(json.dumps(d, ensure_ascii=False))

    def test_request_shape(self):
        body = caption.build_request(b"\xff\xd8\xff\xd9")
        gc = body["generationConfig"]
        self.assertEqual(gc["thinkingConfig"], {"thinkingBudget": 0})   # 사고 토큰이 출력 예산을 먹는 함정
        self.assertEqual(gc["responseMimeType"], "application/json")
        self.assertEqual(gc["responseSchema"]["properties"]["food"]["properties"]["cuisine"]["enum"], caption.CUISINES)
        self.assertEqual(body["contents"][0]["parts"][0]["inlineData"]["mimeType"], "image/jpeg")


class CaptionOneTest(unittest.TestCase):
    def setUp(self):
        self.saved_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "test-key"
        self.logged = []
        self.sleeps = []

    def tearDown(self):
        if self.saved_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = self.saved_key

    def call(self, script, **kw):
        post = FakePost(script)
        o = caption.caption_one(post, "U1", b"\xff\xd8\xff\xd9", model="gemini-3.1-flash-lite", run_id="t",
                                log_fn=lambda rec, raw: self.logged.append((rec, raw)), sleep=self.sleeps.append, **kw)
        return o, post

    def test_parse_failure_retries_twice_then_gives_up(self):
        o, post = self.call([ok_resp("not json"), ok_resp("{}"), ok_resp("[]")])
        self.assertFalse(o.ok)
        self.assertEqual(o.error_kind, "output")
        self.assertEqual(o.attempts, 3)                               # 처음 + 재시도 2회
        self.assertEqual(len(self.logged), 3)
        self.assertTrue(all(rec["images_sent"] == 1 for rec, _ in self.logged))
        self.assertTrue(all(rec["provider"] == "gemini" for rec, _ in self.logged))
        self.assertEqual([raw for _, raw in self.logged], ["not json", "{}", "[]"])   # 원문도 기록
        # 비용 = 1535×0.25 + 250×1.50 (USD/1M) × 3회 — 실패한 시도도 돈은 든다
        self.assertAlmostEqual(o.cost_usd, 3 * (1535 * 0.25 + 250 * 1.50) / 1e6, places=9)

    def test_429_waits_with_retry_delay_then_succeeds(self):
        r429 = Resp(429, {"error": {"code": 429, "details": [{"retryDelay": "7s"}]}})
        o, _ = self.call([r429, ok_resp(json.dumps(GOOD, ensure_ascii=False))])
        self.assertTrue(o.ok)
        self.assertEqual(o.attempts, 2)
        self.assertGreaterEqual(self.sleeps[0], 7)
        self.assertEqual(self.logged[0][0]["http_status"], 429)
        self.assertEqual(self.logged[0][0]["images_sent"], 1)          # 거절돼도 사진은 나갔다

    def test_4xx_is_request_error_without_retry(self):
        o, post = self.call([Resp(404, {"error": {"message": "no longer available"}})])
        self.assertFalse(o.ok)
        self.assertEqual(o.error_kind, "request")
        self.assertEqual(len(post.bodies), 1)

    def test_network_error_retries(self):
        o, _ = self.call([ConnectionError("down"), ok_resp(json.dumps(GOOD, ensure_ascii=False))])
        self.assertTrue(o.ok)
        self.assertIn("network", self.logged[0][0]["error"])

    def test_max_tokens_is_failure_and_raises_limit(self):
        o, post = self.call([ok_resp('{"scene_ko": "잘', finish="MAX_TOKENS"),
                             ok_resp(json.dumps(GOOD, ensure_ascii=False))])
        self.assertTrue(o.ok)
        self.assertEqual(post.bodies[1]["generationConfig"]["maxOutputTokens"],
                         2 * post.bodies[0]["generationConfig"]["maxOutputTokens"])

    def test_blocked_is_output_failure_without_retry(self):
        o, post = self.call([ok_resp("", finish="SAFETY")])
        self.assertFalse(o.ok)
        self.assertEqual(o.error_kind, "output")
        self.assertIn("blocked", o.error)
        self.assertEqual(len(post.bodies), 1)

    def test_http_200_non_json_or_wrong_shape_is_logged_output_failure(self):
        # 200인데 JSON이 아니거나 구조가 틀린 응답 — 예외로 배치를 멈추지 않고 파싱 실패처럼 기록·재시도한다.
        o, post = self.call([Resp(200, "not a dict"),
                             Resp(200, {"candidates": "oops"}),
                             Resp(200, {"candidates": [{"content": {"parts": "oops"}}]})])
        self.assertFalse(o.ok)
        self.assertEqual(o.error_kind, "output")
        self.assertEqual(o.attempts, 3)
        self.assertEqual(len(post.bodies), 3)
        self.assertEqual(len(self.logged), 3)
        self.assertEqual([rec["http_status"] for rec, _ in self.logged], [200, 200, 200])
        self.assertTrue(all("응답 형식" in rec["error"] for rec, _ in self.logged))
        self.assertEqual([raw for _, raw in self.logged], [None, None, None])

    def test_body_shape_failure_keeps_computed_cost(self):
        """사용량은 유효한데 본문 part가 깨진 응답 — 이미 계산한 비용을 기록·합산에서 잃지 않는다."""
        def bad_body():
            r = ok_resp("x")
            r._body["candidates"][0]["content"]["parts"] = [{"text": 123}]   # 글자가 아닌 text
            return r

        o, post = self.call([bad_body(), bad_body(), bad_body()])
        self.assertFalse(o.ok)
        self.assertEqual(o.error_kind, "output")
        self.assertEqual(o.attempts, 3)
        per_attempt = (1535 * 0.25 + 250 * 1.50) / 1e6
        self.assertAlmostEqual(o.cost_usd, 3 * per_attempt, places=9)
        costs = [rec["cost_usd"] for rec, _ in self.logged]
        self.assertEqual(len(costs), 3)
        self.assertTrue(all(c > 0 for c in costs), costs)                    # 기록 비용이 0이 아니다
        self.assertTrue(all("본문 형식" in rec["error"] for rec, _ in self.logged))
        self.assertEqual([raw for _, raw in self.logged], [None, None, None])

    def test_shape_failure_keeps_usage_cost(self):
        """구조 검사에 걸리는 응답도 사용량이 유효하면 비용을 기록·합산한다."""
        def malformed(**changes):
            r = ok_resp("x")
            r._body.update(changes)
            return r

        o, post = self.call([malformed(candidates=[None]),
                             malformed(candidates=[{"content": {"parts": "oops"}}]),
                             malformed(candidates="oops")])
        self.assertFalse(o.ok)
        self.assertEqual(o.error_kind, "output")
        self.assertEqual(o.attempts, 3)
        per_attempt = (1535 * 0.25 + 250 * 1.50) / 1e6
        self.assertAlmostEqual(o.cost_usd, 3 * per_attempt, places=9)
        self.assertTrue(all(rec["cost_usd"] > 0 for rec, _ in self.logged))
        self.assertTrue(all("응답 형식" in rec["error"] for rec, _ in self.logged))


def _segment(marker, payload):
    return bytes((0xFF, marker)) + (len(payload) + 2).to_bytes(2, "big") + payload


SECRET = b"SECRET-GPS-37.5665"
COLOR = (200, 60, 20)


def plain_jpeg(size=(16, 8), **save_args):
    out = io.BytesIO()
    Image.new("RGB", size, COLOR).save(out, format="JPEG", quality=95, **save_args)
    return out.getvalue()


def jpeg_with_metadata():
    """디코더가 받아들이는 JPEG에 메타데이터 구간·EOI 앞뒤 데이터·보조 이미지를 넣는다."""
    exif = Image.Exif()
    exif[0x010F] = SECRET.decode()
    data = plain_jpeg(exif=exif.tobytes(), comment=SECRET, icc_profile=b"color" + SECRET)
    hidden = b"".join(_segment(*seg) for seg in (
        (0xE1, b"Exif\x00\x00" + SECRET),
        (0xE1, b"http://ns.adobe.com/xap/1.0/\x00" + SECRET),
        (0xE2, b"MPF\x00" + SECRET),
        (0xE2, b"urn:iso:std:iso:ts:21496:-1\x00" + SECRET),
        (0xED, b"Photoshop 3.0\x00" + SECRET),
        (0xE5, b"unknown-app-segment" + SECRET),
        (0xFE, b"comment" + SECRET),
    ))
    return data[:2] + hidden + data[2:-2] + SECRET + data[-2:] + b"\xff\xd8" + _segment(0xE1, SECRET) + b"\xff\xd9"


class StripMetadataTest(unittest.TestCase):
    def assert_clean(self, data):
        out = strip_metadata(data)
        self.assertNotIn(SECRET, out)
        self.assertNotIn(b"Exif", out)
        self.assertNotIn(b"ICC_PROFILE", out)
        with Image.open(io.BytesIO(out)) as im:
            self.assertEqual(im.format, "JPEG")
            self.assertFalse(set(im.info) & {"exif", "comment", "icc_profile", "xmp", "photoshop"}, im.info.keys())
            im.load()
        return out

    def test_removes_every_metadata_segment_and_trailing_data(self):
        self.assert_clean(jpeg_with_metadata())

    def test_removes_data_hidden_in_segments_decoders_accept(self):
        data = plain_jpeg()
        density = b"JFIF\x00\x01\x01\x00GPS!\x00\x00"
        for hidden in (data[:2] + _segment(0xDC, b"\x00\x08" + SECRET) + data[2:],     # DNL 뒤 데이터
                       data[:-2] + SECRET + data[-2:],                                 # 스캔 뒤 데이터
                       data[:2] + _segment(0xE0, density + SECRET) + data[20:]):        # JFIF 밀도 칸
            with self.subTest(hidden=hidden[:24]):
                out = self.assert_clean(hidden)
                self.assertNotIn(b"GPS!", out)

    def test_keeps_pixels_size_and_applies_rotation(self):
        out = self.assert_clean(jpeg_with_metadata())
        with Image.open(io.BytesIO(out)) as im:
            self.assertEqual(im.size, (16, 8))
            for got, want in zip(im.convert("RGB").getpixel((8, 4)), COLOR):
                self.assertLess(abs(got - want), 6)
        exif = Image.Exif()
        exif[0x0112] = 6
        with Image.open(io.BytesIO(strip_metadata(plain_jpeg(exif=exif.tobytes())))) as im:
            self.assertEqual(im.size, (8, 16))

    def test_rejects_what_it_cannot_decode(self):
        good = plain_jpeg()
        png = io.BytesIO()
        Image.new("RGB", (4, 4)).save(png, format="PNG")
        for bad in (b"", b"not a jpeg", png.getvalue(), b"\xff\xd8", good[:len(good) // 2], b"\xff\xd8garbage"):
            with self.subTest(bad=bad[:12]), self.assertRaises(JpegMetadataError):
                strip_metadata(bad)


class RunBatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.saved = (config.LLM_LOG, config.CAPTION_RAW_LOG, os.environ.get("GEMINI_API_KEY"))
        config.LLM_LOG = base / "llm.jsonl"
        config.CAPTION_RAW_LOG = base / "raw.jsonl"
        os.environ["GEMINI_API_KEY"] = "test-key"
        self.conn = open_index_rw(base / "cards.sqlite")
        self.items = []
        for i in range(5):
            img = base / f"P{i}.jpeg"
            img.write_bytes(plain_jpeg())
            self.items.append((f"U{i}", str(img)))
        self.stop_file = base / "caption.stop"
        self.out = []

    def tearDown(self):
        config.LLM_LOG, config.CAPTION_RAW_LOG, key = self.saved
        if key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = key
        self.conn.close()
        self.tmp.cleanup()

    def test_batch_sends_only_stripped_bytes(self):
        Path(self.items[0][1]).write_bytes(jpeg_with_metadata())
        sent = []

        def caption_fn(uuid, image, log_fn):
            sent.append(image)
            return caption.CaptionOutcome(uuid=uuid, ok=False, error_kind="output", error="멈춤")

        self.run_batch(None, items=self.items[:1], caption_fn=caption_fn)
        self.assertEqual(len(sent), 1)
        self.assertNotIn(SECRET, sent[0])
        self.assertEqual(Image.open(io.BytesIO(sent[0])).size, (16, 8))

    def test_file_that_cannot_be_stripped_is_never_sent(self):
        Path(self.items[0][1]).write_bytes(b"not a jpeg" + SECRET)
        sent = []

        def caption_fn(uuid, image, log_fn):
            sent.append(uuid)
            return caption.CaptionOutcome(uuid=uuid, ok=False, error_kind="output", error="멈춤")

        stats = self.run_batch(None, items=self.items[:1], caption_fn=caption_fn)
        self.assertEqual(sent, [])
        self.assertEqual(stats["fail_output"], 1)
        row = self.conn.execute("SELECT error FROM caption_failures WHERE uuid='U0'").fetchone()
        self.assertIn("전송 안 함", row[0])

    def run_batch(self, post, items=None, budget=1.0, caption_fn=None):
        if caption_fn is None:
            def caption_fn(uuid, image, log_fn):
                return caption.caption_one(post, uuid, image, model="gemini-3.1-flash-lite", run_id="t",
                                           log_fn=log_fn, sleep=lambda s: None)
        return enrich_caption.run(self.conn, items or self.items, model="test", budget=budget,
                                  concurrency=1, caption_fn=caption_fn, run_id="t",
                                  ledger=enrich_caption.Ledger(0.0),
                                  print_fn=lambda *a, **k: self.out.append(a), stop_file=self.stop_file)

    def test_parse_failure_is_skipped_and_recorded(self):
        good = json.dumps(GOOD, ensure_ascii=False)
        # U0: 세 번 다 깨진 응답 → 건너뜀, U1~U4: 정상
        post = FakePost([ok_resp("x"), ok_resp("y"), ok_resp("z")] + [ok_resp(good) for _ in range(4)])
        res = self.run_batch(post)
        self.assertEqual((res["ok"], res["fail"], res["fail_output"]), (4, 1, 1))
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0], 4)
        f = self.conn.execute("SELECT uuid, error_kind, attempts, raw_text FROM caption_failures").fetchall()
        self.assertEqual(f, [("U0", "output", 3, "z")])
        recs = llm_log.read_all()
        self.assertEqual(len(recs), 7)
        self.assertEqual(sum(r["images_sent"] for r in recs), 7)
        self.assertEqual(len(llm_log.read_all(config.CAPTION_RAW_LOG)), 7)
        self.assertAlmostEqual(llm_log.spent_total(provider="gemini"), res["spent_run"], places=6)
        self.assertEqual(llm_log.spent_total(), 0)          # Anthropic 예산에는 섞이지 않는다

    def test_resume_targets_skip_done_and_output_failures(self):
        from test_cards_query import make_card
        from photo_ask.cards_schema import CARD_COLUMNS
        cols = [n for n, _ in CARD_COLUMNS]
        for i, (u, path) in enumerate(self.items):
            card, _ = make_card(u, f"2026-09-2{i}T10:00:00+09:00", 10)
            card["preview_path"] = path
            self.conn.execute(f"INSERT INTO cards ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                              [card[c] for c in cols])
        good = json.dumps(GOOD, ensure_ascii=False)
        post = FakePost([ok_resp(good), ok_resp("x"), ok_resp("y"), ok_resp("z"),
                         Resp(404, {"error": {}})])
        self.run_batch(post, items=self.items[:3])       # U0 성공, U1 output 실패, U2 404(request)
        left = [u for u, _ in enrich_caption.targets(self.conn)]
        self.assertNotIn("U0", left)
        self.assertNotIn("U1", left)                       # 응답 문제는 --retry-failed 때만
        self.assertIn("U2", left)                          # 요청·설정 문제는 다음 실행에서 다시
        self.assertIn("U1", [u for u, _ in enrich_caption.targets(self.conn, retry_failed=True)])

    def test_targets_skip_hidden_by_default_and_screenshots_on_request(self):
        """숨김은 기본으로 클라우드 대상에서 빠지고, 스크린샷은 옵션을 줄 때만 빠진다."""
        from test_cards_query import make_card
        from photo_ask.cards_schema import CARD_COLUMNS
        cols = [n for n, _ in CARD_COLUMNS]
        for u, hidden, shot in [("N1", 0, 0), ("H1", 1, 0), ("S1", 0, 1)]:
            card, _ = make_card(u, "2026-09-20T10:00:00+09:00", 10, hidden=hidden)
            card["screenshot"] = shot
            card["preview_path"] = "/tmp/preview.jpg"
            self.conn.execute(f"INSERT INTO cards ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                              [card[c] for c in cols])
        ids = lambda **kw: sorted(u for u, _ in enrich_caption.targets(self.conn, **kw))  # noqa: E731
        self.assertEqual(ids(), ["N1", "S1"])                                    # 기본: 숨김 제외
        self.assertEqual(ids(skip_screenshots=True), ["N1"])
        self.assertEqual(ids(skip_hidden=False), ["H1", "N1", "S1"])

    def test_budget_stops_before_sending(self):
        post = FakePost([])
        res = self.run_batch(post, budget=0.0005)          # 장당 추정(첫 성공 전 0.001)×1.5 > 상한
        self.assertIn("예산", res["stop_reason"])
        self.assertEqual(post.bodies, [])

    def test_stop_file(self):
        self.stop_file.write_text("")
        post = FakePost([])
        res = self.run_batch(post)
        self.assertIn("정지 파일", res["stop_reason"])
        self.assertEqual(post.bodies, [])

    def test_consecutive_request_errors_stop_early(self):
        post = FakePost([Resp(404, {"error": {"message": "gone"}}) for _ in range(5)])
        res = self.run_batch(post)
        self.assertIn("요청·설정 오류", res["stop_reason"])
        self.assertEqual(len(post.bodies), enrich_caption.REQUEST_FAIL_STOP)   # 나머지 사진은 안 보냈다

    def test_caption_fn_exception_is_recorded_and_batch_continues(self):
        good = json.dumps(GOOD, ensure_ascii=False)

        def caption_fn(uuid, image, log_fn):
            if uuid == "U0":
                raise ValueError("malformed response")
            return caption.CaptionOutcome(uuid=uuid, ok=True, caption=caption.parse_caption(good),
                                          raw_text=good, served_model="test")

        res = self.run_batch(FakePost([]), caption_fn=caption_fn)
        self.assertEqual((res["ok"], res["fail"], res["fail_environment"]), (4, 1, 1))
        f = self.conn.execute("SELECT uuid, error_kind, error FROM caption_failures").fetchall()
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0][0], "U0")
        self.assertIn("ValueError", f[0][2])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0], 4)

    def test_ledger_write_failure_stops_batch_without_more_requests(self):
        """기록·누적 비용을 남기지 못하면 그 사진의 실패로 넘기지 않고 새 제출을 멈춘다(비용을 잃은 채 계속 과금 금지)."""
        good = json.dumps(GOOD, ensure_ascii=False)
        post = FakePost([ok_resp(good) for _ in range(5)])
        with mock.patch.object(llm_log, "append", side_effect=OSError("disk full")):
            res = self.run_batch(post)
        self.assertEqual(len(post.bodies), 1)                 # 첫 요청 뒤 추가 전송 없음
        self.assertTrue(res["ledger_error"])
        self.assertIn("기록 실패", res["stop_reason"])
        self.assertEqual(res["calls"], 0)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0], 0)
        self.assertEqual(enrich_caption.exit_code(res), 5)    # 0이 아닌 종료 코드

    def test_exit_code_mapping(self):
        self.assertEqual(enrich_caption.exit_code({"stop_reason": None, "ledger_error": None}), 0)
        self.assertEqual(enrich_caption.exit_code({"stop_reason": "정지 파일 x", "ledger_error": None}), 0)
        self.assertEqual(enrich_caption.exit_code({"stop_reason": "예산 상한 근접", "ledger_error": None}), 4)
        self.assertEqual(enrich_caption.exit_code({"stop_reason": "기록 실패 — disk", "ledger_error": "disk"}), 5)



class SendPreviewsGateTest(unittest.TestCase):
    """--send-previews 없이는 사진을 보내지 않고 안내만 한 뒤 종료 코드 2 — 네트워크 호출 0."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        db = base / "cards.sqlite"
        from test_cards_query import make_card
        from photo_ask.cards_schema import CARD_COLUMNS
        cols = [n for n, _ in CARD_COLUMNS]
        card, _ = make_card("G0000001-0000", "2026-09-20T10:00:00+09:00", 10)
        card["preview_path"] = str(base / "p.jpeg")
        conn = open_index_rw(db)
        conn.execute(f"INSERT INTO cards ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                     [card[c] for c in cols])
        conn.commit()
        conn.close()
        self.db = db
        self.saved_key = os.environ.pop("GEMINI_API_KEY", None)

    def tearDown(self):
        if self.saved_key is not None:
            os.environ["GEMINI_API_KEY"] = self.saved_key
        self.tmp.cleanup()

    def test_without_flag_prints_what_goes_out_and_returns_2(self):
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["enrich_caption.py", "--index", str(self.db)]), redirect_stdout(out):
            rc = enrich_caption.main()
        self.assertEqual(rc, 2)
        text = out.getvalue()
        self.assertIn("--send-previews", text)
        self.assertIn("Google", text)          # 무엇이 어디로 나가는지 알려 준다
        self.assertIn("미리보기", text)


if __name__ == "__main__":
    unittest.main()
