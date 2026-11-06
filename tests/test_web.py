"""폰 웹 화면(web/app.py)을 FastAPI TestClient + 가짜 Anthropic 클라이언트로 검사(과금 0·사진 전송 0·보관함 접근 0)."""
import json
import logging
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from photo_ask import agent, config  # noqa: E402
from photo_ask.cards_schema import CARD_COLUMNS, open_index_rw  # noqa: E402
from test_agent_loop import FakeClient, response, text, tool  # noqa: E402
from test_cards_query import make_card  # noqa: E402
from web import app as web_app  # noqa: E402

COLS = [n for n, _ in CARD_COLUMNS]
PW = "test-password-0123456789"
JPEG = b"\xff\xd8\xff\xe0" + b"fake-preview" * 20
C1, C2, C3, C4 = "C0000001-0000", "C0000002-0000", "C0000003-0000", "C0000004-0000"
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")


class AlwaysAnswer:
    """몇 번을 불러도 같은 답(속도 제한 검사용)."""

    def __init__(self):
        self.n = 0
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.n += 1
        return response([text("요약: 후보 없음\n이유: -\nRESULT: none")], "end_turn")


class Boom:
    def __init__(self):
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        raise RuntimeError("network down")


class WebTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.saved = (config.LLM_LOG, config.TRANSCRIPT_DIR, config.PREVIEW_CACHE_DIR)
        config.LLM_LOG = tmp / "llm.jsonl"
        config.TRANSCRIPT_DIR = tmp / "transcripts"
        config.PREVIEW_CACHE_DIR = tmp / "previews"
        config.PREVIEW_CACHE_DIR.mkdir()
        (config.PREVIEW_CACHE_DIR / f"{C1}.jpeg").write_bytes(JPEG)     # C1만 캐시에 있다
        self.db = tmp / "cards.sqlite"
        conn = open_index_rw(self.db)
        cards = [
            make_card(C1, "2026-09-20T13:00:00+09:00", 13, "가상특별자치도", "테스트시", labels=["음식점", "커피잔"]),
            make_card(C2, "2026-05-01T13:00:00+09:00", 13, "가상특별자치도", "테스트시", labels=["음식점"], video=1),
            make_card(C3, "2026-09-22T12:00:00+09:00", 12, "가상특별자치도", "테스트시", labels=["음식점"], hidden=1),
            make_card(C4, "2026-08-02T01:00:00+09:00", 18, "가상주", "가상도시", labels=["해변"]),
        ]
        for card, tags in cards:
            # 보관함 경로는 일부러 없는 곳을 가리킨다 — 서버가 보관함을 열지 않고 캐시만 읽는지 보려고
            card["preview_path"] = f"/nonexistent-library/{card['uuid']}.jpeg"
            if card["uuid"] == C2:
                card["duration_s"] = 12.4
            if card["uuid"] == C4:
                card["date_capture"] = "2026-08-01T18:00:00+02:00"   # 찍은 곳 날짜가 서울 날짜와 다름
            conn.execute(f"INSERT INTO cards ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})",
                         [card[c] for c in COLS])
            conn.executemany("INSERT INTO card_tags VALUES (?,?,?)", [(card["uuid"], t, k) for t, k in tags])
        conn.execute("INSERT INTO captions (uuid, model, json, created_at, scene_ko, cuisine, doc_type) "
                     "VALUES (?, 'fake', '{}', '2026-10-08', '탁자 위 커피잔과 파스타', '양식', '없음')", (C1,))
        conn.execute("INSERT INTO index_meta VALUES ('built_at', '2026-10-08T17:00:00+09:00'), "
                     "('cards', '4'), ('library', '/should/not/leak')")
        conn.commit()
        conn.close()
        self.now = [1000.0]
        self.fake = FakeClient([])
        self.app = web_app.create_app(password=PW, index_path=self.db, clock=lambda: self.now[0],
                                      asker_factory=lambda index: agent.Asker(index, client=self.fake))
        self.c = TestClient(self.app)
        self.auth = ("photo", PW)
        web_app.log.setLevel(logging.WARNING)      # 테스트 중 요청 로그는 끈다

    def tearDown(self):
        self.c.close()
        config.LLM_LOG, config.TRANSCRIPT_DIR, config.PREVIEW_CACHE_DIR = self.saved
        self.tmp.cleanup()

    def ask(self, **body):
        return self.c.post("/api/ask", json=body, auth=self.auth)

    # ---- 인증 ----
    def test_every_route_requires_basic_auth(self):
        for method, path in [("GET", "/"), ("GET", "/health"), ("GET", f"/preview/{C1}"), ("POST", "/api/ask"),
                             ("GET", "/docs"), ("GET", "/openapi.json")]:
            r = self.c.request(method, path, json={"question": "x"} if method == "POST" else None)
            self.assertEqual(r.status_code, 401, path)
            self.assertIn("Basic", r.headers["www-authenticate"])
            r = self.c.request(method, path, auth=("photo", PW + "x"))
            self.assertEqual(r.status_code, 401, path)
            r = self.c.request(method, path, auth=("other", PW))
            self.assertEqual(r.status_code, 401, path)
        self.assertEqual(self.fake.requests, [])

    def test_docs_are_disabled(self):
        for path in ("/docs", "/redoc", "/openapi.json"):
            self.assertEqual(self.c.get(path, auth=self.auth).status_code, 404, path)

    def test_create_app_refuses_without_password(self):
        for pw in ("", "short"):
            with self.assertRaises(RuntimeError):
                web_app.create_app(password=pw, index_path=self.db)

    def test_page_headers_and_content(self):
        r = self.c.get("/", auth=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("사진 찾기", r.text)
        self.assertIn("default-src 'none'", r.headers["content-security-policy"])
        self.assertEqual(r.headers["x-frame-options"], "DENY")
        self.assertEqual(r.headers["cache-control"], "no-store")

    def test_page_has_no_emoji_and_no_external_resources(self):
        html = web_app.STATIC_INDEX.read_text(encoding="utf-8")
        self.assertIsNone(EMOJI.search(html))
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertIn("prefers-color-scheme: dark", html)

    # ---- health ----
    def test_health(self):
        r = self.c.get("/health", auth=self.auth)
        self.assertEqual(r.status_code, 200)
        h = r.json()
        self.assertTrue(h["ok"])
        self.assertEqual((h["cards"], h["captions"], h["previews_cached"], h["previews_expected"]), (4, 1, 1, 3))
        self.assertEqual(h["index_meta"], {"built_at": "2026-10-08T17:00:00+09:00", "cards": "4"})
        self.assertNotIn("/should/not/leak", r.text)        # 로컬 경로 같은 메타는 싣지 않는다

    # ---- 미리보기 ----
    def test_preview_comes_from_cache_not_library(self):
        r = self.c.get(f"/preview/{C1}", auth=self.auth)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/jpeg")
        self.assertEqual(r.content, JPEG)                  # 카드의 preview_path는 없는 경로 → 캐시에서 읽었다
        self.assertIn("private", r.headers["cache-control"])

    def test_preview_404s_and_no_path_tricks(self):
        self.assertEqual(self.c.get("/preview/D0000009-0000", auth=self.auth).status_code, 404)   # 인덱스에 없음
        r = self.c.get(f"/preview/{C2}", auth=self.auth)                                         # 인덱스엔 있고 캐시엔 없음
        self.assertEqual((r.status_code, r.json()["code"]), (404, "not_cached"))
        (Path(self.tmp.name) / "secret.jpeg").write_bytes(b"secret")
        for bad in ("..%2Fsecret", "..%2F..%2Fsecret", f"{C1}.jpeg", "a%20b", "secret", "%00", "%2e%2e%2fsecret"):
            r = self.c.get(f"/preview/{bad}", auth=self.auth)
            self.assertEqual(r.status_code, 404, bad)
            self.assertNotEqual(r.content, b"secret", bad)
        # 「..」·「%2e%2e」는 클라이언트(httpx)가 점 세그먼트를 지워 GET / 가 된다 — 미리보기·파일이 나가지 않으면 된다
        for bad in ("..", "%2e%2e"):
            r = self.c.get(f"/preview/{bad}", auth=self.auth)
            self.assertNotIn("image/jpeg", r.headers.get("content-type", ""), bad)
            self.assertNotEqual(r.content, b"secret", bad)

    # ---- 질문 ----
    def test_ask_answered_payload(self):
        self.fake.script = [
            response([tool("t1", "search_cards", {"labels_any": ["커피잔"]})], "tool_use"),
            response([text("요약: 후보 1건 — 9월 테스트시.\n이유: 커피잔 라벨\nRESULT: r1")], "end_turn"),
        ]
        r = self.ask(question="커피 마신 사진")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual((d["status"], d["total"], d["shown"], d["legs"]), ("answered", 1, 1, 1))
        self.assertEqual(d["summary"], "후보 1건 — 9월 테스트시.")
        self.assertEqual(d["why"], "커피잔 라벨")
        self.assertIsNone(d["question"])
        self.assertTrue(d["can_continue"])
        self.assertGreater(d["cost_usd"], 0)
        c = d["candidates"][0]
        self.assertEqual(c["uuid"], C1)
        self.assertEqual(c["photos_search"], "2026년 9월 20일")
        self.assertFalse(c["photos_search_local_date"])
        self.assertEqual(c["preview_url"], f"/preview/{C1}")
        self.assertEqual(c["scene"], "탁자 위 커피잔과 파스타")
        self.assertEqual(c["cuisine"], "양식")
        self.assertIn("테스트시", c["place"])
        self.assertEqual(c["labels"][0], "커피잔")          # 질문 조건에 걸린 라벨 먼저
        self.assertNotIn("preview_path", r.text)
        self.assertNotIn("nonexistent-library", r.text)    # 보관함 경로는 화면으로 안 나간다
        self.assertEqual(json.loads(Path(config.LLM_LOG).read_text().splitlines()[0])["images_sent"], 0)

    def test_overseas_photo_uses_capture_date_and_video_fields(self):
        self.fake.script = [
            response([tool("t1", "search_cards", {"labels_any": ["해변", "음식점"]})], "tool_use"),
            response([text("요약: 후보 2건\n이유: 라벨\nRESULT: r1")], "end_turn"),
        ]
        d = self.ask(question="바다").json()
        by = {c["uuid"]: c for c in d["candidates"]}
        self.assertEqual(set(by), {C1, C2, C4})            # 숨김(C3)은 기본 검색에서 빠진다
        self.assertEqual(by[C4]["photos_search"], "2026년 8월 1일")
        self.assertTrue(by[C4]["photos_search_local_date"])
        self.assertIsNone(by[C4]["preview_url"])           # 캐시에 없으면 URL을 주지 않는다
        self.assertTrue(by[C2]["is_video"])
        self.assertEqual(by[C2]["duration_s"], 12)

    def test_clarify_roundtrip_keeps_one_question(self):
        self.fake.script = [
            response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
            response([text("요약: 후보 2건\n이유: 라벨\nRESULT: r1\nQUESTION: 언제쯤인가요? (9월 1 / 5월 1)")], "end_turn"),
            response([tool("t2", "search_cards", {"labels_any": ["음식점"], "date_from": "2026-09-01"})], "tool_use"),
            response([text("요약: 후보 1건\n이유: 9월로 좁힘\nRESULT: r2")], "end_turn"),
        ]
        d1 = self.ask(question="음식 사진").json()
        self.assertEqual((d1["status"], d1["total"]), ("asked", 2))
        self.assertEqual(d1["question"], "언제쯤인가요? (9월 1 / 5월 1)")
        self.assertTrue(d1["can_continue"])
        d2 = self.ask(session_id=d1["session_id"], answer="9월").json()
        self.assertEqual((d2["status"], d2["total"], d2["legs"]), ("answered", 1, 1))
        self.assertEqual(d2["query_id"], d1["query_id"])
        self.assertEqual(d2["session_id"], d1["session_id"])
        self.assertEqual(self.fake.requests[2]["messages"][-1], {"role": "user", "content": "9월"})
        tr = json.loads((config.TRANSCRIPT_DIR / f"{d1['query_id']}.json").read_text())
        self.assertEqual(tr["result"]["user_reply"], "9월")
        self.assertEqual(tr["result"]["stop"], "answered")

    def test_follow_up_wider_is_a_new_leg(self):
        self.fake.script = [
            response([tool("t1", "search_cards", {"labels_any": ["커피잔"]})], "tool_use"),
            response([text("요약: 1건\n이유: 라벨\nRESULT: r1")], "end_turn"),
            response([tool("t2", "search_cards", {"labels_any": ["커피잔", "음식점"]})], "tool_use"),
            response([text("요약: 2건\n이유: 라벨을 넓힘\nRESULT: r2")], "end_turn"),
        ]
        d1 = self.ask(question="커피").json()
        d2 = self.ask(session_id=d1["session_id"], follow="wider").json()
        self.assertEqual((d2["status"], d2["total"], d2["legs"]), ("answered", 2, 2))
        self.assertEqual(self.fake.requests[2]["messages"][-1]["content"], web_app.FOLLOW["wider"])
        self.assertAlmostEqual(d2["session_cost_usd"], d1["cost_usd"] + d2["cost_usd"], places=5)

    def test_session_expires_after_30_minutes(self):
        self.fake.script = [response([text("요약: 없음\n이유: -\nRESULT: none")], "end_turn"),
                            response([text("요약: 없음\n이유: -\nRESULT: none")], "end_turn")]
        d1 = self.ask(question="아무거나").json()
        self.now[0] += 31 * 60
        r = self.ask(session_id=d1["session_id"], answer="더")
        self.assertEqual((r.status_code, r.json()["code"]), (410, "session_gone"))
        d2 = self.ask(session_id=d1["session_id"], question="새 질문").json()
        self.assertTrue(d2["session_expired"])
        self.assertNotEqual(d2["session_id"], d1["session_id"])

    def test_not_resumable_after_refusal(self):
        resp = response([text("")], "refusal")
        resp.stop_details = SimpleNamespace(category="test")
        self.fake.script = [resp]
        d = self.ask(question="x").json()
        self.assertEqual(d["status"], "refusal")
        self.assertFalse(d["can_continue"])
        r = self.ask(session_id=d["session_id"], follow="narrower")
        self.assertEqual((r.status_code, r.json()["code"]), (409, "not_resumable"))

    def test_same_session_busy_returns_409(self):
        self.fake.script = [response([text("요약: 없음\n이유: -\nRESULT: none")], "end_turn")]
        d = self.ask(question="x").json()
        sess = self.app.state.sessions.get(d["session_id"])
        self.assertTrue(sess.lock.acquire(blocking=False))
        try:
            r = self.ask(session_id=d["session_id"], follow="wider")
            self.assertEqual((r.status_code, r.json()["code"]), (409, "in_progress"))
        finally:
            sess.lock.release()

    def test_agent_error_returns_502_and_drops_session(self):
        app = web_app.create_app(password=PW, index_path=self.db,
                                 asker_factory=lambda index: agent.Asker(index, client=Boom()))
        with TestClient(app) as c:
            r = c.post("/api/ask", json={"question": "x"}, auth=self.auth)
        self.assertEqual((r.status_code, r.json()["code"], r.json()["detail"]), (502, "agent_error", "RuntimeError"))
        self.assertEqual(len(app.state.sessions), 0)

    def test_input_validation(self):
        self.assertEqual(self.ask().status_code, 400)
        self.assertEqual(self.ask(question="   ").status_code, 400)
        self.assertEqual(self.ask(question="가" * 301).status_code, 422)
        self.assertEqual(self.ask(question="x", follow="sideways").status_code, 422)
        self.assertEqual(self.ask(question="x", limit=999).status_code, 422)
        self.assertEqual(self.ask(session_id="nope", answer="x").status_code, 410)
        self.assertEqual(self.fake.requests, [])

    # ---- 속도 제한 ----
    def test_ask_rate_limit_per_ip_with_window(self):
        always = AlwaysAnswer()
        app = web_app.create_app(password=PW, index_path=self.db, clock=lambda: self.now[0],
                                 asker_factory=lambda index: agent.Asker(index, client=always))
        ip = ["1.1.1.1"]
        with mock.patch.object(web_app, "client_ip", lambda scope: ip[0]):
            with TestClient(app) as c:
                post = lambda: c.post("/api/ask", json={"question": "x"}, auth=self.auth)  # noqa: E731
                codes = [post().status_code for _ in range(web_app.ASK_PER_MIN_IP)]
                self.assertEqual(set(codes), {200})
                r = post()
                self.assertEqual((r.status_code, r.json()["code"]), (429, "rate_limited"))
                self.assertEqual(r.headers["retry-after"], "60")
                ip[0] = "2.2.2.2"
                self.assertEqual(post().status_code, 200)      # IP마다 따로 센다
                ip[0] = "1.1.1.1"
                self.now[0] += 61
                self.assertEqual(post().status_code, 200)      # 1분이 지나면 풀린다
        self.assertEqual(always.n, web_app.ASK_PER_MIN_IP + 2)       # 429는 모델을 부르지 않는다

    def test_wrong_password_lockout_per_ip(self):
        ip = ["3.3.3.3"]
        with mock.patch.object(web_app, "client_ip", lambda scope: ip[0]):
            get = lambda pw: self.c.get("/health", auth=("photo", pw))  # noqa: E731
            for _ in range(web_app.AUTHFAIL_PER_MIN_IP + 3):            # 자격 증명 없는 요청은 세지 않는다
                self.assertEqual(self.c.get("/health").status_code, 401)
            for _ in range(web_app.AUTHFAIL_PER_MIN_IP):
                self.assertEqual(get("wrong-password-xxxxxxxx").status_code, 401)
            self.assertEqual(get(PW).status_code, 429)                  # 맞는 비밀번호여도 그 1분은 막힘
            ip[0] = "4.4.4.4"
            self.assertEqual(get(PW).status_code, 200)
            ip[0] = "3.3.3.3"
            self.now[0] += 61
            self.assertEqual(get(PW).status_code, 200)

    def test_client_ip_uses_socket_address_and_ignores_headers(self):
        # 프록시 없는 로컬 실행에서는 X-Real-Ip·X-Forwarded-For 를 믿으면 IP별 제한을 우회당한다.
        scope = {"headers": [(b"x-forwarded-for", b"5.5.5.5, 6.6.6.6"), (b"x-real-ip", b"7.7.7.7")],
                 "client": ("127.0.0.1", 1)}
        self.assertEqual(web_app.client_ip(scope), "127.0.0.1")
        scope["headers"] = []
        scope["client"] = ("10.0.0.2", 555)
        self.assertEqual(web_app.client_ip(scope), "10.0.0.2")
        self.assertEqual(web_app.client_ip({"headers": [], "client": None}), "unknown")

    def test_split_answer_and_search_term(self):
        self.assertEqual(web_app.split_answer("요약: 후보 3건\n둘째 줄\n이유: 라벨", "라벨"), ("후보 3건\n둘째 줄", "라벨"))
        # 모델 머리말이 앞에 붙은 경우(parse_final이 「요약: 머리말」+「요약: 본문」 두 줄로 넘김) — 화면엔 본문만
        self.assertEqual(web_app.split_answer("요약: QUESTION 줄로 되묻겠습니다.\n요약: 후보 42건\n이유: 도시", "도시"),
                         ("후보 42건", "도시"))
        self.assertEqual(web_app.split_answer("요약: 약국: 근처 3건", ""), ("약국: 근처 3건", ""))
        self.assertEqual(web_app.photos_search_term("2026-10-05T14:03:00+09:00"), "2026년 10월 5일")
        self.assertIsNone(web_app.photos_search_term(""))


if __name__ == "__main__":
    unittest.main()
