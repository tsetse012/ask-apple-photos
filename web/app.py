"""사진 찾기 웹 화면 — 127.0.0.1:4100 에서만 듣는 로컬 웹 서버(FastAPI + uvicorn).

- 모든 라우트 HTTP Basic Auth: 아이디 photo, 비밀번호 = 환경변수 PHOTO_ASK_WEB_PASSWORD.
  값은 어디에도 출력하지 않고, 비밀번호가 없거나 16자보다 짧으면 뜨지 않는다.
- 속도 제한(IP = 소켓에 보이는 클라이언트 주소): /api/ask IP당 분당 20회·전체 분당 60회,
  틀린 비밀번호 IP당 분당 20회 — 넘으면 그 IP는 그 1분 동안 모든 요청 429. 미리보기·화면은 인증만(격자 한 번에 수십 장).
- 사진 보관함은 열지 않는다. 미리보기는 index/previews/<uuid>.jpeg 캐시만 읽는다 — 백그라운드 파이썬이 보관함 안 파일을
  열면 권한 문제로 멈출 수 있기 때문. 캐시는 index/build_index.py·index/sync_previews.py가 만든다.
- 질문은 photo_ask.agent(Anthropic, 사진 0장). 호출 기록은 그대로 logs/llm.jsonl, 대화 원문은 logs/transcripts/.
- 세션(되묻기·이어 묻기)은 메모리 dict, 마지막 사용 30분 뒤 만료. 서버를 다시 띄우면 사라진다.

실행: export PHOTO_ASK_WEB_PASSWORD="$(openssl rand -hex 16)" && \
      .venv/bin/uvicorn --factory web.app:create_app --host 127.0.0.1 --port 4100 --no-proxy-headers
"""
import base64
import binascii
import json
import logging
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
from collections import deque
from contextlib import closing
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from photo_ask import config, llm_log, preview_cache  # noqa: E402
from photo_ask.agent import Asker  # noqa: E402
from photo_ask.cards_query import CardIndex  # noqa: E402

USER = "photo"
MIN_PASSWORD_LEN = 16
SESSION_TTL_S = 30 * 60
MAX_SESSIONS = 30
ASK_PER_MIN_IP = 20
ASK_PER_MIN_ALL = 60
AUTHFAIL_PER_MIN_IP = 20
WINDOW_S = 60.0
MAX_TEXT = 300
DEFAULT_LIMIT = 40
MAX_LIMIT = 80
MAX_CONCURRENT_RUNS = 2
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
WEEKDAY_KO = "월화수목금토일"
STOP_KO = {"answered": "답함", "asked": "되물음", "turn_limit": "호출 횟수 상한에서 멈춤",
           "budget": "누적 예산 상한", "query_budget": "질문당 예산 상한", "refusal": "모델 거절",
           "max_tokens": "출력 길이 상한", "error": "오류"}
# 화면의 「더 넓게」「더 좁게」 버튼이 같은 대화에 덧붙이는 말(새 구간 — agent.Asker.resume)
FOLLOW = {
    "wider": "조건을 한 단계 넓혀서 다시 찾아 주세요. 무엇을 넓혔는지 이유에 적어 주세요.",
    "narrower": "지금 후보에서 한 단계 더 좁혀 주세요. 어떤 조건으로 좁혔는지 이유에 적어 주세요.",
}
META_KEYS = ("built_at", "cards", "videos", "hidden", "with_gps", "states", "cities", "build_seconds",
             "last_inserted", "last_updated", "last_removed", "last_unchanged")
HTML_CSP = ("default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

log = logging.getLogger("photo_ask.web")


# ---------------------------------------------------------------- 속도 제한·인증 ----
class RateLimiter:
    """키마다 최근 WINDOW_S초 안의 시각 목록. 서버 하나·사용자 한 명 규모라 메모리로 충분하다."""

    def __init__(self, window: float = WINDOW_S, clock=time.monotonic):
        self.window, self.clock = window, clock
        self._hits = {}
        self._lock = threading.Lock()

    def _q(self, key):
        now = self.clock()
        q = self._hits.setdefault(key, deque())
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(self._hits) > 5000:          # 빈 키 정리(낯선 IP가 많이 올 때 메모리 상한)
            for k in [k for k, v in self._hits.items() if not v and k != key]:
                del self._hits[k]
        return q

    def count(self, key) -> int:
        with self._lock:
            return len(self._q(key))

    def hit(self, key) -> None:
        with self._lock:
            self._q(key).append(self.clock())

    def allow(self, key, limit: int) -> bool:
        """limit 미만이면 한 번 세고 True."""
        with self._lock:
            q = self._q(key)
            if len(q) >= limit:
                return False
            q.append(self.clock())
            return True


def _header(scope, name: bytes) -> Optional[bytes]:
    for k, v in scope.get("headers") or []:
        if k == name:
            return v
    return None


def client_ip(scope) -> str:
    """속도 제한용 클라이언트 주소 = 소켓에 보이는 주소(scope["client"]).
    X-Real-Ip·X-Forwarded-For 같은 헤더는 쓰지 않는다 — 프록시 없이 127.0.0.1에만 듣는 서버에서는
    요청을 보낸 쪽이 헤더를 얼마든지 지어낼 수 있어 IP별 제한을 우회당한다."""
    client = scope.get("client")
    return (client[0] if client else None) or "unknown"


def check_basic(header: Optional[bytes], user: str, password: str) -> bool:
    if not header or not header[:6].lower() == b"basic ":
        return False
    try:
        decoded = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return False
    u, sep, p = decoded.partition(":")
    if not sep:
        return False
    ok_user = secrets.compare_digest(u.encode("utf-8"), user.encode("utf-8"))
    ok_pass = secrets.compare_digest(p.encode("utf-8"), password.encode("utf-8"))
    return ok_user and ok_pass


async def _send(send, status: int, body: bytes, content_type: str, extra: dict = None):
    headers = [(b"content-type", content_type.encode()), (b"content-length", str(len(body)).encode()),
               (b"cache-control", b"no-store")]
    headers += [(k.encode(), v.encode()) for k, v in (extra or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class Guard:
    """모든 HTTP 요청 앞: ① 틀린 비밀번호가 잦은 IP는 429 ② Basic Auth ③ /api/ask 속도 제한."""

    def __init__(self, app, password: str, limiter: RateLimiter):
        self.app, self.password, self.limiter = app, password, limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            # lifespan·websocket은 인증을 거치지 않는다 — 웹소켓 라우트를 추가하면 여기에 인증을 넣어야 한다
            return await self.app(scope, receive, send)
        ip = client_ip(scope)
        if self.limiter.count(f"authfail:{ip}") >= AUTHFAIL_PER_MIN_IP:
            return await _send(send, 429, "요청이 너무 많아요. 1분 뒤 다시 해 주세요.".encode(),
                               "text/plain; charset=utf-8", {"retry-after": "60"})
        auth = _header(scope, b"authorization")
        if not check_basic(auth, USER, self.password):
            if auth:
                self.limiter.hit(f"authfail:{ip}")
            return await _send(send, 401, "로그인이 필요해요.".encode(), "text/plain; charset=utf-8",
                               {"www-authenticate": 'Basic realm="photo-ask", charset="UTF-8"'})
        if scope.get("path") == "/api/ask" and scope.get("method") == "POST":
            if (self.limiter.count("ask:*") >= ASK_PER_MIN_ALL
                    or not self.limiter.allow(f"ask:{ip}", ASK_PER_MIN_IP)):
                body = json.dumps({"ok": False, "code": "rate_limited",
                                   "error": "질문이 너무 잦아요. 1분 뒤 다시 해 주세요."}, ensure_ascii=False)
                return await _send(send, 429, body.encode(), "application/json", {"retry-after": "60"})
            self.limiter.hit("ask:*")
        return await self.app(scope, receive, send)


class SecurityHeaders:
    """모든 응답에 기본 보안 헤더."""

    HEADERS = [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
               (b"referrer-policy", b"no-referrer")]

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def wrapped(message):
            if message["type"] == "http.response.start":
                names = {k for k, _ in message.get("headers", [])}
                message = {**message, "headers": list(message.get("headers", []))
                           + [h for h in self.HEADERS if h[0] not in names]}
            await send(message)

        return await self.app(scope, receive, wrapped)


# ---------------------------------------------------------------- 세션 ----
class Session:
    def __init__(self, sid: str, index: CardIndex, asker: Asker, now: float):
        self.sid, self.index, self.asker = sid, index, asker
        self.conv = None
        self.lock = threading.Lock()         # 한 세션은 한 번에 한 요청만(인덱스 연결을 스레드끼리 나눠 쓰지 않게)
        self.created = self.last_used = now


class SessionsFull(Exception):
    pass


class Sessions:
    """session_id → Session. 세션마다 CardIndex(읽기 전용 연결·결과 묶음)를 따로 연다 — 대화 사이 결과 id가 섞이지 않게."""

    def __init__(self, index_path, asker_factory, clock=time.monotonic, ttl: float = SESSION_TTL_S,
                 max_sessions: int = MAX_SESSIONS):
        self.index_path, self.asker_factory, self.clock = index_path, asker_factory, clock
        self.ttl, self.max_sessions = ttl, max_sessions
        self._d = {}
        self._lock = threading.Lock()

    def __len__(self):
        with self._lock:
            return len(self._d)

    def _close(self, s: Session):
        try:
            s.index.close()
        except Exception:  # noqa: BLE001 — 닫다가 난 오류는 무시(이미 닫힘 등)
            pass

    def _sweep_locked(self):
        now = self.clock()
        for sid, s in list(self._d.items()):
            if now - s.last_used > self.ttl and not s.lock.locked():
                del self._d[sid]
                self._close(s)

    def get(self, sid: str) -> Optional[Session]:
        with self._lock:
            self._sweep_locked()
            s = self._d.get(sid)
            if s:
                s.last_used = self.clock()
            return s

    def new(self) -> Session:
        index = CardIndex(self.index_path, check_same_thread=False)
        asker = self.asker_factory(index)
        with self._lock:
            self._sweep_locked()
            while len(self._d) >= self.max_sessions:
                idle = [s for s in self._d.values() if not s.lock.locked()]
                if not idle:
                    index.close()
                    raise SessionsFull()
                oldest = min(idle, key=lambda s: s.last_used)
                del self._d[oldest.sid]
                self._close(oldest)
            s = Session(secrets.token_urlsafe(18), index, asker, self.clock())
            self._d[s.sid] = s
            return s

    def drop(self, sid: str):
        with self._lock:
            s = self._d.pop(sid, None)
        if s:
            self._close(s)


def default_asker_factory():
    """세션마다 Asker, Anthropic 클라이언트는 하나를 같이 쓴다(처음 쓸 때 만든다)."""
    holder = {}
    lock = threading.Lock()

    def make(index: CardIndex) -> Asker:
        with lock:
            if "client" not in holder:
                import anthropic
                holder["client"] = anthropic.Anthropic(timeout=180.0, max_retries=2)
        return Asker(index, client=holder["client"])

    return make


# ---------------------------------------------------------------- 응답 만들기 ----
def photos_search_term(date_iso: str) -> Optional[str]:
    """사진 앱 검색창에 넣을 날짜 문자열 「2026년 10월 5일」."""
    try:
        return f"{int(date_iso[0:4])}년 {int(date_iso[5:7])}월 {int(date_iso[8:10])}일"
    except (TypeError, ValueError):
        return None


def candidate(index: CardIndex, uuid: str, prefer=()) -> Optional[dict]:
    """후보 한 장. 보관함 파일 경로는 싣지 않는다(미리보기는 /preview/<uuid>)."""
    r = index.card(uuid)
    if r is None:
        return None
    venues = json.loads(r["venues"] or "[]")
    place = " · ".join(x for x in [r["city"] or r["sub_admin"], r["state"] or r["country"]] if x)
    if venues:
        place = f"{place} · {venues[0]}" if place else venues[0]
    cap = index.caption_of(uuid)
    local = r["date_local"] or ""
    capture = r["date_capture"] or local     # 사진 앱은 찍은 곳 날짜로 보여 준다
    try:
        cached = preview_cache.cache_file(uuid).is_file()
    except ValueError:
        cached = False
    return {
        "uuid": uuid,
        "date": local[:16].replace("T", " "),
        "weekday": WEEKDAY_KO[r["weekday"]] if r["weekday"] is not None else None,
        "place": place or None,
        "labels": index.top_labels_for(r, prefer=prefer),
        "scene": (cap or {}).get("scene_ko") or None,
        "cuisine": cap["cuisine"] if cap and cap.get("cuisine") not in (None, "없음") else None,
        "doc_type": cap["doc_type"] if cap and cap.get("doc_type") not in (None, "없음") else None,
        "is_video": bool(r["is_video"]),
        "duration_s": round(r["duration_s"]) if r["is_video"] and r["duration_s"] else None,
        "hidden": bool(r["hidden"]),
        "preview_url": f"/preview/{uuid}" if cached else None,
        "photos_search": photos_search_term(capture),
        "photos_search_local_date": capture[:10] != local[:10],   # 해외 등 서울 날짜와 다를 때
    }


_SUMMARY_HEAD = re.compile(r"^\s*\**\s*요약\s*\**\s*[:：]\s*")


def split_answer(answer: str, why: str):
    """agent의 답(「요약: …」 줄들 + 「이유: …」 줄) → (요약 본문, 이유).
    모델이 「요약:」 앞에 머리말을 쓰면(예: 「QUESTION 줄로 되묻겠습니다.」) agent.parse_final은 그 줄도
    본문에 남긴다 — 화면에는 마지막 「요약:」 줄부터 보인다(명령줄 출력은 그대로)."""
    lines = [ln for ln in (answer or "").splitlines() if not ln.startswith("이유: ")]
    heads = [i for i, ln in enumerate(lines) if i > 0 and _SUMMARY_HEAD.match(ln)]
    if heads:
        lines = lines[heads[-1]:]
    text = _SUMMARY_HEAD.sub("", "\n".join(lines).strip(), count=1)
    return text, (why or "")


def build_payload(sess: Session, leg_cost: float, seconds: float, limit: int, expired: bool = False) -> dict:
    res = sess.conv.res
    idx = sess.index
    filters = idx.results.get(res.result_id, {}).get("filters", {}) if res.result_id else {}
    prefer = list(filters.get("labels_all") or []) + list(filters.get("labels_any") or [])
    cands = [c for c in (candidate(idx, u, prefer) for u in res.uuids[:limit]) if c]
    summary, why = split_answer(res.answer, res.why)
    return {
        "ok": True, "session_id": sess.sid, "query_id": res.query_id,
        "status": res.stop, "status_ko": STOP_KO.get(res.stop, res.stop),
        "summary": summary, "why": why,
        "question": res.clarify_question if res.stop == "asked" else None,
        "total": res.total, "shown": len(cands), "candidates": cands, "fallback": bool(res.result_fallback),
        "can_continue": sess.conv.resumable, "legs": res.legs,
        "cost_usd": round(leg_cost, 5), "seconds": round(seconds, 1), "session_cost_usd": round(res.cost_usd, 5),
        "error": res.error, "session_expired": expired,
    }


class AskIn(BaseModel):
    question: Optional[str] = Field(default=None, max_length=MAX_TEXT)
    session_id: Optional[str] = Field(default=None, max_length=64)
    answer: Optional[str] = Field(default=None, max_length=MAX_TEXT)
    follow: Optional[str] = Field(default=None, pattern="^(wider|narrower)$")
    limit: Optional[int] = Field(default=None, ge=1, le=MAX_LIMIT)


def _ro(path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, timeout=10)


def _err(status: int, msg: str, code: str = None, **extra) -> JSONResponse:
    return JSONResponse({"ok": False, "error": msg, **({"code": code} if code else {}), **extra}, status_code=status)


# ---------------------------------------------------------------- 앱 ----
def create_app(password: str = None, index_path=None, asker_factory=None, clock=time.monotonic) -> FastAPI:
    password = password if password is not None else os.environ.get("PHOTO_ASK_WEB_PASSWORD", "")
    if len(password) < MIN_PASSWORD_LEN:
        raise RuntimeError(f"PHOTO_ASK_WEB_PASSWORD가 없거나 {MIN_PASSWORD_LEN}자보다 짧다 — 비밀번호 없이 뜨지 않는다")
    index_path = Path(index_path or config.INDEX_DB)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False

    app = FastAPI(title="photo-ask", docs_url=None, redoc_url=None, openapi_url=None)
    limiter = RateLimiter(clock=clock)
    sessions = Sessions(index_path, asker_factory or default_asker_factory(), clock=clock)
    run_slots = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)
    app.state.limiter, app.state.sessions = limiter, sessions

    @app.get("/")
    def page():
        return HTMLResponse(STATIC_INDEX.read_text(encoding="utf-8"),
                            headers={"Content-Security-Policy": HTML_CSP, "Cache-Control": "no-store"})

    @app.get("/health")
    def health():
        try:
            with closing(_ro(index_path)) as c:
                cards = c.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
                expected = c.execute("SELECT COUNT(*) FROM cards WHERE preview_path IS NOT NULL "
                                     "AND coalesce(hidden, 0) = 0").fetchone()[0]
                tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                captions = c.execute("SELECT COUNT(*) FROM captions").fetchone()[0] if "captions" in tables else 0
                meta = dict(c.execute("SELECT key, value FROM index_meta").fetchall()) if "index_meta" in tables else {}
        except sqlite3.Error as e:
            return _err(503, f"인덱스를 읽지 못함: {type(e).__name__}", "index_error")
        try:
            spent = round(llm_log.spent_total(), 4)
        except RuntimeError:
            spent = None
        return JSONResponse({
            "ok": cards > 0, "cards": cards, "captions": captions,
            "previews_cached": preview_cache.cached_count(), "previews_expected": expected,
            "index_meta": {k: meta[k] for k in META_KEYS if k in meta},
            "sessions": len(sessions), "anthropic_spent_usd": spent, "anthropic_budget_usd": config.BUDGET_USD,
        }, headers={"Cache-Control": "no-store"})

    @app.get("/preview/{uuid}")
    def preview(uuid: str):
        try:
            path = preview_cache.cache_file(uuid)        # 경로는 uuid 모양 검사를 통과한 값으로만 만든다
        except ValueError:
            return _err(404, "없음")
        with closing(_ro(index_path)) as c:
            if c.execute("SELECT 1 FROM cards WHERE uuid = ?", (uuid,)).fetchone() is None:
                return _err(404, "없음")
        if not path.is_file():
            return _err(404, "미리보기 캐시에 없음(숨김 사진이거나 캐시 동기화 전)", "not_cached")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.post("/api/ask")
    def api_ask(body: AskIn):
        q = (body.question or "").strip()
        a = (body.answer or "").strip()
        limit = body.limit or DEFAULT_LIMIT
        expired = False
        sess = sessions.get(body.session_id) if body.session_id else None
        if body.session_id and sess is None:
            if not q:
                return _err(410, "대화가 끝났어요(30분이 지났거나 서버가 다시 켜짐). 처음부터 다시 물어 주세요.",
                            "session_gone")
            expired = True
        if sess is None:
            if not q:
                return _err(400, "질문을 적어 주세요.", "empty")
            try:
                sess = sessions.new()
            except SessionsFull:
                return _err(503, "동시에 열린 대화가 너무 많아요. 잠시 뒤 다시 해 주세요.", "busy")
            mode, msg = "start", q
        else:
            mode, msg = "resume", (FOLLOW[body.follow] if body.follow else (a or q))
            if not msg:
                return _err(400, "답을 적어 주세요.", "empty")
        if not sess.lock.acquire(blocking=False):
            return _err(409, "이 대화의 앞 요청을 아직 처리하고 있어요.", "in_progress")
        try:
            if mode == "resume" and (sess.conv is None or not sess.conv.resumable):
                stop = sess.conv.res.stop if sess.conv else "none"
                return _err(409, f"이 대화는 더 이어 갈 수 없어요({STOP_KO.get(stop, stop)}). 새로 물어 주세요.",
                            "not_resumable")
            if not run_slots.acquire(blocking=False):
                if mode == "start":
                    sessions.drop(sess.sid)
                return _err(503, "다른 질문을 처리하고 있어요. 잠시 뒤 다시 해 주세요.", "busy")
            cost0 = sess.conv.res.cost_usd if (mode == "resume" and sess.conv) else 0.0
            t0 = time.perf_counter()
            try:
                if mode == "start":
                    sess.conv = sess.asker.start(msg)
                else:
                    sess.asker.resume(sess.conv, msg)
            except Exception as e:  # noqa: BLE001 — 원인은 서버 로그에, 화면엔 종류만
                log.exception("ask failed sid=%s mode=%s", sess.sid[:6], mode)
                if mode == "start":
                    sessions.drop(sess.sid)
                return _err(502, "질문을 처리하다 오류가 났어요. 잠시 뒤 다시 해 주세요.", "agent_error",
                            detail=type(e).__name__)
            finally:
                run_slots.release()
            seconds = time.perf_counter() - t0
            res = sess.conv.res
            log.info("ask %s sid=%s mode=%s status=%s legs=%d total=%d cost=%.5f secs=%.1f",
                     res.query_id, sess.sid[:6], mode, res.stop, res.legs, res.total, res.cost_usd - cost0, seconds)
            return JSONResponse(build_payload(sess, res.cost_usd - cost0, seconds, limit, expired),
                                headers={"Cache-Control": "no-store"})
        finally:
            sess.lock.release()

    # 바깥부터: SecurityHeaders → Guard(인증·속도 제한) → 앱. add_middleware는 나중에 넣은 것이 바깥이 된다.
    app.add_middleware(Guard, password=password, limiter=limiter)
    app.add_middleware(SecurityHeaders)
    return app
