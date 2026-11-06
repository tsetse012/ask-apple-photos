"""카드 인덱스 검색 도구 3종: search_cards · stats · list_vocab.

LLM 도구 호출과 단위 테스트가 같은 함수를 쓴다. 결과에는 「없다(0건)」와
「못 찾은 이름(unknown)」을 구분해 담는다 — 라벨·장소 이름이 인덱스에 없으면 조건을
조용히 빼지 않고 0건으로 두고 unknown에 후보를 적는다.

2단계 보강 표를 쓰는 필터: cuisine·doc_type·caption_like(captions — 클라우드 VLM 캡션),
vision_labels_any(+vision_min_conf, 공개 Vision 분류기), apple_label_min_conf(label_scores — 사진 앱 자체 확신도),
text_like는 우리 OCR(ocr_text)·캡션 글자 조각까지. 보강 표가 없는 인덱스에서도 돌아간다(그 조건은 0건+unknown).
"""
import json
import re
import sqlite3
from collections import Counter
from datetime import date

from .caption import CUISINES, DOC_TYPES
from .cards_schema import open_index_ro

WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]
PLACE_SUFFIXES = sorted(["특별자치도", "특별자치시", "특별시", "광역시", "자치구", "자치시",
                         "도", "시", "군", "구", "현"], key=len, reverse=True)
SORTS = {"date_desc", "date_asc", "aesthetic", "faces", "text_len", "relevance"}
ROW_LIMIT_MAX = 30
VOCAB_LIMIT_MAX = 400
VISION_MIN_CONF = 0.05     # 자체 측정에서 이 문턱이면 동물 라벨을 충분히 잡고 비동물 거짓양성이 적었다
VISION_FLOOR = 0.02        # vision_labels·label_scores에 저장된 하한
CUISINE_ALIASES = {"서양식": "양식", "서양": "양식", "서양요리": "양식", "한국": "한식", "한국음식": "한식",
                   "일본": "일식", "일본음식": "일식", "중국": "중식", "중국음식": "중식", "태국": "동남아",
                   "베트남": "동남아", "동남아식": "동남아", "카페": "카페·디저트", "디저트": "카페·디저트",
                   "카페/디저트": "카페·디저트", "카페디저트": "카페·디저트", "카페 디저트": "카페·디저트"}


def place_core(name: str) -> str:
    """행정구역 꼬리(도·시·군·특별자치도…)와 공백을 뗀 이름. 「강원도」와 「강원특별자치도」를 같게 본다."""
    s = re.sub(r"\s+", "", name or "").lower()
    for suf in PLACE_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return s


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    return [str(x) for x in v if str(x).strip()]


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class FilterError(ValueError):
    """도구 입력이 잘못됐을 때. 메시지는 그대로 LLM에 돌려준다."""


class CardIndex:
    def __init__(self, db_path, check_same_thread: bool = True):
        self.conn: sqlite3.Connection = open_index_ro(db_path, check_same_thread=check_same_thread)
        self.results = {}          # result_id -> {"uuids": [...], "total": n, "filters": {...}}
        self._next_id = 1
        self._tag_counts = None
        self._place_values = None
        self._score_names = None
        self._vision_names = None
        self.tables = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def has(self, table: str) -> bool:
        """보강 표가 있고 비어 있지 않은가(보강 표가 없으면 False)."""
        if table not in self.tables:
            return False
        return self.conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None

    def reset_results(self):
        self.results = {}
        self._next_id = 1

    # ---------- 어휘 캐시 ----------
    def tag_counts(self) -> dict:
        """{(tag, kind): n} — 숨김 사진 포함 전체."""
        if self._tag_counts is None:
            self._tag_counts = {(t, k): n for t, k, n in self.conn.execute(
                "SELECT tag, kind, COUNT(*) FROM card_tags GROUP BY tag, kind")}
        return self._tag_counts

    def place_values(self) -> dict:
        """{column: Counter(value)} for state·city·sub_admin·district·country."""
        if self._place_values is None:
            pv = {}
            for col in ("country", "state", "city", "sub_admin", "district"):
                pv[col] = Counter(dict(self.conn.execute(
                    f"SELECT {col}, COUNT(*) FROM cards WHERE {col} IS NOT NULL GROUP BY {col}")))
            self._place_values = pv
        return self._place_values

    def _resolve_places(self, wanted, columns):
        """요청한 장소 이름 → 인덱스에 실제 있는 값들. (matched, unknown)"""
        values = Counter()
        for col in columns:
            values.update(self.place_values()[col])
        matched, unknown = [], []
        for w in wanted:
            core = place_core(w)
            hits = [v for v in values if v == w or (core and place_core(v) == core)]
            if not hits and len(core) >= 2:
                hits = [v for v in values if core in place_core(v)]
            if hits:
                matched.extend(h for h in hits if h not in matched)
            else:
                unknown.append(w)
        return matched, unknown

    def score_names(self) -> Counter:
        """label_scores의 한국어 이름별 사진 수(확신도 무관). 현재 카드와 조인해 고아 행은 세지 않는다."""
        if self._score_names is None:
            self._score_names = Counter(dict(self.conn.execute(
                "SELECT ls.label_ko, COUNT(*) FROM label_scores ls JOIN cards c ON c.uuid = ls.uuid "
                "GROUP BY ls.label_ko"))) if "label_scores" in self.tables else Counter()
        return self._score_names

    def vision_names(self) -> dict:
        """{이름(한국어 또는 영어 id): [영어 id…]} — vision_labels에 실제 있는 라벨(한국어 이름이 겹치면 여러 id).
        현재 카드와 조인해 고아 행은 빼고 센다."""
        if self._vision_names is None:
            names = {}
            if "vision_labels" in self.tables:
                for en, ko in self.conn.execute(
                        "SELECT DISTINCT v.label_en, v.label_ko FROM vision_labels v "
                        "JOIN cards c ON c.uuid = v.uuid"):
                    names.setdefault(en, [])
                    if en not in names[en]:
                        names[en].append(en)
                    if ko:
                        names.setdefault(ko, [])
                        if en not in names[ko]:
                            names[ko].append(en)
            self._vision_names = names
        return self._vision_names

    def _resolve_tags(self, wanted, with_scores=False):
        """요청 라벨 → (있는 라벨, {없는 라벨: 비슷한 후보}). with_scores면 label_scores 이름도 있는 것으로 본다."""
        vocab = Counter()
        for (t, _k), n in self.tag_counts().items():
            vocab[t] += n
        if with_scores:
            for t, n in self.score_names().items():
                vocab[t] = max(vocab[t], n)
        ok, unknown = [], {}
        for w in wanted:
            if w in vocab:
                ok.append(w)
                continue
            cands = Counter()
            for t, n in vocab.items():
                if w in t or (len(t) >= 2 and t in w):
                    cands[t] += n
            unknown[w] = [t for t, _ in cands.most_common(6)]
        return ok, unknown

    def _resolve_vision(self, wanted):
        """요청 Vision 라벨(한국어 이름 또는 영어 id) → (영어 id 목록, {없는 것: 후보})."""
        names = self.vision_names()
        ok, unknown = [], {}
        for w in wanted:
            ens = names.get(w) or names.get(w.lower().replace(" ", "_"))
            if ens:
                ok += [e for e in ens if e not in ok]
                continue
            cands = [n for n in names if w in n or (len(n) >= 2 and n in w)]
            unknown[w] = sorted(cands, key=len)[:6]
        return ok, unknown

    @staticmethod
    def _enum_values(raw, valid, aliases=None):
        """'a|b'·목록 → (유효 값, 목록 밖 값)."""
        vals = []
        for v in _as_list(raw):
            vals += [x.strip() for x in re.split(r"[|,]", v) if x.strip()]
        ok, bad = [], []
        for v in vals:
            v2 = (aliases or {}).get(v, v)
            (ok if v2 in valid else bad).append(v2 if v2 in valid else v)
        return list(dict.fromkeys(ok)), bad

    def _label_expr(self, conf):
        """한 라벨의 '있다' 판정 SQL(파라미터: 라벨[, 라벨, 확신도]). conf가 있으면 사진 앱 확신도 표까지."""
        base = "EXISTS (SELECT 1 FROM card_tags t WHERE t.uuid = c.uuid AND t.tag = ?)"
        if conf is None:
            return base, 1
        return (f"({base} OR EXISTS (SELECT 1 FROM label_scores ls WHERE ls.uuid = c.uuid AND ls.label_ko = ? "
                "AND ls.confidence >= ?))"), 3

    # ---------- 필터 → SQL ----------
    def _where(self, f: dict):
        """필터 dict → (where_sql, params, info). 모르는 이름이 섞인 조건은 0건이 되게 만든다."""
        clauses, params = [], []
        info = {"resolved": {}, "unknown": {}}

        if not f.get("include_hidden"):
            clauses.append("c.hidden = 0")

        for key, op in (("date_from", ">="), ("date_to", "<=")):
            v = f.get(key)
            if v:
                try:
                    date.fromisoformat(v)
                except ValueError:
                    raise FilterError(f"{key}는 YYYY-MM-DD 형식이어야 합니다: {v!r}")
                clauses.append(f"substr(c.date_local, 1, 10) {op} ?")
                params.append(v)

        states = _as_list(f.get("states"))
        if states:
            matched, unknown = self._resolve_places(states, ["state"])
            if unknown:
                info["unknown"]["states"] = unknown
            if matched:
                info["resolved"]["states"] = matched
                clauses.append(f"c.state IN ({','.join('?' * len(matched))})")
                params += matched
            else:
                clauses.append("0")

        cities = _as_list(f.get("cities"))
        if cities:
            matched, unknown = self._resolve_places(cities, ["city", "sub_admin", "district"])
            if unknown:
                info["unknown"]["cities"] = unknown
            if matched:
                info["resolved"]["cities"] = matched
                q = ",".join("?" * len(matched))
                clauses.append(f"(c.city IN ({q}) OR c.sub_admin IN ({q}) OR c.district IN ({q}))")
                params += matched * 3
            else:
                clauses.append("0")

        # 사진 내용 조건(labels_any · vision_labels_any · caption_like)은 따로 모았다가 content_match에 따라
        # AND(all, 기본) 또는 OR(any — 「둘 중 하나라도 맞으면」, 전부 찾기용)로 붙인다.
        content = []
        mode = f.get("content_match") or "all"
        if mode not in ("all", "any"):
            raise FilterError("content_match는 all 또는 any")
        conf = f.get("apple_label_min_conf")
        if conf is not None:
            conf = float(conf)
            if not (VISION_FLOOR <= conf <= 1):
                raise FilterError(f"apple_label_min_conf는 {VISION_FLOOR}~1 (저장된 하한이 {VISION_FLOOR})")
            if "label_scores" not in self.tables:
                info["unknown"]["apple_label_min_conf"] = "label_scores 표 없음(build_index.py 다시 실행)"
                conf = None
        for key in ("labels_any", "labels_all", "exclude_labels"):
            wanted = _as_list(f.get(key))
            if not wanted:
                continue
            ok, unknown = self._resolve_tags(wanted, with_scores=conf is not None)
            if unknown:
                info["unknown"].setdefault("labels", {}).update(unknown)
            expr, nparam = self._label_expr(conf)

            def p_of(tag):
                return [tag] if nparam == 1 else [tag, tag, conf]
            if key == "labels_any":
                if ok:
                    content.append(("(" + " OR ".join(expr for _ in ok) + ")",
                                    [x for tag in ok for x in p_of(tag)]))
                else:
                    content.append(("0", []))
            elif key == "labels_all":
                if unknown:
                    clauses.append("0")
                for tag in ok:
                    clauses.append(expr)
                    params += p_of(tag)
            else:
                for tag in ok:
                    clauses.append(f"NOT {expr}")
                    params += p_of(tag)

        vis = _as_list(f.get("vision_labels_any"))
        if vis:
            vmin = f.get("vision_min_conf")
            vmin = VISION_MIN_CONF if vmin is None else float(vmin)
            if not (VISION_FLOOR <= vmin <= 1):
                raise FilterError(f"vision_min_conf는 {VISION_FLOOR}~1 (저장된 하한이 {VISION_FLOOR})")
            ok, unknown = self._resolve_vision(vis)
            if unknown:
                info["unknown"].setdefault("vision_labels", {}).update(unknown)
            if ok:
                info["resolved"]["vision_labels"] = ok
                content.append(("EXISTS (SELECT 1 FROM vision_labels v WHERE v.uuid = c.uuid AND v.label_en IN "
                                f"({','.join('?' * len(ok))}) AND v.confidence >= ?)", ok + [vmin]))
            else:
                content.append(("0", []))

        for key, col, valid, aliases in (("cuisine", "cuisine", CUISINES, CUISINE_ALIASES),
                                         ("doc_type", "doc_type", DOC_TYPES, None)):
            if not f.get(key):
                continue
            ok, bad = self._enum_values(f.get(key), valid, aliases)
            if bad:
                info["unknown"][key] = {"목록 밖": bad, "쓸 수 있는 값": valid}
            if ok and "captions" in self.tables:
                clauses.append(f"EXISTS (SELECT 1 FROM captions p WHERE p.uuid = c.uuid AND p.{col} IN "
                               f"({','.join('?' * len(ok))}))")
                params += ok
            else:
                if "captions" not in self.tables:
                    info["unknown"][key] = "captions 표 없음(index/enrich_caption.py 실행 필요)"
                clauses.append("0")

        cap_text = f.get("caption_like")
        if cap_text:
            alts = [a.strip().lower() for a in str(cap_text).split("|") if a.strip()]
            if "captions" not in self.tables:
                info["unknown"]["caption_like"] = "captions 표 없음(index/enrich_caption.py 실행 필요)"
                content.append(("0", []))
            elif alts:
                ors = ["p.search_blob LIKE ? ESCAPE '\\'" for _ in alts]
                content.append(("EXISTS (SELECT 1 FROM captions p WHERE p.uuid = c.uuid AND (" + " OR ".join(ors) + "))",
                                [f"%{_like_escape(a)}%" for a in alts]))

        text = f.get("text_like")
        if text:
            alts = [a.strip().lower() for a in str(text).split("|") if a.strip()]
            ors = []
            for a in alts:
                pat = f"%{_like_escape(a)}%"
                parts = ["c.search_text LIKE ? ESCAPE '\\'", "lower(coalesce(c.detected_text, '')) LIKE ? ESCAPE '\\'"]
                params += [pat, pat]
                if "ocr_text" in self.tables:
                    parts.append("EXISTS (SELECT 1 FROM ocr_text o WHERE o.uuid = c.uuid AND lower(o.text) LIKE ? ESCAPE '\\')")
                    params.append(pat)
                if "captions" in self.tables:
                    parts.append("EXISTS (SELECT 1 FROM captions p WHERE p.uuid = c.uuid "
                                 "AND lower(p.text_snippet) LIKE ? ESCAPE '\\')")
                    params.append(pat)
                ors.append("(" + " OR ".join(parts) + ")")
            if ors:
                clauses.append("(" + " OR ".join(ors) + ")")

        for key, col in (("is_video", "is_video"), ("favorite", "favorite"), ("screenshot", "screenshot")):
            v = f.get(key)
            if v is not None:
                clauses.append(f"c.{col} = ?")
                params.append(1 if v else 0)

        if f.get("min_faces") is not None:
            clauses.append("coalesce(c.face_count, 0) >= ?")
            params.append(int(f["min_faces"]))
        if f.get("max_faces") is not None:
            clauses.append("coalesce(c.face_count, 0) <= ?")
            params.append(int(f["max_faces"]))

        h1, h2 = f.get("hour_from"), f.get("hour_to")
        if h1 is not None or h2 is not None:
            h1 = 0 if h1 is None else int(h1)
            h2 = 23 if h2 is None else int(h2)
            if not (0 <= h1 <= 23 and 0 <= h2 <= 23):
                raise FilterError("hour_from/hour_to는 0~23")
            if h1 <= h2:
                clauses.append("c.hour_capture BETWEEN ? AND ?")
            else:  # 자정을 넘는 구간(예: 20시~4시)
                clauses.append("(c.hour_capture >= ? OR c.hour_capture <= ?)")
            params += [h1, h2]

        if content:
            if mode == "any" and len(content) > 1:
                clauses.append("(" + " OR ".join(sql for sql, _ in content) + ")")
                for _, ps in content:
                    params += ps
            else:
                for sql, ps in content:
                    clauses.append(sql)
                    params += ps
        where = " AND ".join(clauses) if clauses else "1"
        return where, params, info

    # ---------- 도구 1: search_cards ----------
    def search_cards(self, **f) -> dict:
        sort = f.get("sort") or "date_desc"
        if sort not in SORTS:
            raise FilterError(f"sort는 {sorted(SORTS)} 중 하나")
        limit = max(1, min(int(f.get("limit") or 8), ROW_LIMIT_MAX))
        where, params, info = self._where(f)

        order = {
            "date_desc": "c.date_local DESC",
            "date_asc": "c.date_local ASC",
            "aesthetic": "c.aesthetic IS NULL, c.aesthetic DESC, c.date_local DESC",
            "faces": "coalesce(c.face_count, 0) DESC, c.date_local DESC",
            "text_len": f"{self._text_len_expr()} DESC, c.date_local DESC",
            "relevance": "c.date_local DESC",
        }[sort]
        uuids = [r[0] for r in self.conn.execute(
            f"SELECT c.uuid FROM cards c WHERE {where} ORDER BY {order}", params)]
        if sort == "relevance" and f.get("text_like") and uuids:
            uuids = self._order_by_fts(uuids, f["text_like"])

        rid = f"r{self._next_id}"
        self._next_id += 1
        clean = {k: v for k, v in f.items() if v not in (None, [], "")}
        self.results[rid] = {"uuids": uuids, "total": len(uuids), "filters": clean}

        out = {"result_id": rid, "total": len(uuids), "filters": clean}
        if info["resolved"]:
            out["resolved"] = info["resolved"]
        if info["unknown"]:
            out["unknown"] = info["unknown"]
        if uuids:
            out["breakdown"] = self._breakdown(where, params)
            prefer = _as_list(f.get("labels_all")) + _as_list(f.get("labels_any"))
            out["rows"] = [self._row_brief(u, prefer) for u in uuids[:limit]]
        return out

    def _text_len_expr(self) -> str:
        """사진 속 글자 수: 사진 앱 색인 글자와 우리 OCR 글자 중 큰 쪽."""
        if "ocr_text" in self.tables:
            return ("max(coalesce(c.text_len, 0), coalesce((SELECT length(o.text) FROM ocr_text o "
                    "WHERE o.uuid = c.uuid), 0))")
        return "coalesce(c.text_len, 0)"

    def _order_by_fts(self, uuids, text):
        terms = [t.strip() for t in str(text).split("|") if t.strip()]
        fts_q = " OR ".join('"' + t.replace('"', '""') + '"*' for t in terms)
        try:
            score = dict(self.conn.execute(
                "SELECT uuid, bm25(cards_fts) FROM cards_fts WHERE cards_fts MATCH ?", (fts_q,)))
        except sqlite3.OperationalError:
            return uuids
        pos = {u: i for i, u in enumerate(uuids)}
        return sorted(uuids, key=lambda u: (score.get(u, 0.0), pos[u]))

    def _breakdown(self, where, params) -> dict:
        sub = f"SELECT c.uuid FROM cards c WHERE {where}"
        b = {}
        # 결과 집합 전체의 실제 날짜 범위. 요약의 「날짜 범위」는 이 값으로 쓴다(by_month_top6의 가장 이른·늦은 달이 아니라).
        lo, hi = self.conn.execute(
            f"SELECT MIN(c.date_local), MAX(c.date_local) FROM cards c WHERE {where}", params).fetchone()
        b["date_min"] = (lo or "")[:10] or None
        b["date_max"] = (hi or "")[:10] or None
        # 건수 상위 6개월만(전체 월 목록이 아님) — 키 이름에 드러낸다.
        b["by_month_top6"] = dict(self.conn.execute(
            f"SELECT substr(c.date_local,1,7) m, COUNT(*) FROM cards c WHERE {where} "
            "GROUP BY m ORDER BY COUNT(*) DESC LIMIT 6", params).fetchall())
        b["by_place"] = {k or "(위치 없음)": n for k, n in self.conn.execute(
            f"SELECT coalesce(c.city, c.sub_admin, c.state, c.country) p, COUNT(*) FROM cards c "
            f"WHERE {where} GROUP BY p ORDER BY COUNT(*) DESC LIMIT 6", params)}
        b["top_labels"] = dict(self.conn.execute(
            f"SELECT tag, COUNT(*) FROM card_tags WHERE uuid IN ({sub}) AND kind IN "
            "('label','venue_type','document_type','activity') GROUP BY tag ORDER BY COUNT(*) DESC LIMIT 10",
            params).fetchall())
        b["times_of_day"] = dict(self.conn.execute(
            f"SELECT tag, COUNT(*) FROM card_tags WHERE uuid IN ({sub}) AND kind='time_of_day' "
            "GROUP BY tag ORDER BY COUNT(*) DESC", params).fetchall())
        r = self.conn.execute(
            f"SELECT SUM(coalesce(c.face_count,0)=0), SUM(coalesce(c.face_count,0) BETWEEN 1 AND 2), "
            f"SUM(coalesce(c.face_count,0) >= 3), SUM(c.is_video), SUM(c.screenshot) FROM cards c WHERE {where}",
            params).fetchone()
        b["faces"] = {"0": r[0] or 0, "1-2": r[1] or 0, "3+": r[2] or 0}
        b["videos"] = r[3] or 0
        if r[4]:
            b["screenshots"] = r[4]
        if "captions" in self.tables:
            rows = self.conn.execute(
                f"SELECT p.cuisine, p.doc_type, COUNT(*) FROM captions p WHERE p.uuid IN ({sub}) GROUP BY 1, 2",
                params).fetchall()
            captioned = sum(n for _, _, n in rows)
            if captioned:
                b["captioned"] = captioned
                cu, dt = Counter(), Counter()
                for c_, d_, n in rows:
                    cu[c_] += n
                    dt[d_] += n
                cu.pop("없음", None)
                dt.pop("없음", None)
                if cu:
                    b["cuisine"] = dict(cu.most_common())
                if dt:
                    b["doc_type"] = dict(dt.most_common())
        return b

    def card(self, uuid) -> sqlite3.Row:
        return self.conn.execute("SELECT * FROM cards WHERE uuid=?", (uuid,)).fetchone()

    def top_labels_for(self, row, n=3, prefer=()):
        """카드의 Apple 라벨 n개: 질문 조건에 걸린 라벨 먼저, 나머지는 보관함 전체에서 드문 순
        (확신도가 없어서 흔한 '야외'·'사람'보다 드문 라벨을 정보가 많은 것으로 본다)."""
        labels = json.loads(row["apple_labels"] or "[]")
        counts = self.tag_counts()
        pref = [t for t in prefer if t in labels]
        rest = sorted((t for t in labels if t not in pref), key=lambda t: (counts.get((t, "label"), 0), t))
        return (pref + rest)[:n]

    def _row_brief(self, uuid, prefer=()) -> dict:
        row = self.card(uuid)
        place = ", ".join(x for x in [row["city"] or row["sub_admin"], row["state"] or row["country"]] if x)
        d = {"id": uuid[:8], "date": row["date_local"][:16].replace("T", " ")}
        if place:
            d["place"] = place
        labels = self.top_labels_for(row, prefer=prefer)
        if labels:
            d["labels"] = labels
        vt = json.loads(row["venue_types"] or "[]")
        if vt:
            d["venue_type"] = vt[:2]
        if row["face_count"]:
            d["faces"] = row["face_count"]
        if row["is_video"]:
            d["video_s"] = row["duration_s"]
        if row["text_len"]:
            d["text_chars"] = row["text_len"]
        cap = self.caption_of(uuid)
        if cap:
            d["scene"] = cap["scene_ko"][:60]
            if cap["cuisine"] and cap["cuisine"] != "없음":
                d["cuisine"] = cap["cuisine"]
            if cap["doc_type"] and cap["doc_type"] != "없음":
                d["doc_type"] = cap["doc_type"]
        return d

    def caption_of(self, uuid):
        """그 카드의 캡션 요약 값(없으면 None). 글자 조각(text_snippet)은 로컬 표시용 — LLM 도구 결과엔 안 싣는다."""
        if "captions" not in self.tables:
            return None
        r = self.conn.execute("SELECT scene_ko, cuisine, doc_type, text_snippet FROM captions WHERE uuid=?",
                              (uuid,)).fetchone()
        return dict(r) if r else None

    # ---------- 도구 2: stats ----------
    def stats(self, group_by: str, **f) -> dict:
        where, params, info = self._where(f)
        sub = f"SELECT c.uuid FROM cards c WHERE {where}"
        if group_by in ("label", "venue_type", "time_of_day", "activity"):
            kind = {"label": "label"}.get(group_by, group_by)
            rows = self.conn.execute(
                f"SELECT tag, COUNT(*) FROM card_tags WHERE uuid IN ({sub}) AND kind=? "
                "GROUP BY tag ORDER BY COUNT(*) DESC LIMIT 40", params + [kind]).fetchall()
        else:
            expr = {
                "month": "substr(c.date_local,1,7)",
                "year": "c.year",
                "weekday": "c.weekday",
                "hour": "c.hour_capture",
                "state": "coalesce(c.state, c.country, '(위치 없음)')",
                "city": "coalesce(c.city, c.sub_admin, '(위치 없음)') || ' / ' || coalesce(c.state, c.country, '')",
            }.get(group_by)
            if expr is None:
                raise FilterError("group_by는 month·year·weekday·hour·state·city·label·venue_type·time_of_day·activity 중 하나")
            order = "k" if group_by in ("month", "year", "weekday", "hour") else "COUNT(*) DESC"
            rows = self.conn.execute(
                f"SELECT {expr} k, COUNT(*) FROM cards c WHERE {where} GROUP BY k ORDER BY {order} LIMIT 60",
                params).fetchall()
            if group_by == "weekday":
                rows = [(WEEKDAY_KO[k], n) for k, n in rows]
        total = self.conn.execute(f"SELECT COUNT(*) FROM cards c WHERE {where}", params).fetchone()[0]
        out = {"group_by": group_by, "total": total, "groups": [[str(k), n] for k, n in rows]}
        if info["resolved"]:
            out["resolved"] = info["resolved"]
        if info["unknown"]:
            out["unknown"] = info["unknown"]
        return out

    # ---------- 도구 3: list_vocab ----------
    def list_vocab(self, kind: str, prefix: str = None, contains: str = None, limit: int = 80,
                   min_conf: float = None) -> dict:
        limit = max(1, min(int(limit or 80), VOCAB_LIMIT_MAX))

        def keep(name):
            if prefix and not name.startswith(prefix):
                return False
            if contains and contains not in name:
                return False
            return True

        if kind == "labels":
            items = [(t, k, n) for (t, k), n in self.tag_counts().items() if keep(t)]
            items.sort(key=lambda x: -x[2])
            grouped = {}
            for t, k, n in items[:limit]:
                grouped.setdefault(k, []).append(f"{t} {n}")
            return {"kind": kind, "matched": len(items), "shown": min(len(items), limit), "entries": grouped}
        if kind == "states":
            rows = self.conn.execute(
                "SELECT state, max(country), COUNT(*) FROM cards WHERE state IS NOT NULL GROUP BY state "
                "ORDER BY COUNT(*) DESC").fetchall()
            items = [f"{s} ({c}) {n}" for s, c, n in rows if keep(s)]
        elif kind == "cities":
            rows = self.conn.execute(
                "SELECT v, max(st), COUNT(*) FROM ("
                " SELECT city v, coalesce(state, country) st FROM cards WHERE city IS NOT NULL"
                " UNION ALL SELECT sub_admin, coalesce(state, country) FROM cards WHERE sub_admin IS NOT NULL"
                " UNION ALL SELECT district, coalesce(city, state) FROM cards WHERE district IS NOT NULL"
                ") GROUP BY v ORDER BY COUNT(*) DESC").fetchall()
            items = [f"{v} ({s}) {n}" for v, s, n in rows if keep(v)]
        elif kind == "places":
            cnt = Counter()
            for pn, nb, vn in self.conn.execute("SELECT place_names, neighborhoods, venues FROM cards"):
                for col in (pn, nb, vn):
                    cnt.update(json.loads(col or "[]"))
            items = [f"{v} {n}" for v, n in cnt.most_common() if keep(v)]
        elif kind in ("cuisine", "doc_types", "dishes"):
            if "captions" not in self.tables:
                return {"kind": kind, "matched": 0, "shown": 0, "entries": [],
                        "note": "captions 표 없음(index/enrich_caption.py 실행 필요)"}
            if kind == "dishes":
                cnt = Counter()
                # 현재 카드와 조인 — 카드가 지워진 뒤 남은 고아 캡션 행은 어휘에 세지 않는다.
                for (j,) in self.conn.execute("SELECT p.json FROM captions p JOIN cards c ON c.uuid = p.uuid"):
                    cnt.update(json.loads(j)["food"]["dishes_ko"])
                items = [f"{v} {n}" for v, n in cnt.most_common() if keep(v)]
            else:
                col = "cuisine" if kind == "cuisine" else "doc_type"
                items = [f"{v} {n}" for v, n in self.conn.execute(
                    f"SELECT p.{col}, COUNT(*) FROM captions p JOIN cards c ON c.uuid = p.uuid "
                    "GROUP BY 1 ORDER BY 2 DESC") if keep(v)]
        elif kind == "vision_labels":
            mc = VISION_MIN_CONF if min_conf is None else float(min_conf)
            if "vision_labels" not in self.tables:
                return {"kind": kind, "matched": 0, "shown": 0, "entries": [],
                        "note": "vision_labels 표 없음(index/enrich_vision.py 실행 필요)"}
            rows = self.conn.execute(
                "SELECT v.label_ko, v.label_en, COUNT(*) FROM vision_labels v JOIN cards c ON c.uuid = v.uuid "
                "WHERE v.confidence >= ? GROUP BY v.label_en ORDER BY 3 DESC", (mc,)).fetchall()
            items = [f"{ko or en} ({en}) {n}" for ko, en, n in rows if keep(ko or "") or keep(en)]
            return {"kind": kind, "min_conf": mc, "matched": len(items), "shown": min(len(items), limit),
                    "entries": items[:limit]}
        else:
            raise FilterError("kind는 labels·states·cities·places·cuisine·dishes·doc_types·vision_labels 중 하나")
        return {"kind": kind, "matched": len(items), "shown": min(len(items), limit), "entries": items[:limit]}

    # ---------- 시스템 프롬프트용 요약 ----------
    def summary_text(self) -> str:
        q = lambda sql, *p: self.conn.execute(sql, p).fetchall()
        total, videos, hidden, gps = q(
            "SELECT COUNT(*), SUM(is_video), SUM(hidden), SUM(lat IS NOT NULL) FROM cards")[0]
        videos, hidden, gps = videos or 0, hidden or 0, gps or 0   # 카드 0장이면 SUM이 NULL이다
        if not total:
            # 빈 색인에서도 요약을 만든다(ask.py가 이 문장을 보고 build_index.py를 안내한다).
            return ("- 색인이 비어 있다(카드 0장) — index/build_index.py 를 먼저 실행해야 한다. "
                    "지금은 날짜·장소·라벨·캡션 요약이 없고 어떤 조건으로도 0건이다.")
        dmin, dmax = q("SELECT min(substr(date_local,1,10)), max(substr(date_local,1,10)) FROM cards")[0]
        years = q("SELECT year, COUNT(*) FROM cards GROUP BY year ORDER BY year DESC")
        recent = [f"{y} {n:,}" for y, n in years if y >= 2024]
        older = sum(n for y, n in years if y < 2024)
        states = q("SELECT coalesce(state, '(state 없음: ' || coalesce(country,'위치 없음') || ')'), COUNT(*) "
                   "FROM cards GROUP BY 1 ORDER BY 2 DESC LIMIT 25")
        cities = q("SELECT coalesce(city, sub_admin), coalesce(state, country), COUNT(*) FROM cards "
                   "WHERE coalesce(city, sub_admin) IS NOT NULL GROUP BY 1 ORDER BY 3 DESC LIMIT 20")
        kinds = q("SELECT kind, COUNT(DISTINCT tag) FROM card_tags GROUP BY kind ORDER BY 2 DESC")
        top_labels = q("SELECT tag, COUNT(*) FROM card_tags WHERE kind='label' GROUP BY tag "
                       "ORDER BY 2 DESC LIMIT 50")

        def kind_values(kind):
            return " · ".join(f"{t} {n}" for t, n in q(
                "SELECT tag, COUNT(*) FROM card_tags WHERE kind=? GROUP BY tag ORDER BY 2 DESC", kind))

        lines = [
            f"- 전체 {total:,}건(사진 {total - videos:,} · 영상 {videos:,}), 숨김 {hidden}건은 기본 검색에서 뺌. "
            f"날짜 {dmin} ~ {dmax}. 연도별: {' · '.join(recent)} · 2024년 이전 {older}건. "
            f"위치(GPS) 있는 비율 {gps / total:.0%}.",
            "- state(1차 행정구역) 상위: " + " · ".join(f"{s} {n:,}" for s, n in states),
            "- 도시 상위 20: " + " · ".join(f"{c}({s}) {n:,}" for c, s, n in cities),
            "- 태그 종류별 낱말 수: " + " · ".join(f"{k} {n}" for k, n in kinds),
            "- Apple 장면 라벨 상위 50(한국어 그대로): " + " · ".join(f"{t} {n:,}" for t, n in top_labels),
            "- 장소 유형(venue_type, 가게 정보에서 옴) 전체: " + kind_values("venue_type"),
            "- 장소 유형이 붙은 사진 비율(최근 6개월, 월별): " + " · ".join(
                f"{m} {v / n:.0%}({v}/{n})" for m, n, v in q(
                    "SELECT substr(c.date_local,1,7) m, COUNT(*), SUM(EXISTS(SELECT 1 FROM card_tags t "
                    "WHERE t.uuid=c.uuid AND t.kind='venue_type')) FROM cards c GROUP BY m ORDER BY m DESC LIMIT 6")),
            "- 시간대(time_of_day): " + kind_values("time_of_day"),
            "- 활동(activity): " + kind_values("activity"),
            "- 문서 유형(document_type): " + kind_values("document_type")
            + " / 신분증류(id_document_type): " + kind_values("id_document_type"),
            "- 미디어 유형(media_type): " + kind_values("media_type"),
            "- 계절: " + kind_values("season") + " / 공휴일: " + kind_values("holiday"),
            "- 기타 플래그(flag): " + kind_values("flag") + " / 앱 출처(source_app): " + kind_values("source_app"),
        ]
        lines += self._enrich_summary(q, total, videos)
        return "\n".join(lines)

    def _enrich_summary(self, q, total, videos) -> list:
        out = []
        if not self.has("captions"):
            # 캡션이 0장이어도 에이전트는 돌아야 한다 — 없다는 사실을 요약에 적어 모델이 캡션 필터를 고르지 않게.
            out.append("- 캡션: 없음(캡션 단계를 돌리지 않았거나 이 인덱스에 캡션이 0장) — "
                       "cuisine·doc_type·caption_like 필터는 0건이 되고 rows에 scene이 없다. "
                       "사진 내용은 장면 라벨·Vision 라벨·사진 속 글자로 찾는다.")
        if self.has("captions"):
            # 집계는 모두 현재 cards와 조인한다 — 카드가 사라진 뒤 남은 고아 캡션 행을 세지 않는다.
            (n_cap,), = q("SELECT COUNT(*) FROM captions p JOIN cards c ON c.uuid = p.uuid")
            cu = q("SELECT p.cuisine, COUNT(*) FROM captions p JOIN cards c ON c.uuid = p.uuid "
                   "GROUP BY 1 ORDER BY 2 DESC")
            dt = q("SELECT p.doc_type, COUNT(*) FROM captions p JOIN cards c ON c.uuid = p.uuid "
                   "WHERE p.doc_type != '없음' GROUP BY 1 ORDER BY 2 DESC")
            out.append(f"- 캡션(클라우드 VLM이 미리보기를 보고 쓴 장면·물체·음식·키워드): 사진 {n_cap:,}/{total - videos:,}장"
                       "(영상 없음). 요리 국적(cuisine): " + " · ".join(f"{k} {n:,}" for k, n in cu)
                       + " / 문서 종류(doc_type): " + (" · ".join(f"{k} {n}" for k, n in dt) or "없음"))
        if self.has("vision_labels"):
            (n_v,), = q("SELECT COUNT(DISTINCT v.uuid) FROM vision_labels v JOIN cards c ON c.uuid = v.uuid")
            out.append(f"- Vision 라벨(공개 분류기, 확신도 {VISION_FLOOR} 이상 저장): {n_v:,}장. "
                       f"vision_labels_any 기본 확신도 {VISION_MIN_CONF}.")
        if self.has("label_scores"):
            (n_s, n_names), = q("SELECT COUNT(DISTINCT s.uuid), COUNT(DISTINCT s.label_ko) FROM label_scores s "
                                "JOIN cards c ON c.uuid = s.uuid")
            out.append(f"- 사진 앱 장면 확신도(label_scores, {VISION_FLOOR} 이상): {n_s:,}장 · 이름 {n_names:,}종"
                       " — apple_label_min_conf로 문턱 아래 라벨까지 넓힐 때.")
        if self.has("ocr_text"):
            (n_o, n_lines), = q("SELECT COUNT(*), SUM(o.line_count > 0) FROM ocr_text o JOIN cards c ON c.uuid = o.uuid")
            out.append(f"- 우리 OCR(글자 있을 법한 사진만): {n_o:,}장 중 글자 나온 {n_lines or 0:,}장 — text_like가 함께 찾는다.")
        return out

    def close(self):
        self.conn.close()
