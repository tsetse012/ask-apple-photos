"""검색 도구 단위 테스트 — 가상의 카드로만(실제 보관함 내용 없음). 실행: .venv/bin/python -m unittest discover -s tests"""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photo_ask.cards_query import CardIndex, FilterError, place_core  # noqa: E402
from photo_ask.cards_schema import CARD_COLUMNS, open_index_rw  # noqa: E402

COLS = [n for n, _ in CARD_COLUMNS]


def make_card(uuid, date_local, hour_capture, state=None, city=None, country="가상국", labels=(),
              tags=(), faces=0, video=0, hidden=0, detected=None, venue_types=()):
    c = {k: None for k in COLS}
    c.update(uuid=uuid, date_local=date_local, date_capture=date_local,
             year=int(date_local[:4]), month=int(date_local[5:7]), day=int(date_local[8:10]),
             weekday=0, hour=hour_capture, hour_capture=hour_capture,
             country=country, state=state, city=city,
             apple_labels=json.dumps(list(labels), ensure_ascii=False),
             venue_types=json.dumps(list(venue_types), ensure_ascii=False),
             place_names="[]", neighborhoods="[]", venues="[]",
             face_count=faces, is_video=video, hidden=hidden, screenshot=0, favorite=0,
             detected_text=detected, text_len=len(detected or ""),
             search_text=" ".join(list(labels) + [t for t, _ in tags] + list(venue_types)
                                  + [x for x in (state, city, country) if x]).lower())
    all_tags = [(t, "label") for t in labels] + [(t, "venue_type") for t in venue_types] + list(tags)
    return c, all_tags


class CardIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / "cards.sqlite"
        conn = open_index_rw(cls.db)
        cards = [
            make_card("A0000001-0000", "2026-09-20T13:00:00+09:00", 13, "가상특별자치도", "테스트시",
                      labels=["음식점", "커피잔", "탁자"], tags=[("오후", "time_of_day")], venue_types=["카페"], faces=2),
            make_card("A0000002-0000", "2026-09-21T23:30:00+09:00", 23, "가상특별자치도", "테스트시",
                      labels=["바다", "하늘"], tags=[("밤", "time_of_day")]),
            make_card("A0000003-0000", "2026-08-01T02:00:00+09:00", 2, "샘플도", "샘플군",
                      labels=["바다"], tags=[("밤", "time_of_day")], video=1),
            make_card("A0000004-0000", "2026-07-10T10:00:00+09:00", 10, None, None,
                      labels=["문서", "텍스트"], detected="영수증 합계 12000원", faces=None),
            make_card("A0000005-0000", "2026-09-22T12:00:00+09:00", 12, "가상특별자치도", "테스트시",
                      labels=["음식점"], hidden=1),
            make_card("A0000006-0000", "2025-01-01T09:00:00+09:00", 9, "샘플도", "샘플군",
                      labels=["사람", "결혼식"], faces=7),
        ]
        for card, tags in cards:
            conn.execute(f"INSERT INTO cards ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})",
                         [card[c] for c in COLS])
            conn.executemany("INSERT INTO card_tags VALUES (?,?,?)", [(card["uuid"], t, k) for t, k in tags])
        conn.execute("INSERT INTO cards_fts(uuid, body) SELECT uuid, search_text FROM cards")
        conn.commit()
        conn.close()
        cls.ix = CardIndex(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.tmp.cleanup()

    def test_place_core(self):
        self.assertEqual(place_core("강원도"), place_core("강원특별자치도"))
        self.assertEqual(place_core("부산광역시"), place_core("부산"))
        self.assertEqual(place_core("제주시"), "제주")
        self.assertEqual(place_core("서울특별시"), "서울")

    def test_state_alias_resolves(self):
        r = self.ix.search_cards(states=["가상도"])
        self.assertEqual(r["resolved"]["states"], ["가상특별자치도"])
        self.assertEqual(r["total"], 2)  # 숨김 1건 제외

    def test_hidden_default_excluded(self):
        self.assertEqual(self.ix.search_cards(labels_any=["음식점"])["total"], 1)
        self.assertEqual(self.ix.search_cards(labels_any=["음식점"], include_hidden=True)["total"], 2)

    def test_unknown_label_is_zero_not_ignored(self):
        r = self.ix.search_cards(labels_any=["커피"])
        self.assertEqual(r["total"], 0)
        self.assertIn("커피잔", r["unknown"]["labels"]["커피"])

    def test_unknown_state_is_zero(self):
        r = self.ix.search_cards(states=["없는곳"])
        self.assertEqual(r["total"], 0)
        self.assertEqual(r["unknown"]["states"], ["없는곳"])

    def test_labels_any_all_exclude(self):
        self.assertEqual(self.ix.search_cards(labels_any=["바다", "커피잔"])["total"], 3)
        self.assertEqual(self.ix.search_cards(labels_all=["바다", "하늘"])["total"], 1)
        self.assertEqual(self.ix.search_cards(labels_any=["바다"], exclude_labels=["하늘"])["total"], 1)
        self.assertEqual(self.ix.search_cards(labels_all=["바다", "없는라벨"])["total"], 0)

    def test_date_window_inclusive(self):
        r = self.ix.search_cards(date_from="2026-09-20", date_to="2026-09-21")
        self.assertEqual(r["total"], 2)
        with self.assertRaises(FilterError):
            self.ix.search_cards(date_from="2026/09/20")

    def test_hour_wraps_midnight(self):
        r = self.ix.search_cards(hour_from=22, hour_to=3)
        self.assertEqual(sorted(x["id"] for x in r["rows"]), ["A0000002", "A0000003"])

    def test_min_faces_with_null(self):
        r = self.ix.search_cards(min_faces=3)
        self.assertEqual([x["id"] for x in r["rows"]], ["A0000006"])
        self.assertEqual(self.ix.search_cards(max_faces=0)["total"], 3)  # NULL 얼굴 수는 0으로 본다

    def test_text_like_matches_detected_text_without_leaking_it(self):
        r = self.ix.search_cards(text_like="영수증")
        self.assertEqual(r["total"], 1)
        self.assertNotIn("영수증", json.dumps(r["rows"], ensure_ascii=False))
        self.assertEqual(r["rows"][0]["text_chars"], len("영수증 합계 12000원"))

    def test_text_like_alternatives(self):
        self.assertEqual(self.ix.search_cards(text_like="결혼식|카페")["total"], 2)

    def test_city_matches_sub_admin_and_city(self):
        r = self.ix.search_cards(cities=["샘플"])
        self.assertEqual(r["total"], 2)

    def test_video_filter(self):
        self.assertEqual(self.ix.search_cards(is_video=True)["total"], 1)

    def test_stats(self):
        s = self.ix.stats("label", labels_any=["바다"])
        self.assertEqual(dict(s["groups"])["바다"], 2)
        s = self.ix.stats("month")
        self.assertEqual(s["total"], 5)
        with self.assertRaises(FilterError):
            self.ix.stats("color")

    def test_breakdown_reports_true_date_range_beyond_top6_months(self):
        """by_month_top6 는 상위 6개월일 뿐이라 날짜 범위로 못 쓴다 → date_min·date_max 가 따로 정확해야 한다."""
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "cards.sqlite"
            conn = open_index_rw(db)
            # 8개월(01~08)에 걸친 가상 카드 — 03월만 2장이라 03이 상위, 나머지 월은 1장씩(동률)이라 6개월에서 잘린다
            days = ["2026-01-02", "2026-02-03", "2026-03-04", "2026-03-05", "2026-04-06",
                    "2026-05-07", "2026-06-08", "2026-07-09", "2026-08-10"]
            for i, d in enumerate(days):
                card, tags = make_card(f"B000000{i}-0000", f"{d}T10:00:00+09:00", 10, labels=["바다"])
                conn.execute(f"INSERT INTO cards ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})",
                             [card[c] for c in COLS])
                conn.executemany("INSERT INTO card_tags VALUES (?,?,?)", [(card["uuid"], t, k) for t, k in tags])
            conn.commit()
            conn.close()
            ix = CardIndex(db)
            try:
                b = ix.search_cards(labels_any=["바다"])["breakdown"]
            finally:
                ix.close()
        self.assertEqual(b["date_min"], "2026-01-02")
        self.assertEqual(b["date_max"], "2026-08-10")
        self.assertEqual(len(b["by_month_top6"]), 6)          # 8개월 중 6개만
        self.assertNotIn("by_month", b)
        self.assertEqual(b["by_month_top6"]["2026-03"], 2)

    def test_list_vocab(self):
        v = self.ix.list_vocab("labels", contains="음식")
        self.assertIn("음식점 2", v["entries"]["label"])
        v = self.ix.list_vocab("states", prefix="가상")
        self.assertEqual(v["matched"], 1)

    def test_relevance_sort_uses_fts(self):
        by_date = self.ix.search_cards(text_like="바다", sort="date_desc")
        by_rel = self.ix.search_cards(text_like="바다", sort="relevance")
        self.assertEqual([x["id"] for x in by_date["rows"]], ["A0000002", "A0000003"])
        # bm25: 더 짧은 본문(A3)이 앞으로 — 날짜순과 순서가 달라야 FTS가 실제로 쓰인 것
        self.assertEqual([x["id"] for x in by_rel["rows"]], ["A0000003", "A0000002"])

    def test_result_ids_are_kept(self):
        r = self.ix.search_cards(labels_any=["바다"])
        self.assertIn(r["result_id"], self.ix.results)
        self.assertEqual(self.ix.results[r["result_id"]]["total"], r["total"])


class EmptyIndexSummaryTest(unittest.TestCase):
    """카드 0장 색인: SUM이 NULL이고 0으로 나누게 되던 자리 — 예외 없이 빈 보관함 요약을 돌려준다."""

    def test_empty_summary_is_returned_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "cards.sqlite"
            conn = open_index_rw(db)
            conn.close()
            ix = CardIndex(db)
            try:
                text = ix.summary_text()
            finally:
                ix.close()
        self.assertIn("색인이 비어 있다", text)
        self.assertIn("build_index.py", text)


class Fts5KoreanTokenizerTest(unittest.TestCase):
    """FTS5 토크나이저 차이: trigram은 2음절 한국어 낱말을 못 찾고 unicode61은 찾는다."""

    def test_trigram_misses_two_syllable_words(self):
        c = sqlite3.connect(":memory:")
        c.execute("CREATE VIRTUAL TABLE tri USING fts5(body, tokenize='trigram')")
        c.execute("CREATE VIRTUAL TABLE uni USING fts5(body, tokenize='unicode61')")
        for t in ("tri", "uni"):
            c.execute(f"INSERT INTO {t}(body) VALUES ('바다 하늘 음식점')")
        self.assertEqual(c.execute("SELECT COUNT(*) FROM tri WHERE tri MATCH '\"바다\"'").fetchone()[0], 0)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM uni WHERE uni MATCH '\"바다\"'").fetchone()[0], 1)
        self.assertEqual(c.execute("SELECT COUNT(*) FROM tri WHERE tri MATCH '\"음식점\"'").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
