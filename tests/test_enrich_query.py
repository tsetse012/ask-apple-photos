"""2단계 보강 필터(캡션·Vision·사진 앱 확신도·OCR) 단위 테스트 — 가상의 카드로만. 과금 0."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from photo_ask import caption  # noqa: E402
from photo_ask.cards_query import CardIndex, FilterError  # noqa: E402
from photo_ask.cards_schema import CARD_COLUMNS, ENRICH_TABLES, open_index_rw  # noqa: E402
from test_cards_query import make_card  # noqa: E402

COLS = [n for n, _ in CARD_COLUMNS]


def cap_json(scene, cuisine="없음", dishes=(), doc="없음", keywords=(), objects=(), snippet="", people=0):
    present = cuisine != "없음"
    return {"scene_ko": scene, "objects_ko": list(objects), "food": {"present": present, "cuisine": cuisine,
                                                                     "dishes_ko": list(dishes)},
            "people_count": people, "indoor_outdoor": "실내", "time_of_day_guess": "낮", "text_visible": bool(snippet),
            "text_snippet": snippet, "doc_type": doc, "keywords_ko": list(keywords)}


def insert_cards(conn, cards):
    for card, tags in cards:
        conn.execute(f"INSERT INTO cards ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})",
                     [card[c] for c in COLS])
        conn.executemany("INSERT INTO card_tags VALUES (?,?,?)", [(card["uuid"], t, k) for t, k in tags])


def insert_caption(conn, uuid, cap):
    cap = caption.parse_caption(json.dumps(cap, ensure_ascii=False))
    conn.execute("INSERT INTO captions(uuid, model, json, created_at, scene_ko, food_present, cuisine, doc_type, "
                 "people_count, text_snippet, search_blob) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 (uuid, "test-model", json.dumps(cap, ensure_ascii=False), "2026-10-08T00:00:00+09:00",
                  cap["scene_ko"], int(cap["food"]["present"]), cap["food"]["cuisine"], cap["doc_type"],
                  cap["people_count"], cap["text_snippet"], caption.search_blob(cap)))


class EnrichQueryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / "cards.sqlite"
        conn = open_index_rw(cls.db)
        insert_cards(conn, [
            make_card("E0000001-0000", "2026-09-20T12:00:00+09:00", 12, "가상도", "테스트시", labels=["음식"]),
            make_card("E0000002-0000", "2026-09-19T19:00:00+09:00", 19, "가상도", "테스트시", labels=["음식"]),
            make_card("E0000003-0000", "2026-09-18T15:00:00+09:00", 15, "가상도", "테스트시", labels=["문서"]),
            make_card("E0000004-0000", "2026-09-17T10:00:00+09:00", 10, "샘플도", "샘플군", labels=["야외"], video=1),
            make_card("E0000005-0000", "2026-09-16T16:00:00+09:00", 16, "샘플도", "샘플군", labels=["음식"], hidden=1),
            make_card("E0000006-0000", "2026-09-15T09:00:00+09:00", 9, "샘플도", "샘플군", labels=["실내"]),
        ])
        insert_caption(conn, "E0000001-0000", cap_json("가상 식탁 위 면 요리", "양식", ["파스타"], keywords=["점심"]))
        insert_caption(conn, "E0000002-0000", cap_json("가상 그릇에 담긴 밥", "한식", ["비빔밥"], keywords=["저녁"]))
        insert_caption(conn, "E0000003-0000", cap_json("가상 종이 한 장", doc="영수증", snippet="합계 12000"))
        insert_caption(conn, "E0000005-0000", cap_json("가상 접시 위 디저트", "카페·디저트", ["케이크"]))
        insert_caption(conn, "E0000006-0000", cap_json("가상 방 안의 동물", objects=["고양이"]))
        conn.executemany("INSERT INTO vision_labels VALUES (?,?,?,?)", [
            ("E0000001-0000", "food", "음식", 0.9), ("E0000001-0000", "pasta", "파스타", 0.6),
            ("E0000002-0000", "food", "음식", 0.8),
            ("E0000003-0000", "receipt", "영수증", 0.7), ("E0000003-0000", "document", "문서", 0.04),
            ("E0000004-0000", "dog", "개", 0.03),
            ("E0000006-0000", "cat", "고양이", 0.5),
            ("E0000006-0000", "road", "도로", 0.3), ("E0000002-0000", "road_other", "도로", 0.3),
        ])
        conn.executemany("INSERT INTO label_scores VALUES (?,?,?)", [
            ("E0000001-0000", "파스타", 0.3), ("E0000004-0000", "개", 0.15), ("E0000006-0000", "고양이", 0.05),
        ])
        conn.execute("INSERT INTO ocr_text VALUES (?,?,?,?)", ("E0000003-0000", "가상상점 영수증\n합계 12,000원", 2, 0.5))
        conn.execute("INSERT INTO cards_fts(uuid, body) SELECT uuid, search_text FROM cards")
        conn.commit()
        conn.close()
        cls.ix = CardIndex(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.ix.close()
        cls.tmp.cleanup()

    def ids(self, r):
        return sorted(x["id"] for x in r.get("rows", []))

    def test_cuisine_filter_and_alias(self):
        self.assertEqual(self.ids(self.ix.search_cards(cuisine=["양식"])), ["E0000001"])
        self.assertEqual(self.ids(self.ix.search_cards(cuisine="서양식")), ["E0000001"])
        self.assertEqual(self.ix.search_cards(cuisine="양식|한식")["total"], 2)
        # 숨김은 기본 제외, include_hidden이면 포함
        self.assertEqual(self.ix.search_cards(cuisine=["카페·디저트"])["total"], 0)
        self.assertEqual(self.ix.search_cards(cuisine=["카페"], include_hidden=True)["total"], 1)

    def test_unknown_cuisine_is_zero_with_valid_values(self):
        r = self.ix.search_cards(cuisine=["브런치"])
        self.assertEqual(r["total"], 0)
        self.assertEqual(r["unknown"]["cuisine"]["목록 밖"], ["브런치"])
        self.assertIn("양식", r["unknown"]["cuisine"]["쓸 수 있는 값"])

    def test_caption_like_scene_dishes_keywords(self):
        self.assertEqual(self.ix.search_cards(caption_like="파스타|비빔밥")["total"], 2)
        self.assertEqual(self.ids(self.ix.search_cards(caption_like="점심")), ["E0000001"])
        self.assertEqual(self.ids(self.ix.search_cards(caption_like="종이")), ["E0000003"])
        self.assertEqual(self.ids(self.ix.search_cards(caption_like="고양이")), ["E0000006"])  # objects_ko

    def test_doc_type(self):
        self.assertEqual(self.ids(self.ix.search_cards(doc_type=["영수증"])), ["E0000003"])
        self.assertEqual(self.ix.search_cards(doc_type="명함")["total"], 0)

    def test_vision_labels_threshold(self):
        self.assertEqual(self.ids(self.ix.search_cards(vision_labels_any=["고양이"])), ["E0000006"])
        self.assertEqual(self.ix.search_cards(vision_labels_any=["개"], include_hidden=True)["total"], 0)  # 0.03 < 0.05
        r = self.ix.search_cards(vision_labels_any=["개"], vision_min_conf=0.02)
        self.assertEqual(self.ids(r), ["E0000004"])
        self.assertEqual(self.ids(self.ix.search_cards(vision_labels_any=["cat"])), ["E0000006"])  # 영어 id도
        with self.assertRaises(FilterError):
            self.ix.search_cards(vision_labels_any=["개"], vision_min_conf=0.01)

    def test_vision_korean_name_shared_by_two_ids(self):
        r = self.ix.search_cards(vision_labels_any=["도로"])
        self.assertEqual(self.ids(r), ["E0000002", "E0000006"])
        self.assertEqual(sorted(r["resolved"]["vision_labels"]), ["road", "road_other"])

    def test_unknown_vision_label(self):
        r = self.ix.search_cards(vision_labels_any=["용"])
        self.assertEqual(r["total"], 0)
        self.assertIn("용", r["unknown"]["vision_labels"])

    def test_apple_label_min_conf_widens_to_label_scores(self):
        r = self.ix.search_cards(labels_any=["파스타"])
        self.assertEqual(r["total"], 0)                         # 사진 앱 검색 라벨(card_tags)엔 없다
        self.assertIn("파스타", r["unknown"]["labels"])
        r = self.ix.search_cards(labels_any=["파스타"], apple_label_min_conf=0.1)
        self.assertEqual(self.ids(r), ["E0000001"])              # 확신도 0.3 ≥ 0.1
        self.assertNotIn("unknown", r)
        self.assertEqual(self.ix.search_cards(labels_any=["파스타"], apple_label_min_conf=0.5)["total"], 0)
        # 검색 라벨과 확신도 표를 함께 본다(OR): 음식(검색 라벨) 또는 파스타(확신도)
        r = self.ix.search_cards(labels_any=["음식", "파스타"], apple_label_min_conf=0.1)
        self.assertEqual(self.ids(r), ["E0000001", "E0000002"])
        r = self.ix.search_cards(labels_all=["음식", "파스타"], apple_label_min_conf=0.1)
        self.assertEqual(self.ids(r), ["E0000001"])
        with self.assertRaises(FilterError):
            self.ix.search_cards(labels_any=["개"], apple_label_min_conf=0.01)

    def test_content_match_any_unions_labels_caption_and_vision(self):
        # 기본(all): 라벨 「음식」 AND 캡션 「파스타」 → E1만
        self.assertEqual(self.ids(self.ix.search_cards(labels_any=["음식"], caption_like="파스타")), ["E0000001"])
        # any: 라벨 「문서」 OR 캡션 「파스타」 → E3(라벨) + E1(캡션)
        r = self.ix.search_cards(labels_any=["문서"], caption_like="파스타", content_match="any")
        self.assertEqual(self.ids(r), ["E0000001", "E0000003"])
        # any에 Vision까지: + 고양이(E6)
        r = self.ix.search_cards(labels_any=["문서"], caption_like="파스타", vision_labels_any=["고양이"],
                                 content_match="any")
        self.assertEqual(self.ids(r), ["E0000001", "E0000003", "E0000006"])
        # 다른 조건은 그대로 AND: 날짜로 E1만 남긴다
        r = self.ix.search_cards(labels_any=["문서"], caption_like="파스타", content_match="any",
                                 date_from="2026-09-20", date_to="2026-09-20")
        self.assertEqual(self.ids(r), ["E0000001"])
        # 모르는 라벨은 any 묶음에서 빠질 뿐(캡션 쪽은 살아 있다)
        r = self.ix.search_cards(labels_any=["없는라벨"], caption_like="파스타", content_match="any")
        self.assertEqual(self.ids(r), ["E0000001"])
        with self.assertRaises(FilterError):
            self.ix.search_cards(caption_like="파스타", content_match="some")

    def test_text_like_reaches_ocr_and_snippet_without_leaking(self):
        r = self.ix.search_cards(text_like="가상상점")             # 우리 OCR 글자에만 있다
        self.assertEqual(self.ids(r), ["E0000003"])
        r2 = self.ix.search_cards(text_like="합계 12000")           # 캡션 글자 조각에만 있다
        self.assertEqual(self.ids(r2), ["E0000003"])
        rows = json.dumps([r["rows"], r2["rows"], r["breakdown"], r2["breakdown"]], ensure_ascii=False)
        for secret in ("가상상점", "12,000", "합계"):
            self.assertNotIn(secret, rows)               # 글자 내용은 매칭에만 쓰고 결과엔 안 싣는다

    def test_rows_and_breakdown_carry_caption(self):
        r = self.ix.search_cards(labels_any=["음식"])
        rows = {x["id"]: x for x in r["rows"]}
        self.assertEqual(rows["E0000001"]["scene"], "가상 식탁 위 면 요리")
        self.assertEqual(rows["E0000001"]["cuisine"], "양식")
        self.assertEqual(r["breakdown"]["cuisine"], {"양식": 1, "한식": 1})
        self.assertEqual(r["breakdown"]["captioned"], 2)
        r = self.ix.search_cards(doc_type=["영수증"])
        self.assertEqual(r["rows"][0]["doc_type"], "영수증")
        self.assertNotIn("text_snippet", json.dumps(r, ensure_ascii=False))

    def test_text_len_sort_uses_ocr(self):
        r = self.ix.search_cards(sort="text_len", limit=1)
        self.assertEqual(r["rows"][0]["id"], "E0000003")

    def test_list_vocab_new_kinds(self):
        self.assertIn("양식 1", self.ix.list_vocab("cuisine")["entries"])
        self.assertIn("파스타 1", self.ix.list_vocab("dishes")["entries"])
        self.assertIn("영수증 1", self.ix.list_vocab("doc_types")["entries"])
        v = self.ix.list_vocab("vision_labels")
        self.assertEqual(v["min_conf"], 0.05)
        self.assertIn("음식 (food) 2", v["entries"])
        self.assertFalse(any("(dog)" in e for e in v["entries"]))
        v = self.ix.list_vocab("vision_labels", min_conf=0.02, contains="개")
        self.assertEqual(v["entries"], ["개 (dog) 1"])
        with self.assertRaises(FilterError):
            self.ix.list_vocab("colors")

    def test_summary_mentions_enrichment(self):
        text = self.ix.summary_text()
        self.assertIn("캡션", text)
        self.assertIn("요리 국적(cuisine)", text)
        self.assertIn("Vision 라벨", text)
        self.assertIn("label_scores", text)


class V0IndexWithoutEnrichTablesTest(unittest.TestCase):
    """보강 표가 없는 인덱스에서도 새 필터가 터지지 않고 0건 + 이유를 돌려준다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Path(self.tmp.name) / "cards.sqlite"
        conn = open_index_rw(db)
        insert_cards(conn, [make_card("F0000001-0000", "2026-09-20T12:00:00+09:00", 12, "가상도", "테스트시",
                                      labels=["음식"])])
        for t in ENRICH_TABLES:
            conn.execute(f"DROP TABLE {t}")
        conn.commit()
        conn.close()
        self.ix = CardIndex(db)

    def tearDown(self):
        self.ix.close()
        self.tmp.cleanup()

    def test_filters_degrade_to_zero_with_reason(self):
        r = self.ix.search_cards(cuisine=["양식"])
        self.assertEqual(r["total"], 0)
        self.assertIn("captions 표 없음", r["unknown"]["cuisine"])
        r = self.ix.search_cards(caption_like="파스타")
        self.assertEqual(r["total"], 0)
        r = self.ix.search_cards(vision_labels_any=["개"])
        self.assertEqual(r["total"], 0)
        r = self.ix.search_cards(labels_any=["음식"], apple_label_min_conf=0.1)
        self.assertEqual(r["total"], 1)                          # 확신도 표가 없으면 검색 라벨만
        self.assertIn("apple_label_min_conf", r["unknown"])
        self.assertEqual(self.ix.search_cards(text_like="음식")["total"], 1)
        self.assertEqual(self.ix.list_vocab("cuisine")["entries"], [])
        self.assertIn("캡션: 없음", self.ix.summary_text())


class EmptyEnrichTablesTest(unittest.TestCase):
    """보강 표는 만들었지만 0행(캡션 0장) — 필터는 0건일 뿐 예외 없이 돌고, 요약이 그 사실을 알린다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Path(self.tmp.name) / "cards.sqlite"
        conn = open_index_rw(db)
        insert_cards(conn, [make_card("G0000001-0000", "2026-09-20T12:00:00+09:00", 12, "가상도", "테스트시",
                                      labels=["음식"])])
        conn.commit()
        conn.close()
        self.ix = CardIndex(db)

    def tearDown(self):
        self.ix.close()
        self.tmp.cleanup()

    def test_summary_says_no_captions_and_filters_do_not_raise(self):
        self.assertIn("캡션: 없음", self.ix.summary_text())
        self.assertEqual(self.ix.search_cards(cuisine=["양식"])["total"], 0)
        self.assertEqual(self.ix.search_cards(caption_like="파스타")["total"], 0)
        self.assertEqual(self.ix.search_cards(doc_type=["영수증"])["total"], 0)
        self.assertEqual(self.ix.search_cards(labels_any=["음식"])["total"], 1)      # 캡션 없이도 라벨로 찾는다
        self.assertEqual(self.ix.list_vocab("dishes")["entries"], [])


class OrphanEnrichRowsTest(unittest.TestCase):
    """카드가 지워진 뒤 보강 표에 남은 고아 행은 어휘·요약 집계에 세지 않는다(현재 cards와 조인)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Path(self.tmp.name) / "cards.sqlite"
        conn = open_index_rw(db)
        insert_cards(conn, [make_card("H0000001-0000", "2026-09-20T12:00:00+09:00", 12, "가상도", "테스트시")])
        insert_caption(conn, "H0000001-0000", cap_json("가상 식탁", "양식", ["파스타"]))
        # 카드가 없는 uuid의 보강 행(고아)
        insert_caption(conn, "H0000ORPH-0000", cap_json("고아 캡션", "한식", ["비빔밥"], doc="영수증"))
        conn.executemany("INSERT INTO vision_labels VALUES (?,?,?,?)", [
            ("H0000ORPH-0000", "ghost", "유령", 0.9),
        ])
        conn.execute("INSERT INTO label_scores VALUES ('H0000ORPH-0000', '유령', 0.5)")
        conn.execute("INSERT INTO ocr_text VALUES ('H0000ORPH-0000', '고아 글자', 1, 0.9)")
        conn.commit()
        conn.close()
        self.ix = CardIndex(db)

    def tearDown(self):
        self.ix.close()
        self.tmp.cleanup()

    def test_vocab_skips_orphans(self):
        self.assertEqual(self.ix.list_vocab("cuisine")["entries"], ["양식 1"])
        self.assertEqual(self.ix.list_vocab("dishes")["entries"], ["파스타 1"])
        self.assertEqual(self.ix.list_vocab("doc_types")["entries"], ["없음 1"])   # 고아 영수증은 안 센다
        self.assertEqual(self.ix.list_vocab("vision_labels")["entries"], [])    # 고아 유령 라벨도

    def test_summary_skips_orphans(self):
        text = self.ix.summary_text()
        self.assertIn("사진 1/1장", text)
        self.assertIn("양식 1", text)
        self.assertNotIn("한식", text)
        self.assertNotIn("영수증", text)
        self.assertIn("0장", text)                                             # Vision·label_scores·OCR 모두 0장
        self.assertNotIn("유령", text)

    def test_label_name_lookups_skip_orphans(self):
        self.assertNotIn("유령", self.ix.score_names())
        self.assertNotIn("유령", self.ix.vision_names())
        self.assertNotIn("ghost", self.ix.vision_names())


if __name__ == "__main__":
    unittest.main()
