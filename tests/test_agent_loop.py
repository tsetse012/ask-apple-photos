"""질문 에이전트 루프를 가짜 클라이언트로 검사(과금 0). 실제 API 동작은 별도 측정 스크립트로 잰다."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from photo_ask import agent, config, llm_log  # noqa: E402
from photo_ask.cards_query import CardIndex  # noqa: E402
from photo_ask.cards_schema import CARD_COLUMNS, open_index_rw  # noqa: E402
from test_cards_query import make_card  # noqa: E402

COLS = [n for n, _ in CARD_COLUMNS]


class Block(SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return {k: v for k, v in vars(self).items() if v is not None or not exclude_none}


def text(t):
    return Block(type="text", text=t)


def tool(id_, name, inp):
    return Block(type="tool_use", id=id_, name=name, input=inp)


def response(content, stop):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=3000,
                            cache_creation_input_tokens=500, cache_creation=None)
    return SimpleNamespace(content=content, stop_reason=stop, usage=usage, model="claude-sonnet-5-5",
                           _request_id="req_test")


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        # SDK는 호출 시점에 직렬화한다 — 같은 리스트가 나중에 늘어나므로 그 시점 사본을 남긴다.
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return self.script.pop(0)


class AgentLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(self.tmp.name)
        self.saved = (config.LLM_LOG, config.TRANSCRIPT_DIR)
        config.LLM_LOG = tmpdir / "llm.jsonl"
        config.TRANSCRIPT_DIR = tmpdir / "transcripts"
        db = tmpdir / "cards.sqlite"
        conn = open_index_rw(db)
        for card, tags in [
            make_card("B0000001-0000", "2026-09-20T13:00:00+09:00", 13, "가상특별자치도", "테스트시",
                      labels=["음식점", "커피잔"], venue_types=["카페"]),
            make_card("B0000002-0000", "2026-05-01T13:00:00+09:00", 13, "가상특별자치도", "테스트시",
                      labels=["음식점"], venue_types=["카페"]),
        ]:
            conn.execute(f"INSERT INTO cards ({','.join(COLS)}) VALUES ({','.join('?' * len(COLS))})",
                         [card[c] for c in COLS])
            conn.executemany("INSERT INTO card_tags VALUES (?,?,?)", [(card["uuid"], t, k) for t, k in tags])
        conn.commit()
        conn.close()
        self.index = CardIndex(db)

    def tearDown(self):
        config.LLM_LOG, config.TRANSCRIPT_DIR = self.saved
        self.index.close()
        self.tmp.cleanup()

    def test_widen_then_answer(self):
        client = FakeClient([
            response([tool("t1", "search_cards", {"states": ["가상도"], "labels_any": ["카페"],
                                                   "date_from": "2026-09-25", "date_to": "2026-10-08"})], "tool_use"),
            response([tool("t2", "search_cards", {"states": ["가상도"], "labels_any": ["카페"],
                                                   "date_from": "2026-04-01", "date_to": "2026-10-08"})], "tool_use"),
            response([text("요약: 후보 2건\n이유: 날짜 창을 넓힘\nRESULT: r2")], "end_turn"),
        ])
        res = agent.Asker(self.index, client=client).ask("저번달 가상도 카페")
        self.assertEqual(res.stop, "answered")
        self.assertEqual(res.search_totals, [0, 2])
        self.assertEqual(res.widen_count, 1)
        self.assertEqual(res.result_id, "r2")
        self.assertEqual(res.total, 2)
        self.assertFalse(res.result_fallback)
        rows = llm_log.read_all()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["images_sent"], 0)
        self.assertAlmostEqual(sum(r["cost_usd"] for r in rows), res.cost_usd, places=6)
        # 비용 = (1000*2 + 200*10 + 3000*0.2 + 500*2.5) / 1e6 × 3회
        self.assertAlmostEqual(res.cost_usd, 3 * 5850 / 1e6, places=9)
        # 대화는 앞부분을 고치지 않고 덧붙이기만 한다(preserved thinking 조건)
        first, last = client.requests[0]["messages"], client.requests[-1]["messages"]
        self.assertEqual(len(first), 1)
        self.assertEqual(len(last), 5)
        tr = json.loads(Path(res.transcript_path).read_text())
        self.assertEqual(tr["result"]["widen_count"], 1)
        self.assertEqual(len(tr["messages"]), 6)

    def test_question_non_interactive_stops(self):
        client = FakeClient([
            response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
            response([text("요약: 후보 2건\n이유: 라벨\nRESULT: r1\nQUESTION: 언제쯤인가요? (9월 1 / 5월 1)")], "end_turn"),
        ])
        res = agent.Asker(self.index, client=client).ask("음식")
        self.assertEqual(res.stop, "asked")
        self.assertTrue(res.asked)
        self.assertEqual(res.total, 2)

    def test_question_interactive_continues(self):
        client = FakeClient([
            response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
            response([text("요약: 2건\n이유: 라벨\nRESULT: r1\nQUESTION: 언제쯤?")], "end_turn"),
            response([tool("t2", "search_cards", {"labels_any": ["음식점"], "date_from": "2026-09-01"})], "tool_use"),
            response([text("요약: 1건\n이유: 9월로 좁힘\nRESULT: r2")], "end_turn"),
        ])
        res = agent.Asker(self.index, client=client, interactive=True,
                          input_fn=lambda _p: "9월", print_fn=lambda *_a: None).ask("음식")
        self.assertEqual(res.stop, "answered")
        self.assertEqual(res.user_reply, "9월")
        self.assertEqual(res.total, 1)
        self.assertEqual(client.requests[2]["messages"][-1], {"role": "user", "content": "9월"})

    def test_bad_tool_args_return_error_to_model(self):
        client = FakeClient([
            response([tool("t1", "search_cards", {"label_any": ["음식점"]})], "tool_use"),
            response([text("요약: 없음\n이유: -\nRESULT: none")], "end_turn"),
        ])
        res = agent.Asker(self.index, client=client).ask("음식")
        tool_result = client.requests[1]["messages"][-1]["content"][0]
        self.assertTrue(tool_result["is_error"])
        self.assertIn("label_any", tool_result["content"])
        self.assertIsNone(res.result_id)

    def test_missing_result_line_falls_back_to_last_nonempty(self):
        client = FakeClient([
            response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
            response([tool("t2", "search_cards", {"labels_any": ["없는라벨"]})], "tool_use"),
            response([text("요약: 2건")], "end_turn"),
        ])
        res = agent.Asker(self.index, client=client).ask("음식")
        self.assertEqual(res.result_id, "r1")
        self.assertTrue(res.result_fallback)

    def test_nudge_when_many_results_and_no_question(self):
        saved = agent.ASK_THRESHOLD
        agent.ASK_THRESHOLD = 1
        try:
            client = FakeClient([
                response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
                response([text("요약: 2건\n이유: 라벨\nRESULT: r1")], "end_turn"),
                response([text("요약: 2건\n이유: 라벨\nRESULT: r1\nQUESTION: 언제쯤? (9월 1 / 5월 1)")], "end_turn"),
            ])
            res = agent.Asker(self.index, client=client).ask("음식")
        finally:
            agent.ASK_THRESHOLD = saved
        self.assertTrue(res.nudged)
        self.assertTrue(res.asked)
        self.assertEqual(res.stop, "asked")
        self.assertIn("자동 점검", client.requests[2]["messages"][-1]["content"])

    def test_no_nudge_for_ranking_sort(self):
        saved = agent.ASK_THRESHOLD
        agent.ASK_THRESHOLD = 1
        try:
            client = FakeClient([
                response([tool("t1", "search_cards", {"labels_any": ["음식점"], "sort": "aesthetic"})], "tool_use"),
                response([text("요약: 2건\n이유: 미학 점수 순\nRESULT: r1")], "end_turn"),
            ])
            res = agent.Asker(self.index, client=client).ask("제일 잘 나온 음식 사진")
        finally:
            agent.ASK_THRESHOLD = saved
        self.assertFalse(res.nudged)
        self.assertEqual(res.stop, "answered")

    def test_repeated_identical_call_is_refused(self):
        same = {"labels_any": ["음식점"]}
        client = FakeClient([
            response([tool("t1", "search_cards", same)], "tool_use"),
            response([tool("t2", "search_cards", same)], "tool_use"),
            response([text("요약: 2건\n이유: 라벨\nRESULT: r1")], "end_turn"),
        ])
        res = agent.Asker(self.index, client=client).ask("음식")
        second = client.requests[2]["messages"][-1]["content"][0]
        self.assertTrue(second["is_error"])
        self.assertIn("이미 불렀다", second["content"])
        self.assertEqual(res.repeated_calls, 1)
        self.assertEqual(res.search_totals, [2])  # 거절된 호출은 검색으로 세지 않는다

    def test_comma_string_for_array_is_split(self):
        out, is_err = agent.run_tool(self.index, "search_cards", {"labels_any": "음식점, 커피잔", "sort": " date_desc "})
        self.assertFalse(is_err)
        self.assertEqual(json.loads(out)["total"], 2)
        out, is_err = agent.run_tool(self.index, "search_cards", {"labels_any": "음식점|커피잔"})
        self.assertEqual(json.loads(out)["total"], 2)

    def test_images_are_counted_not_assumed(self):
        msgs = [{"role": "user", "content": "q"},
                {"role": "user", "content": [{"type": "image", "source": {}},
                                              {"type": "tool_result", "content": [{"type": "image"}]}]}]
        self.assertEqual(sum(agent._count_images(m["content"]) for m in agent._messages_dump(msgs)
                             if isinstance(m["content"], list)), 2)

    def test_budget_guard_blocks_call(self):
        llm_log.append({"cost_usd": config.BUDGET_USD})
        client = FakeClient([])
        res = agent.Asker(self.index, client=client).ask("음식")
        self.assertEqual(res.stop, "budget")
        self.assertEqual(client.requests, [])

    def test_gemini_spend_does_not_count_against_agent_budget(self):
        llm_log.append({"provider": "gemini", "cost_usd": config.BUDGET_USD})   # 캡션 비용은 따로 센다
        client = FakeClient([response([text("요약: 없음\n이유: -\nRESULT: none")], "end_turn")])
        res = agent.Asker(self.index, client=client).ask("음식")
        self.assertEqual(res.stop, "answered")
        self.assertEqual(llm_log.read_all()[-1]["provider"], "anthropic")

    # ---- 질문당 예산·refusal·max_tokens·turn_limit 분기 ----
    def test_query_budget_stops_and_falls_back(self):
        saved = config.QUERY_BUDGET_USD
        config.QUERY_BUDGET_USD = 0.000001      # 첫 호출 뒤 바로 넘는다
        try:
            client = FakeClient([response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use")])
            res = agent.Asker(self.index, client=client).ask("음식")
        finally:
            config.QUERY_BUDGET_USD = saved
        self.assertEqual(res.stop, "query_budget")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(res.result_id, "r1")    # 그때까지의 결과로 끝낸다
        self.assertTrue(res.result_fallback)
        self.assertEqual(res.total, 2)

    def test_refusal_branch(self):
        resp = response([text("")], "refusal")
        resp.stop_details = SimpleNamespace(category="test_category")
        res = agent.Asker(self.index, client=FakeClient([resp])).ask("음식")
        self.assertEqual(res.stop, "refusal")
        self.assertIn("test_category", res.error)
        self.assertIsNone(res.result_id)

    def test_max_tokens_branch(self):
        client = FakeClient([response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
                             response([text("요약: 후보 2")], "max_tokens")])
        res = agent.Asker(self.index, client=client).ask("음식")
        self.assertEqual(res.stop, "max_tokens")
        self.assertIn("max_tokens", res.error)

    def test_turn_limit_branch(self):
        client = FakeClient([
            response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
            response([tool("t2", "search_cards", {"labels_any": ["커피잔"]})], "tool_use"),
        ])
        res = agent.Asker(self.index, client=client, max_turns=2).ask("음식")
        self.assertEqual(res.stop, "turn_limit")
        self.assertEqual(res.turns, 2)
        self.assertEqual(res.result_id, "r2")    # 마지막 0건 아닌 검색
        self.assertTrue(res.result_fallback)

    # ---- 답에 한자가 섞이면 한 번 다시 쓰게 하고, 「요약」 표제는 코드가 붙인다 ----
    def test_han_chars_trigger_one_rewrite(self):
        client = FakeClient([
            response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
            response([text("要약: 후보 2건 料理\n이유: 라벨\nRESULT: r1")], "end_turn"),
            response([text("요약: 후보 2건 요리\n이유: 라벨\nRESULT: r1")], "end_turn"),
        ])
        res = agent.Asker(self.index, client=client).ask("음식")
        self.assertTrue(res.han_retry)
        self.assertEqual(res.han_left, 0)
        self.assertEqual(res.answer, "요약: 후보 2건 요리\n이유: 라벨")
        self.assertIn("한자", client.requests[2]["messages"][-1]["content"])
        self.assertEqual(res.stop, "answered")

    def test_header_only_han_is_fixed_by_code_without_rewrite(self):
        client = FakeClient([
            response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
            response([text("要약: 후보 2건\n이유: 라벨\nRESULT: r1")], "end_turn"),
        ])
        res = agent.Asker(self.index, client=client).ask("음식")
        self.assertFalse(res.han_retry)                  # 표제의 한자는 코드가 떼고 다시 붙인다
        self.assertTrue(res.answer.startswith("요약: 후보 2건"))
        self.assertEqual(len(client.requests), 2)

    def test_parse_final_keeps_body_that_only_looks_like_a_summary_label(self):
        """첫 줄의 「약국:」 같은 본문은 표제로 오인해 지우지 않는다. 진짜 요약 표제만 코드가 갈아 붙인다."""
        body, _, rid, _ = agent.parse_final("약국: 근처에서 찍은 사진은 3건입니다.\n이유: 라벨\nRESULT: r1")
        self.assertEqual(body, "요약: 약국: 근처에서 찍은 사진은 3건입니다.\n이유: 라벨")
        self.assertEqual(rid, "r1")
        for head in ("요약", "要약", "要約", "Summary", "**요약**"):     # 표제 변형은 그대로 정리됨
            body, _, _, _ = agent.parse_final(f"{head}: 후보 3건\n이유: 라벨\nRESULT: r1")
            self.assertEqual(body, "요약: 후보 3건\n이유: 라벨", head)

    def test_second_han_answer_is_accepted_and_counted(self):
        client = FakeClient([
            response([text("요약: 料理 없음\n이유: -\nRESULT: none")], "end_turn"),
            response([text("요약: 料理 없음\n이유: -\nRESULT: none")], "end_turn"),
        ])
        res = agent.Asker(self.index, client=client).ask("음식")
        self.assertTrue(res.han_retry)
        self.assertEqual(res.han_left, 2)
        self.assertEqual(res.stop, "answered")

    def test_new_v1_filter_args_are_accepted(self):
        out, is_err = agent.run_tool(self.index, "search_cards", {
            "cuisine": "양식|한식", "caption_like": "파스타", "doc_type": ["영수증"],
            "vision_labels_any": "개,고양이", "vision_min_conf": 0.05, "apple_label_min_conf": 0.1})
        self.assertFalse(is_err)
        self.assertEqual(json.loads(out)["total"], 0)     # 가짜 인덱스엔 캡션이 없다 → 0건(오류 아님)
        out, is_err = agent.run_tool(self.index, "list_vocab", {"kind": "vision_labels", "min_conf": 0.02})
        self.assertFalse(is_err)
        out, is_err = agent.run_tool(self.index, "search_cards", {"vision_labels_any": ["개"], "vision_min_conf": 0.001})
        self.assertTrue(is_err)                            # 저장 하한(0.02) 아래는 입력 오류로 돌려준다

    def test_captionless_index_still_answers(self):
        """캡션이 0장이어도: 요약이 없다는 사실을 알리고, 캡션 필터는 0건일 뿐 예외가 아니며, 라벨 검색은 그대로 돈다."""
        summary = self.index.summary_text()
        self.assertIn("캡션: 없음", summary)
        for args in ({"cuisine": ["양식"]}, {"caption_like": "파스타", "doc_type": ["영수증"]},
                     {"vision_labels_any": ["개"]}, {"apple_label_min_conf": 0.1, "labels_any": ["음식점"]}):
            out, is_err = agent.run_tool(self.index, "search_cards", args)
            self.assertFalse(is_err, args)
        self.assertEqual(json.loads(agent.run_tool(self.index, "search_cards", {"cuisine": ["양식"]})[0])["total"], 0)
        vocab, is_err = agent.run_tool(self.index, "list_vocab", {"kind": "cuisine"})
        self.assertFalse(is_err)
        self.assertEqual(json.loads(vocab)["entries"], [])
        client = FakeClient([
            response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
            response([text("요약: 캡션이 없어 라벨로 찾았습니다\n이유: 라벨\nRESULT: r1")], "end_turn"),
        ])
        res = agent.Asker(self.index, client=client).ask("음식")
        self.assertEqual((res.stop, res.total), ("answered", 2))
        self.assertIn("캡션: 없음", client.requests[0]["system"][0]["text"])   # 모델에게도 없다고 간다

    # ---- 웹 화면용 재개(start/resume) ----
    def test_start_then_resume_answer_matches_interactive_cli(self):
        client = FakeClient([
            response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
            response([text("요약: 2건\n이유: 라벨\nRESULT: r1\nQUESTION: 언제쯤?")], "end_turn"),
            response([tool("t2", "search_cards", {"labels_any": ["음식점"], "date_from": "2026-09-01"})], "tool_use"),
            response([text("요약: 1건\n이유: 9월로 좁힘\nRESULT: r2")], "end_turn"),
        ])
        asker = agent.Asker(self.index, client=client)
        conv = asker.start("음식")
        self.assertEqual((conv.res.stop, conv.res.total), ("asked", 2))
        self.assertTrue(conv.resumable)
        asker.resume(conv, "9월")
        self.assertEqual((conv.res.stop, conv.res.total, conv.res.user_reply, conv.res.legs), ("answered", 1, "9월", 1))
        self.assertEqual(conv.res.turns, 4)
        self.assertIn("r1", self.index.results)        # 앞 결과 묶음은 대화가 끝날 때까지 남는다

    def test_follow_up_after_answer_is_new_leg_with_fresh_limits(self):
        client = FakeClient([
            response([tool("t1", "search_cards", {"labels_any": ["음식점"]})], "tool_use"),
            response([text("요약: 2건\n이유: 라벨\nRESULT: r1\nQUESTION: 언제쯤?")], "end_turn"),
            response([text("요약: 2건\n이유: 그대로\nRESULT: r1")], "end_turn"),
            response([text("요약: 2건\n이유: 라벨\nRESULT: r1\nQUESTION: 사람이 나온 사진만?")], "end_turn"),
        ])
        asker = agent.Asker(self.index, client=client, max_turns=3)
        conv = asker.start("음식")
        asker.resume(conv, "모르겠어요")                 # 되물음의 답 — 같은 구간(호출 2+1 = 상한 3을 다 씀)
        self.assertEqual((conv.res.stop, conv.res.turns), ("answered", 3))
        asker.resume(conv, "더 좁게")                    # 새 구간: 호출 상한을 새로 세고(아니면 turn_limit) 다시 되물을 수 있다
        self.assertEqual((conv.res.stop, conv.res.legs, conv.res.followups), ("asked", 2, ["더 좁게"]))
        self.assertEqual(conv.res.clarify_question, "사람이 나온 사진만?")
        self.assertIsNone(conv.res.user_reply)

    def test_resume_refuses_non_resumable(self):
        resp = response([text("")], "refusal")
        resp.stop_details = SimpleNamespace(category="x")
        asker = agent.Asker(self.index, client=FakeClient([resp]))
        conv = asker.start("음식")
        self.assertFalse(conv.resumable)
        with self.assertRaises(ValueError):
            asker.resume(conv, "더")


if __name__ == "__main__":
    unittest.main()
