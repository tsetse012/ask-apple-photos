#!/usr/bin/env python3
"""사진첩에 말로 묻기.

  .venv/bin/python ask.py "저번달쯤 강원도 카페에서 찍은 사진"
  .venv/bin/python ask.py "고양이" --interactive      # 후보가 많으면 되묻고 답을 받는다
  .venv/bin/python ask.py "작년 여름 바다" --sheet --open --limit 30

밖으로 나가는 것(Anthropic): 질문 문장 + 인덱스 요약 + 도구 결과(날짜·장소·라벨·캡션 장면 한 줄·건수). 이 단계는 사진을 안 보낸다
(사진은 캡션을 만들 때 index/enrich_caption.py가 Gemini로 보낸다).
기록: logs/llm.jsonl(호출마다 한 줄), logs/transcripts/<질의 id>.json(대화 원문).
"""
import argparse
import sys

from photo_ask import config, llm_log
from photo_ask.agent import Asker
from photo_ask.cards_query import CardIndex
from photo_ask.output import copy_previews, format_table, open_paths, result_rows, write_sheet

STOP_KO = {"answered": "답함", "asked": "되물음(--interactive로 답할 수 있음)", "turn_limit": "호출 횟수 상한",
           "budget": "누적 예산 상한", "query_budget": "질문당 예산 상한", "refusal": "모델 거절",
           "max_tokens": "출력 길이 상한", "error": "오류"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--interactive", action="store_true", help="되물음에 답을 입력받는다")
    ap.add_argument("--limit", type=int, default=20, help="표에 보일 후보 수(기본 20)")
    ap.add_argument("--open", action="store_true", help="미리보기를 복사해 open으로 띄운다(--sheet면 시트를 연다)")
    ap.add_argument("--sheet", action="store_true", help="out/<id>/index.html 컨택트 시트 생성")
    ap.add_argument("--show-paths", action="store_true", help="표에 보관함 원본 미리보기 경로도 보인다(기본은 숨김)")
    ap.add_argument("--model", default=config.MODEL)
    ap.add_argument("--effort", default=config.EFFORT, choices=["low", "medium", "high"])
    ap.add_argument("--index", default=str(config.INDEX_DB))
    args = ap.parse_args()

    index = CardIndex(args.index)
    if index.conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 0:
        print("색인이 비어 있다 — index/build_index.py 를 먼저 실행해 색인을 만드세요.", file=sys.stderr)
        index.close()
        return 2
    asker = Asker(index, model=args.model, effort=args.effort, interactive=args.interactive)
    res = asker.ask(args.question)

    print(f"\n질문: {res.question}")
    if res.answer:
        print(res.answer)
    if res.clarify_question:
        print(f"되물음: {res.clarify_question}" + (f"  → 답: {res.user_reply}" if res.user_reply else ""))
    if res.error:
        print(f"⚠ {res.error}")
    filters = index.results.get(res.result_id, {}).get("filters", {}) if res.result_id else {}
    prefer = list(filters.get("labels_all") or []) + list(filters.get("labels_any") or [])
    rows = result_rows(index, res.uuids, args.limit, prefer=prefer)
    copied = copy_previews(rows, res.query_id) if (rows and (args.sheet or args.open)) else None
    if rows:
        note = " (마지막 0건 아닌 검색으로 대체)" if res.result_fallback else ""
        print(f"\n후보 {res.total}건 중 {len(rows)}건{note}:")
        print(format_table(rows, copied=copied, show_paths=args.show_paths))
    else:
        print("\n후보 없음")
    if copied is not None:
        if args.sheet:
            sheet = write_sheet(rows, copied, res.query_id, res.question, res.answer)
            print(f"\n컨택트 시트: {sheet}")
            if args.open:
                open_paths([sheet])
        elif args.open:
            open_paths(copied[:12])
    print(f"\n끝난 이유: {STOP_KO.get(res.stop, res.stop)} · 호출 {res.turns}회 · 도구 {len(res.tool_calls)}회"
          f" · 넓히기 {res.widen_count}회 · {res.seconds}초 · ${res.cost_usd:.4f}"
          f" (Anthropic 누적 ${llm_log.spent_total():.4f} / 상한 ${config.BUDGET_USD})")
    print(f"기록: {config.LLM_LOG} · {res.transcript_path}")
    return 0 if res.stop in ("answered", "asked") else 1


if __name__ == "__main__":
    sys.exit(main())
