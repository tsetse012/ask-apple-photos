"""결과 표·컨택트 시트·미리보기 열기. 전부 로컬에서만 쓴다(LLM에 안 보낸다)."""
import html
import json
import shutil
import subprocess
from pathlib import Path

from . import config
from .cards_query import CardIndex


CAPTION_COL_MAX = 40


def result_rows(index: CardIndex, uuids, limit: int, prefer=()) -> list:
    rows = []
    for u in uuids[:limit]:
        r = index.card(u)
        if r is None:
            continue
        venues = json.loads(r["venues"] or "[]")
        place = " · ".join(x for x in [r["city"] or r["sub_admin"], r["state"] or r["country"]] if x)
        if venues:
            place = f"{place} · {venues[0]}" if place else venues[0]
        cap = index.caption_of(u)
        caption = cap["scene_ko"] if cap else ""
        if cap and cap.get("cuisine") not in (None, "없음"):
            caption = f"[{cap['cuisine']}] {caption}"
        rows.append({
            "uuid": u, "date": (r["date_local"] or "")[:16].replace("T", " "),
            "place": place or "-", "labels": ", ".join(index.top_labels_for(r, prefer=prefer)) or "-",
            "caption": caption or "-", "video": bool(r["is_video"]), "preview": r["preview_path"],
        })
    return rows


def _cell(text: str, limit: int = None) -> str:
    text = (text or "-").replace("|", "/").replace("\n", " ")
    return text if not limit or len(text) <= limit else text[: limit - 1] + "…"


def _short(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(config.ROOT))
    except ValueError:
        return str(path)


def format_table(rows, copied=None, show_paths: bool = False) -> str:
    """결과 표. 보관함 원본 경로는 show_paths일 때만 — 기본은 uuid 앞 8자,
    copied(out/ 사본 경로 목록)를 주면 그 사본 경로를 함께 보인다."""
    head = ["#", "날짜", "장소", "라벨 3개", "캡션", "uuid"]
    if copied is not None:
        head.append("사본(out/)")
    if show_paths:
        head.append("원본 경로")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, r in enumerate(rows, 1):
        kind = " (영상)" if r["video"] else ""
        cells = [str(i), f"{r['date']}{kind}", _cell(r["place"]), _cell(r["labels"]),
                 _cell(r.get("caption"), CAPTION_COL_MAX), r["uuid"][:8]]
        if copied is not None:
            path = copied[i - 1] if i - 1 < len(copied) else None
            cells.append(_short(path) if path else "-")
        if show_paths:
            cells.append(r["preview"] or "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def copy_previews(rows, query_id: str) -> list:
    """미리보기를 out/<질의 id>/ 로 복사(보관함 안 파일을 앱으로 직접 열지 않으려고)."""
    dest = config.OUT_DIR / query_id
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for i, r in enumerate(rows, 1):
        src = r["preview"]
        if not src or not Path(src).exists():
            copied.append(None)
            continue
        target = dest / f"{i:02d}_{r['uuid'][:8]}{Path(src).suffix}"
        shutil.copyfile(src, target)
        copied.append(target)
    return copied


def write_sheet(rows, copied, query_id: str, question: str, answer: str) -> Path:
    cells = []
    for i, (r, path) in enumerate(zip(rows, copied), 1):
        img = (f'<img src="{html.escape(path.name)}" loading="lazy">' if path
               else '<div class="noimg">미리보기 없음</div>')
        kind = " · 영상" if r["video"] else ""
        cap = r.get("caption") or "-"
        cells.append(f'<figure>{img}<figcaption><b>{i}</b> {html.escape(r["date"])}{kind}<br>'
                     f'{html.escape(r["place"])}<br>{html.escape(cap)}<br>'
                     f'<small>{html.escape(r["labels"])} · {r["uuid"][:8]}</small>'
                     f'</figcaption></figure>')
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>photo-ask {html.escape(query_id)}</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:16px;background:#fafafa;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px}}
figure{{margin:0;background:#fff;border:1px solid #ddd;border-radius:8px;overflow:hidden}}
img{{width:100%;height:180px;object-fit:cover;display:block}}
.noimg{{height:180px;display:flex;align-items:center;justify-content:center;color:#999}}
figcaption{{padding:6px 8px;font-size:12px;line-height:1.4}}
pre{{white-space:pre-wrap;background:#fff;border:1px solid #ddd;padding:8px;border-radius:8px}}
</style></head><body>
<h3>{html.escape(question)}</h3><pre>{html.escape(answer)}</pre>
<div class="grid">{''.join(cells)}</div></body></html>"""
    path = config.OUT_DIR / query_id / "index.html"
    path.write_text(doc, encoding="utf-8")
    return path


def open_paths(paths) -> None:
    paths = [str(p) for p in paths if p]
    if paths:
        subprocess.run(["open", *paths], check=True)
