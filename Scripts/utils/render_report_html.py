from __future__ import annotations

import html
import re
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "MENTOR_PROGRESS_REPORT_2026-03-23.md"
OUTPUT = ROOT / "MENTOR_PROGRESS_REPORT_2026-03-23.html"


def extract_required_strings(markdown_text: str) -> list[str]:
    required: list[str] = []

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = re.sub(r"^#+\s*", "", stripped).strip()
            if heading:
                required.append(heading)

    required.extend(re.findall(r"`([^`]+)`", markdown_text))

    phrases = [
        "The project has progressed in three broad waves:",
        "This section is the full architecture inventory as of the current repo and the March 23 CSV.",
        "Architecture summary:",
        "Why it matters scientifically:",
        "Status in W&B:",
        "Completed and Tracked with Usable Metrics",
        "If summarized in one sentence:",
    ]
    required.extend(phrases)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in required:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def build_html(markdown_text: str) -> str:
    body = markdown.markdown(
        markdown_text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
            "toc",
        ],
        output_format="html5",
    )

    source_markdown = html.escape(markdown_text)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EEG Sleep Staging Project Report</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --paper: #fffdf8;
      --ink: #1f2328;
      --muted: #5d6670;
      --rule: #d8d1c3;
      --accent: #0f5c4d;
      --code-bg: #f4f1ea;
      --table-head: #ede7da;
      --shadow: 0 12px 40px rgba(31, 35, 40, 0.12);
    }}

    * {{
      box-sizing: border-box;
    }}

    html {{
      scroll-behavior: smooth;
    }}

    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(15, 92, 77, 0.12), transparent 28%),
        linear-gradient(180deg, #f6f1e8 0%, #efe8dd 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.65;
    }}

    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      justify-content: center;
      gap: 12px;
      padding: 14px 18px;
      backdrop-filter: blur(12px);
      background: rgba(243, 239, 231, 0.88);
      border-bottom: 1px solid rgba(216, 209, 195, 0.9);
    }}

    .toolbar button,
    .toolbar a {{
      appearance: none;
      border: 1px solid var(--accent);
      background: var(--accent);
      color: #fffdf8;
      border-radius: 999px;
      padding: 10px 16px;
      font: 600 14px/1.2 "Segoe UI", Arial, sans-serif;
      text-decoration: none;
      cursor: pointer;
    }}

    .toolbar a.secondary {{
      background: transparent;
      color: var(--accent);
    }}

    main {{
      max-width: 980px;
      margin: 32px auto 56px;
      padding: 0 24px;
    }}

    .report {{
      background: var(--paper);
      border: 1px solid rgba(216, 209, 195, 0.95);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 52px 60px;
    }}

    h1, h2, h3, h4 {{
      color: #132c28;
      line-height: 1.2;
      margin-top: 1.6em;
      margin-bottom: 0.7em;
    }}

    h1 {{
      margin-top: 0;
      font-size: 2.3rem;
      letter-spacing: -0.02em;
    }}

    h2 {{
      font-size: 1.55rem;
      border-top: 1px solid var(--rule);
      padding-top: 1.1em;
    }}

    h3 {{
      font-size: 1.2rem;
    }}

    h4 {{
      font-size: 1.05rem;
    }}

    p, li {{
      font-size: 1rem;
    }}

    ul, ol {{
      padding-left: 1.4rem;
    }}

    li + li {{
      margin-top: 0.28rem;
    }}

    blockquote {{
      margin: 1.3rem 0;
      padding: 0.7rem 1rem 0.7rem 1.2rem;
      border-left: 4px solid var(--accent);
      background: #f4f8f7;
      color: #21433d;
    }}

    code {{
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 0.92em;
      background: var(--code-bg);
      padding: 0.08em 0.32em;
      border-radius: 5px;
    }}

    pre code {{
      display: block;
      padding: 16px 18px;
      overflow-x: auto;
      border-radius: 14px;
      border: 1px solid var(--rule);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1.3rem 0 1.6rem;
      font-size: 0.96rem;
    }}

    th, td {{
      border: 1px solid var(--rule);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}

    th {{
      background: var(--table-head);
    }}

    hr {{
      border: 0;
      border-top: 1px solid var(--rule);
      margin: 2rem 0;
    }}

    .source-markdown {{
      display: none;
    }}

    @page {{
      size: A4;
      margin: 14mm 12mm 16mm;
    }}

    @media print {{
      body {{
        background: #fff;
      }}

      .toolbar {{
        display: none;
      }}

      main {{
        max-width: none;
        margin: 0;
        padding: 0;
      }}

      .report {{
        border: 0;
        box-shadow: none;
        border-radius: 0;
        padding: 0;
      }}

      h2 {{
        page-break-after: avoid;
      }}

      h3, h4, table, blockquote {{
        page-break-inside: avoid;
      }}

      a {{
        color: inherit;
        text-decoration: none;
      }}
    }}

    @media (max-width: 760px) {{
      main {{
        padding: 0 14px;
      }}

      .report {{
        padding: 26px 20px;
        border-radius: 18px;
      }}

      .toolbar {{
        flex-wrap: wrap;
      }}
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <button type="button" onclick="window.print()">Print / Save as PDF</button>
    <a class="secondary" href="#report-top">Jump to Top</a>
  </div>
  <main>
    <article class="report" id="report-top">
{body}
    </article>
  </main>
  <template class="source-markdown" id="source-markdown">{source_markdown}</template>
</body>
</html>
"""


def verify_output(markdown_text: str, html_text: str) -> None:
    normalized_html = html.unescape(html_text)
    missing = [item for item in extract_required_strings(markdown_text) if item not in normalized_html]
    if missing:
        preview = ", ".join(repr(item) for item in missing[:8])
        raise RuntimeError(f"HTML verification failed. Missing rendered content: {preview}")


def main() -> None:
    markdown_text = SOURCE.read_text(encoding="utf-8")
    html_text = build_html(markdown_text)
    verify_output(markdown_text, html_text)
    OUTPUT.write_text(html_text, encoding="utf-8")
    print(f"Wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
