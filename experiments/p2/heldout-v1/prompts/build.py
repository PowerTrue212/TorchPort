"""Render prompts/none.md and prompts/full.md from body.md + tools-*.md.

Verifies that the two outputs differ only inside the TOOLS section.
Run: python build.py
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTION = re.compile(r"<!-- TOOLS:BEGIN -->.*?<!-- TOOLS:END -->", re.S)


def render(body: str, tools: str) -> str:
    return body.replace("{{TOOLS}}", tools.strip())


def main() -> None:
    body = (HERE / "body.md").read_text(encoding="utf-8")
    out = {}
    for cond in ("none", "full"):
        tools = (HERE / f"tools-{cond}.md").read_text(encoding="utf-8")
        out[cond] = render(body, tools)
        (HERE / f"{cond}.md").write_text(out[cond], encoding="utf-8", newline="\n")

    stripped = {k: SECTION.sub("<!-- TOOLS -->", v) for k, v in out.items()}
    diff = list(difflib.unified_diff(stripped["none"].splitlines(), stripped["full"].splitlines(), lineterm=""))
    if diff:
        raise SystemExit("prompts differ outside the tool section:\n" + "\n".join(diff))
    full_diff = [l for l in difflib.unified_diff(out["none"].splitlines(), out["full"].splitlines(), lineterm="")
                 if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    print(f"OK: only the tool section differs ({len(full_diff)} changed lines).")


if __name__ == "__main__":
    main()
