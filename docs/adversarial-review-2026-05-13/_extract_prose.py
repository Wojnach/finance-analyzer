"""Extract assistant prose blocks from codex exec transcript stderr files.

The 6 codex runs that hit the OpenAI usage limit never emitted their final
`-output-last-message` markdown. Their reasoning + observations survive in
the stderr transcript as blocks between `codex` markers. Pull those.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent
STDERR_DIR = ROOT / "_codex_out"

# Only extract for runs that DID NOT produce a final markdown report.
TRUNCATED = [
    "1-signals-core",
    "2-orchestration",
    "3-portfolio-risk",
    "4-metals-core",
    "6-signals-modules",
    "7-data-external",
]


def extract_prose(stderr_path: Path) -> str:
    out: list[str] = []
    in_codex = False
    for raw in stderr_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if line == "codex":
            in_codex = True
            out.append("\n---\n")
            continue
        if not in_codex:
            continue
        # End-of-prose markers
        if line.startswith(("exec ", "exec\t")) or line == "exec":
            in_codex = False
            continue
        if line.startswith(("tokens used", "ERROR", "user", "thinking")):
            in_codex = False
            continue
        if re.match(r"^\d{4}-\d{2}-\d{2}T", line):
            in_codex = False
            continue
        # Skip raw file-content lines like "    1: import logging"
        if re.match(r"^\s+\d+:\s", line):
            in_codex = False
            continue
        if re.match(r"^\d+:\s", line):
            in_codex = False
            continue
        if line.startswith(" succeeded") or line.startswith(" declined"):
            in_codex = False
            continue
        if line.strip() == "":
            continue
        out.append(line)
    return "\n".join(out)


def main() -> None:
    for base in TRUNCATED:
        stderr_path = STDERR_DIR / f"{base}.stderr"
        if not stderr_path.exists():
            print(f"missing {stderr_path.name}")
            continue
        prose = extract_prose(stderr_path)
        target = ROOT / f"{base}-codex-prose.md"
        body = f"# Codex prose extract: {base}\n\n"
        body += "_(Codex run hit the OpenAI usage limit before emitting its final report. "
        body += "This is the model's narrative observations extracted from the live transcript.)_\n"
        body += prose
        target.write_text(body, encoding="utf-8")
        print(f"{target.name}: {len(prose.splitlines())} lines")


if __name__ == "__main__":
    main()
