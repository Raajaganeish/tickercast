"""Collect check results and write them as a Markdown report."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}


@dataclass
class Check:
    name: str
    status: str            # PASS | WARN | FAIL | INFO
    detail: str
    table: pd.DataFrame | None = None


@dataclass
class Report:
    title: str
    intro: str = ""
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, table: pd.DataFrame | None = None) -> None:
        self.checks.append(Check(name, status, detail, table))
        print(f"  {ICON[status]} {name}: {detail}")

    @property
    def n_fail(self) -> int:
        return sum(c.status == "FAIL" for c in self.checks)

    def to_markdown(self) -> str:
        counts = {s: sum(c.status == s for c in self.checks) for s in ICON}
        lines = [
            f"# {self.title}",
            "",
            f"_Generated {datetime.now():%Y-%m-%d %H:%M}_",
            "",
            self.intro,
            "",
            f"**Summary:** {counts['PASS']} passed · {counts['WARN']} warnings · "
            f"{counts['FAIL']} failed · {counts['INFO']} info",
            "",
            "| Check | Status | Result |",
            "|---|---|---|",
        ]
        lines += [f"| {c.name} | {ICON[c.status]} {c.status} | {c.detail} |" for c in self.checks]
        for c in self.checks:
            if c.table is not None and not c.table.empty:
                lines += ["", f"## {c.name}", "", c.table.to_markdown(floatfmt=".4g"), ""]
        return "\n".join(lines) + "\n"

    def save(self, path: Path) -> None:
        path.write_text(self.to_markdown())
        print(f"report -> {path}  ({self.n_fail} failures)")
