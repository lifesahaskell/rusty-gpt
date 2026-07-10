#!/usr/bin/env python3
"""Self-check for ci_triage_agent.py: run . -m unittest scripts.test_ci_triage_agent"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ci_triage_agent.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "ci_triage_sample.json"


def run_agent(input_json: Path, artifact_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            "lifesahaskell/rusty-gpt",
            "--input-json",
            str(input_json),
            "--artifact-dir",
            str(artifact_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


class CiTriageAgentTest(unittest.TestCase):
    def test_fixture_detects_and_classifies_broken_ci(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            completed = run_agent(FIXTURE, artifact_dir)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("2 broken CI finding(s)", completed.stdout)

            findings = json.loads((artifact_dir / "findings.json").read_text())
            self.assertEqual(len(findings), 2)

            by_source = {f["source"]: f for f in findings}

            pr_finding = by_source["pull-request"]
            self.assertEqual(pr_finding["severity"], "medium")
            self.assertTrue(pr_finding["trivial"])
            self.assertEqual(pr_finding["pr_number"], 42)

            run_finding = by_source["workflow-run"]
            self.assertEqual(run_finding["severity"], "high")
            self.assertTrue(run_finding["trivial"])
            self.assertEqual(run_finding["validation_command"], "cargo fmt --all -- --check")

            report_files = list((artifact_dir / "reports").glob("*.md"))
            self.assertEqual(len(report_files), 2)
            brief_files = list((artifact_dir / "briefs").glob("*.md"))
            self.assertEqual(len(brief_files), 2)

    def test_empty_input_reports_zero_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp)
            empty_fixture = artifact_dir / "empty.json"
            empty_fixture.write_text(json.dumps({"repository": "o/r", "default_branch": "main"}))

            completed = run_agent(empty_fixture, artifact_dir)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("0 broken CI finding(s)", completed.stdout)
            self.assertEqual(json.loads((artifact_dir / "findings.json").read_text()), [])
            self.assertIn("No broken CI findings", (artifact_dir / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
