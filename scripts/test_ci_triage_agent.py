#!/usr/bin/env python3
"""Self-check for ci_triage_agent.py: run . -m unittest scripts.test_ci_triage_agent"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "ci_triage_agent.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "ci_triage_sample.json"


def load_agent_module():
    spec = importlib.util.spec_from_file_location("ci_triage_agent", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Must be registered before exec: @dataclass resolves cls.__module__ via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


class EnsureLabelsTest(unittest.TestCase):
    """Labels already exist after the first run, so provisioning must be idempotent.

    Regression guard: the previous implementation POSTed to the labels API and tried to
    tolerate duplicates by string-matching stderr. `gh api` reports a duplicate as a bare
    "Validation Failed (HTTP 422)", the match never fired, and every scheduled run raised.
    """

    def setUp(self) -> None:
        self.agent = load_agent_module()
        self.calls: list[list[str]] = []
        self.agent.run = lambda argv, **kwargs: self.calls.append(argv)

    def test_uses_forced_label_create_so_duplicates_are_not_errors(self) -> None:
        self.agent.ensure_labels("o/r", ["ci-triage", "ci-severity/low"], dry_run=False)

        self.assertEqual(len(self.calls), 2)
        for argv in self.calls:
            self.assertEqual(argv[:3], ["gh", "label", "create"])
            self.assertIn("--force", argv)
            self.assertEqual(argv[argv.index("--repo") + 1], "o/r")
        self.assertEqual([argv[3] for argv in self.calls], ["ci-severity/low", "ci-triage"])

    def test_dry_run_touches_github_not_at_all(self) -> None:
        self.agent.ensure_labels("o/r", ["ci-triage"], dry_run=True)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
