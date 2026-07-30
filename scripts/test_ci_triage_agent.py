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


class UpsertIssueSearchTest(unittest.TestCase):
    """The REST endpoint `search/issues` was removed and now returns 404.

    Regression guard: this path was unreachable while ensure_labels raised first, so the
    404 only surfaced once label provisioning was fixed.
    """

    def setUp(self) -> None:
        self.agent = load_agent_module()
        self.json_calls: list[list[str]] = []
        self.run_calls: list[list[str]] = []
        self.agent.gh_json = lambda argv: self.json_calls.append(argv) or []
        self.agent.run = lambda argv, **kwargs: self.run_calls.append(argv)

    def test_searches_via_issue_list_not_the_removed_search_api(self) -> None:
        finding = self.agent.Finding(
            id="run-1-cargo-fmt",
            source="workflow-run",
            title="CI: cargo fmt",
            severity="high",
            severity_reason="default-branch workflow is broken",
            workflow="CI",
            job="cargo fmt",
            conclusion="failure",
            url="https://example.invalid/run/1",
            labels=["ci-triage"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "r.md"
            report.write_text("report")
            self.agent.upsert_issue("o/r", finding, report, dry_run=False)

        self.assertEqual(len(self.json_calls), 1)
        argv = self.json_calls[0]
        self.assertEqual(argv[:2], ["issue", "list"])
        self.assertNotIn("search/issues", argv)
        self.assertEqual(argv[argv.index("--search") + 1], "run-1-cargo-fmt")


class RunFindingIdStabilityTest(unittest.TestCase):
    """The finding id is the dedupe key upsert_issue searches on, so the same
    broken job re-running must produce the same id or every run files a new
    issue."""

    def setUp(self) -> None:
        self.agent = load_agent_module()

    def run_obj(self, run_id: int) -> dict:
        return {
            "id": run_id,
            "name": "CI",
            "head_branch": "main",
            "head_sha": f"sha{run_id}",
            "conclusion": "failure",
            "html_url": f"https://example.invalid/run/{run_id}",
        }

    def finding_id_for(self, run_id: int) -> str:
        finding = self.agent.finding_from_run(
            default_branch="main",
            run_obj=self.run_obj(run_id),
            job={"name": "cargo fmt", "conclusion": "failure"},
        )
        self.assertIsNotNone(finding)
        return finding.id

    def test_same_job_on_same_branch_keeps_one_id_across_runs(self) -> None:
        self.assertEqual(self.finding_id_for(111), self.finding_id_for(222))

    def test_id_does_not_embed_the_run_id(self) -> None:
        self.assertNotIn("111", self.finding_id_for(111))

    def test_different_jobs_stay_separate(self) -> None:
        other = self.agent.finding_from_run(
            default_branch="main",
            run_obj=self.run_obj(111),
            job={"name": "cargo clippy", "conclusion": "failure"},
        )
        self.assertNotEqual(self.finding_id_for(111), other.id)


if __name__ == "__main__":
    unittest.main()
