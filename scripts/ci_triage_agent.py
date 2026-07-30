#!/usr/bin/env python3
"""Scheduled CI triage for GitHub Actions.

The script is intentionally dependency-free: it shells out to `gh` for live
GitHub data, writes local triage artifacts, and optionally creates or updates
GitHub issues. Trivial fix execution is delegated through an explicit command
hook so the repo can choose which coding agent is allowed to make changes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BROKEN_CONCLUSIONS = {
    "action_required",
    "cancelled",
    "failure",
    "startup_failure",
    "timed_out",
}

TRIVIAL_TITLE_RE = re.compile(
    r"\b(bump|deps?|dependabot|dependency|dependencies|renovate|lockfile|"
    r"format|formatter|rustfmt|cargo fmt|snapshot|generated)\b",
    re.IGNORECASE,
)


@dataclass
class Finding:
    id: str
    source: str
    title: str
    severity: str
    severity_reason: str
    workflow: str
    job: str
    conclusion: str
    url: str
    pr_number: int | None = None
    pr_title: str | None = None
    branch: str | None = None
    sha: str | None = None
    created_at: str | None = None
    trivial: bool = False
    trivial_reason: str | None = None
    auto_fix_status: str = "not-applicable"
    validation_command: str | None = None
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "severity": self.severity,
            "severity_reason": self.severity_reason,
            "workflow": self.workflow,
            "job": self.job,
            "conclusion": self.conclusion,
            "url": self.url,
            "pr_number": self.pr_number,
            "pr_title": self.pr_title,
            "branch": self.branch,
            "sha": self.sha,
            "created_at": self.created_at,
            "trivial": self.trivial,
            "trivial_reason": self.trivial_reason,
            "auto_fix_status": self.auto_fix_status,
            "validation_command": self.validation_command,
            "labels": self.labels,
        }


def run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(argv, text=True, capture_output=True, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(argv)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def gh_json(args: list[str]) -> Any:
    completed = run(["gh", *args])
    if not completed.stdout.strip():
        return None
    return json.loads(completed.stdout)


def slug(value: str, *, limit: int = 80) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
    return cleaned[:limit] or "finding"


def parse_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def severity_for(
    *,
    default_branch: str,
    branch: str | None,
    workflow: str,
    source: str,
    conclusion: str,
) -> tuple[str, str]:
    name = workflow.lower()
    if branch == default_branch and any(term in name for term in ("deploy", "release", "security")):
        return "critical", "default-branch release, deploy, or security workflow is broken"
    if branch == default_branch:
        return "high", "default-branch workflow is broken"
    if source == "pull-request":
        return "medium", "one pull request is blocked by a failing check"
    if conclusion in {"timed_out", "startup_failure"}:
        return "medium", "workflow infrastructure or timeout failure needs investigation"
    return "low", "recent non-PR workflow failure"


def trivial_signal(text: str, *, actor: str | None = None) -> tuple[bool, str | None, str | None]:
    haystack = text.lower()
    if actor and actor.lower().startswith(("dependabot", "renovate")):
        return True, "dependency automation authored the change", "agent-defined validation command from failing workflow"
    if "cargo fmt" in haystack or "rustfmt" in haystack or "formatter" in haystack:
        return True, "formatter-only failure", "cargo fmt --all -- --check"
    if TRIVIAL_TITLE_RE.search(text):
        return True, "mechanical dependency, generated-file, or formatting signal", "agent-defined validation command from failing workflow"
    return False, None, None


def finding_from_check(
    *,
    repo: str,
    default_branch: str,
    pull: dict[str, Any],
    check: dict[str, Any],
) -> Finding | None:
    conclusion = check.get("conclusion")
    if conclusion not in BROKEN_CONCLUSIONS:
        return None

    workflow = str(check.get("app", {}).get("name") or "GitHub Checks")
    job = str(check.get("name") or "unknown job")
    pr_number = int(pull["number"])
    pr_title = str(pull.get("title") or f"PR #{pr_number}")
    branch = pull.get("head", {}).get("ref")
    sha = pull.get("head", {}).get("sha")
    actor = pull.get("user", {}).get("login")
    severity, reason = severity_for(
        default_branch=default_branch,
        branch=branch,
        workflow=workflow,
        source="pull-request",
        conclusion=conclusion,
    )
    text = f"{pr_title} {workflow} {job} {conclusion}"
    trivial, trivial_reason, validation = trivial_signal(text, actor=actor)
    labels = ["ci-triage", f"ci-severity/{severity}"]
    if trivial:
        labels.append("trivial-fix-eligible")

    return Finding(
        id=slug(f"pr-{pr_number}-{sha}-{job}"),
        source="pull-request",
        title=f"PR #{pr_number}: {job}",
        severity=severity,
        severity_reason=reason,
        workflow=workflow,
        job=job,
        conclusion=conclusion,
        url=str(check.get("html_url") or check.get("details_url") or pull.get("html_url") or ""),
        pr_number=pr_number,
        pr_title=pr_title,
        branch=branch,
        sha=sha,
        created_at=check.get("completed_at") or check.get("started_at"),
        trivial=trivial,
        trivial_reason=trivial_reason,
        validation_command=validation,
        labels=labels,
    )


def finding_from_run(
    *,
    default_branch: str,
    run_obj: dict[str, Any],
    job: dict[str, Any] | None,
) -> Finding | None:
    conclusion = str((job or run_obj).get("conclusion") or "")
    if conclusion not in BROKEN_CONCLUSIONS:
        return None

    workflow = str(run_obj.get("name") or run_obj.get("workflow_name") or "workflow")
    job_name = str((job or {}).get("name") or workflow)
    branch = run_obj.get("head_branch")
    severity, reason = severity_for(
        default_branch=default_branch,
        branch=branch,
        workflow=workflow,
        source="workflow-run",
        conclusion=conclusion,
    )
    text = f"{workflow} {job_name} {run_obj.get('display_title') or ''} {conclusion}"
    trivial, trivial_reason, validation = trivial_signal(text, actor=run_obj.get("actor", {}).get("login"))
    labels = ["ci-triage", f"ci-severity/{severity}"]
    if trivial:
        labels.append("trivial-fix-eligible")

    return Finding(
        id=slug(f"run-{run_obj.get('id')}-{job_name}"),
        source="workflow-run",
        title=f"{workflow}: {job_name}",
        severity=severity,
        severity_reason=reason,
        workflow=workflow,
        job=job_name,
        conclusion=conclusion,
        url=str((job or {}).get("html_url") or run_obj.get("html_url") or ""),
        branch=branch,
        sha=run_obj.get("head_sha"),
        created_at=(job or run_obj).get("completed_at") or run_obj.get("created_at"),
        trivial=trivial,
        trivial_reason=trivial_reason,
        validation_command=validation,
        labels=labels,
    )


def collect_live(repo: str, window_hours: int) -> tuple[str, list[Finding]]:
    repo_info = gh_json(["api", f"repos/{repo}"])
    default_branch = str(repo_info.get("default_branch") or "main")
    findings: list[Finding] = []

    pulls = gh_json(["api", f"repos/{repo}/pulls?state=open&per_page=100"]) or []
    for pull in pulls:
        sha = pull.get("head", {}).get("sha")
        if not sha:
            continue
        checks = gh_json(["api", f"repos/{repo}/commits/{sha}/check-runs?per_page=100"]) or {}
        for check in checks.get("check_runs", []):
            finding = finding_from_check(repo=repo, default_branch=default_branch, pull=pull, check=check)
            if finding:
                findings.append(finding)

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=window_hours)
    runs_payload = gh_json(["api", f"repos/{repo}/actions/runs?status=completed&per_page=100"]) or {}
    for run_obj in runs_payload.get("workflow_runs", []):
        created = parse_time(run_obj.get("created_at"))
        if created and created < cutoff:
            continue
        if run_obj.get("conclusion") not in BROKEN_CONCLUSIONS:
            continue
        jobs_payload = gh_json(["api", f"repos/{repo}/actions/runs/{run_obj['id']}/jobs?per_page=100"]) or {}
        failed_jobs = [job for job in jobs_payload.get("jobs", []) if job.get("conclusion") in BROKEN_CONCLUSIONS]
        if not failed_jobs:
            failed_jobs = [None]
        for job in failed_jobs:
            finding = finding_from_run(default_branch=default_branch, run_obj=run_obj, job=job)
            if finding:
                findings.append(finding)

    return default_branch, dedupe_findings(findings)


def collect_from_fixture(path: Path) -> tuple[str, list[Finding]]:
    payload = json.loads(path.read_text())
    repo = str(payload.get("repository", "owner/repo"))
    default_branch = str(payload.get("default_branch", "main"))
    findings: list[Finding] = []
    checks_by_sha = payload.get("check_runs_by_sha", {})
    for pull in payload.get("pulls", []):
        sha = pull.get("head", {}).get("sha")
        for check in checks_by_sha.get(sha, {}).get("check_runs", []):
            finding = finding_from_check(repo=repo, default_branch=default_branch, pull=pull, check=check)
            if finding:
                findings.append(finding)
    jobs_by_run_id = payload.get("jobs_by_run_id", {})
    for run_obj in payload.get("workflow_runs", []):
        jobs = jobs_by_run_id.get(str(run_obj.get("id")), {}).get("jobs", [])
        if not jobs:
            jobs = [None]
        for job in jobs:
            finding = finding_from_run(default_branch=default_branch, run_obj=run_obj, job=job)
            if finding:
                findings.append(finding)
    return default_branch, dedupe_findings(findings)


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    seen: set[str] = set()
    unique: list[Finding] = []
    for finding in findings:
        if finding.id in seen:
            continue
        seen.add(finding.id)
        unique.append(finding)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    unique.sort(key=lambda item: (severity_order.get(item.severity, 9), item.title))
    return unique


def render_report(finding: Finding) -> str:
    marker = f"<!-- ci-triage-agent:finding={finding.id} -->"
    affected = f"PR #{finding.pr_number}: {finding.pr_title}" if finding.pr_number else finding.branch or "workflow run"
    return f"""{marker}
# CI Triage Report: {finding.title}

## Summary

- Severity: {finding.severity}
- Status: open
- Affected PRs or branches: {affected}
- Workflows and jobs: {finding.workflow} / {finding.job}
- First observed: {finding.created_at or "unknown"}
- Current owner or area: CI

## Failure Signature

- Conclusion: {finding.conclusion}
- URL: {finding.url or "not available"}
- Commit: {finding.sha or "unknown"}

## Impact

{finding.severity_reason}.

## Likely Cause

Confidence: medium.

Evidence points to `{finding.workflow}` job `{finding.job}` ending with `{finding.conclusion}`.

## Recommended Next Step

Investigate the failed job logs, reproduce the listed failure locally where possible, and land the smallest fix that restores the required CI check.

Validation command: {finding.validation_command or "use the failed workflow job command or rerun the workflow"}

## Automation

- Trivial fix eligible: {str(finding.trivial).lower()}
- Trivial reason: {finding.trivial_reason or "not applicable"}
- Auto-fix status: {finding.auto_fix_status}

## Open Questions

- Is the failure reproducible on retry?
- Does this block a required branch protection check?
"""


def render_brief(finding: Finding) -> str:
    allowed_scope = "dependency/config/generated/formatting files only"
    return f"""# Trivial CI Fix Brief

Finding ID: {finding.id}
Severity: {finding.severity}
Workflow/job: {finding.workflow} / {finding.job}
Conclusion: {finding.conclusion}
URL: {finding.url}
Branch: {finding.branch or "unknown"}
Commit: {finding.sha or "unknown"}

Suspected root cause:
{finding.trivial_reason}

Allowed file scope:
{allowed_scope}

Validation command:
{finding.validation_command or "derive the failing local command from the workflow job and run it"}

Rules:
- Make the smallest mechanical change.
- Do not alter product behavior, APIs, schemas, deployment config, or broad tests.
- Create an independently mergeable branch or PR from the remote target branch, usually origin/main.
- Before pushing, verify git log <target>..HEAD and git diff --name-status <target>...HEAD contain only this fix.
- Preserve unrelated worktree changes.
- Stop and report instead of guessing if the failure is not mechanical.
"""


def write_artifacts(artifact_dir: Path, findings: list[Finding]) -> None:
    reports_dir = artifact_dir / "reports"
    briefs_dir = artifact_dir / "briefs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    briefs_dir.mkdir(parents=True, exist_ok=True)

    (artifact_dir / "findings.json").write_text(
        json.dumps([finding.to_dict() for finding in findings], indent=2) + "\n"
    )

    if not findings:
        (artifact_dir / "summary.md").write_text("# CI Triage Summary\n\nNo broken CI findings in scope.\n")
        return

    lines = ["# CI Triage Summary", ""]
    for finding in findings:
        lines.append(
            f"- `{finding.severity}` {finding.title} "
            f"({finding.conclusion}, trivial={str(finding.trivial).lower()})"
        )
        report = render_report(finding)
        (reports_dir / f"{finding.id}.md").write_text(report)
        if finding.trivial:
            (briefs_dir / f"{finding.id}.md").write_text(render_brief(finding))
    (artifact_dir / "summary.md").write_text("\n".join(lines) + "\n")


def ensure_labels(repo: str, labels: list[str], *, dry_run: bool) -> None:
    colors = {
        "ci-triage": "5319e7",
        "trivial-fix-eligible": "0e8a16",
        "ci-severity/critical": "b60205",
        "ci-severity/high": "d93f0b",
        "ci-severity/medium": "fbca04",
        "ci-severity/low": "c5def5",
    }
    for label in sorted(set(labels)):
        if dry_run:
            continue
        # `gh label create --force` creates the label or updates it if it already exists,
        # exiting 0 either way. `gh api POST .../labels` returns 422 on a duplicate and
        # reports it as a bare "Validation Failed (HTTP 422)" with no machine-readable
        # reason, which is not safely distinguishable from a real validation error.
        run(
            [
                "gh",
                "label",
                "create",
                label,
                "--repo",
                repo,
                "--color",
                colors.get(label, "ededed"),
                "--force",
            ]
        )


def upsert_issue(repo: str, finding: Finding, report_path: Path, *, dry_run: bool) -> None:
    title = f"CI triage: {finding.title} [{finding.id}]"
    if dry_run:
        print(f"dry-run: would upsert issue: {title}")
        return

    # The REST endpoint `search/issues` was removed and now 404s; `gh issue list --search`
    # is repo-scoped, matches title and body, and does not consume the search rate limit.
    items = gh_json(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--search",
            finding.id,
            "--json",
            "number",
            "--limit",
            "1",
        ]
    ) or []
    labels = ",".join(finding.labels)
    if items:
        number = str(items[0]["number"])
        run(["gh", "issue", "comment", number, "--repo", repo, "--body-file", str(report_path)])
        print(f"updated existing issue #{number}: {title}")
        return

    run(
        [
            "gh",
            "issue",
            "create",
            "--repo",
            repo,
            "--title",
            title,
            "--body-file",
            str(report_path),
            "--label",
            labels,
        ]
    )
    print(f"created issue: {title}")


def run_auto_fix(command: str, finding: Finding, artifact_dir: Path) -> str:
    brief_path = artifact_dir / "briefs" / f"{finding.id}.md"
    finding_path = artifact_dir / "findings" / f"{finding.id}.json"
    finding_path.parent.mkdir(parents=True, exist_ok=True)
    finding_path.write_text(json.dumps(finding.to_dict(), indent=2) + "\n")

    env = os.environ.copy()
    env["CI_TRIAGE_BRIEF_PATH"] = str(brief_path)
    env["CI_TRIAGE_FINDING_JSON"] = str(finding_path)
    completed = subprocess.run(shlex.split(command), text=True, env=env, check=False)
    if completed.returncode == 0:
        return "auto-fix-command-succeeded"
    return f"auto-fix-command-failed:{completed.returncode}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor and report broken GitHub CI")
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--artifact-dir", type=Path, default=Path("ci-triage-reports"))
    parser.add_argument("--input-json", type=Path, help="Fixture payload instead of live GitHub")
    parser.add_argument("--create-issues", action="store_true")
    parser.add_argument("--attempt-trivial-fixes", action="store_true")
    parser.add_argument("--auto-fix-command", default=os.getenv("CI_TRIAGE_AUTO_FIX_COMMAND", ""))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.window_hours <= 0:
        raise SystemExit("--window-hours must be > 0")

    if args.input_json:
        _default_branch, findings = collect_from_fixture(args.input_json)
    else:
        _default_branch, findings = collect_live(args.repo, args.window_hours)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    write_artifacts(args.artifact_dir, findings)

    if args.attempt_trivial_fixes:
        for finding in findings:
            if not finding.trivial:
                continue
            if not args.auto_fix_command:
                finding.auto_fix_status = "trivial-fix-eligible:no-command-configured"
                continue
            finding.auto_fix_status = run_auto_fix(args.auto_fix_command, finding, args.artifact_dir)
        write_artifacts(args.artifact_dir, findings)

    if args.create_issues and findings:
        all_labels = [label for finding in findings for label in finding.labels]
        ensure_labels(args.repo, all_labels, dry_run=args.dry_run)
        for finding in findings:
            report_path = args.artifact_dir / "reports" / f"{finding.id}.md"
            upsert_issue(args.repo, finding, report_path, dry_run=args.dry_run)

    print(f"ci-triage-agent: {len(findings)} broken CI finding(s)")
    print(f"ci-triage-agent: wrote {args.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
