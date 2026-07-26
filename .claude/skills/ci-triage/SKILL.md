---
name: ci-triage
description: Monitor scheduled CI and pull request health, classify broken workflows by severity, delegate trivial dependency or version-bump fixes, and write actionable reports for non-trivial failures. Use when asked to triage CI, watch PR checks, summarize broken workflows, or prepare CI failure backlog items. Run from the main session — it delegates trivial fixes to subagents.
---

# CI Triage

Monitor repository CI health on a scheduled or on-demand run, classify failures, and route each broken workflow to the lightest responsible next action.

## Repo Automation

This repo attaches the agent through `.github/workflows/ci-triage-agent.yml`.
The workflow runs `scripts/ci_triage_agent.py`, writes artifacts under
`ci-triage-reports/`, and creates or updates GitHub issues labeled
`ci-triage`. See `docs/ci-triage-agent.md` for operational setup.

## Example Requests

- Use `$ci-triage` to check the repo's open PRs and report broken CI.
- Run scheduled CI triage and log anything currently failing.
- Use `$ci-triage` to fix trivial dependency-update CI failures.
- Triage the latest failed workflow runs and create backlog-ready reports for the rest.

## Mission

Produce a concise triage record of broken PRs or CI runs. When a failure is clearly trivial, such as a dependency version bump, lockfile refresh, linter metadata update, or generated snapshot refresh, delegate a focused fix to a subagent if subagents are available and the environment permits changes. For anything higher-risk, create a report another agent can turn into a backlog item.

## Workflow

1. Establish scope: repository, branch, open PRs, recent scheduled runs, and time window. Default to open PRs plus failed workflow runs from the last 24 hours when the request does not specify scope.
2. Gather CI state from primary sources: GitHub PR checks, workflow runs, job logs, failing annotations, and recent commits. Prefer GitHub app tools when available; use `gh` only when connector coverage is insufficient.
3. Group failures by root cause, not by repeated job. Note affected PRs, workflow names, failing jobs, first failing commit, and whether the failure is reproducible across retries.
4. Classify severity:
   - `critical`: default branch release/deploy blocked, security workflow broken, many PRs blocked by shared infrastructure, or production emergency workflow failing.
   - `high`: multiple active PRs blocked, required checks consistently failing, or dependency/security updates blocked.
   - `medium`: one PR blocked by a likely code or test issue.
   - `low`: flaky, optional, documentation-only, or non-required workflow failures.
5. Decide action:
   - Trivial and low-risk: delegate a subagent to implement the smallest fix, then validate with the failing command or workflow-equivalent command.
   - Non-trivial: write a backlog-ready report using `references/ci-triage-report-template.md`.
   - Ambiguous: report the ambiguity and the smallest investigation needed; do not make broad code changes.
6. Log the triage outcome in the user's requested place. If no destination is given, provide the report in the response and name any local file only if you created one.

## Trivial Fix Criteria

Treat a fix as trivial only when all are true:

- The failing signal points to a mechanical update: dependency version, lockfile, generated snapshot, formatter output, lint config drift, or documented deprecation rename.
- The fix is contained to the dependency/config/generated files or one small call-site adjustment.
- The validation command is known and can run locally or has a reliable equivalent.
- The change does not alter product behavior, public APIs, schemas, authentication, deployment topology, or data migration paths.

When any criterion is false, create a report instead of delegating an automatic fix.

## Delegation

When subagents are available, spawn exactly one focused subagent per trivial root cause. Give the subagent:

- The failing workflow/job and PR or commit.
- The suspected root cause.
- The allowed file scope.
- The validation command.
- A requirement to avoid unrelated changes.
- A requirement that any branch or PR is independently mergeable from the remote target branch, usually `origin/main`, with no stacked commits unless explicitly requested.

Review the subagent result before reporting success. If subagents are unavailable, state that the issue is trivial-fix eligible and provide the same brief for a later agent.

## Report Requirements

Each non-trivial report must include:

- Severity and reason.
- Affected PRs, branches, workflows, jobs, and URLs when available.
- Failure signature with the smallest useful log excerpt.
- Likely owner or area if inferable from files or workflow names.
- Reproduction or validation command.
- Recommended next step suitable for backlog grooming.
- Confidence level and open questions.

## Non-Goals

- Do not run as a long-lived daemon from inside the agent session. This skill supports one scheduled invocation at a time.
- Do not retry or rerun workflows unless the user explicitly asks or the repository policy requires it.
- Do not auto-fix behavioral regressions, security failures, deployment failures, schema/data changes, or broad test breakage.
- Do not hide flaky failures; record them with lower severity and suggested quarantine or stabilization work.

## Validation

Before finishing:

- Confirm every reported failure came from a current PR check or workflow run within scope.
- Confirm trivial fixes were validated with the targeted failing command or explain why validation could not run.
- Confirm non-trivial failures have backlog-ready reports.
- Confirm unrelated repository changes were not included in any generated fix.
- Confirm any generated fix branch compares cleanly against the remote target branch and contains only the intended fix commits.
