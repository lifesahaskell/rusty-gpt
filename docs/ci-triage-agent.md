# CI Triage Agent

This repository carries a scheduled CI triage automation in
`.github/workflows/ci-triage-agent.yml`.

The workflow runs every six hours and can also be started manually from
GitHub Actions. It inspects open pull requests plus recent workflow runs,
groups broken checks into findings, writes Markdown/JSON reports as workflow
artifacts, and creates or updates GitHub issues labeled `ci-triage`.

## Trivial Fix Automation

The triage runner only attempts automatic fixes when both conditions are true:

- The finding matches the trivial-fix criteria in
  `.codex/skills/ci-triage-agent/SKILL.md`.
- The repository variable `CI_TRIAGE_AUTO_FIX_COMMAND` is set.

The command is invoked once per trivial finding with:

- `CI_TRIAGE_BRIEF_PATH`: path to a Markdown brief for the subagent.
- `CI_TRIAGE_FINDING_JSON`: path to the finding JSON payload.

Keep the command narrow. A typical setup is a wrapper script that starts Codex
or another coding agent with the brief path, lets it make a small mechanical
change, and validates with the command listed in the brief.

Any fixer launched by this hook must create an independently mergeable branch
or PR. It should branch from the remote target branch, usually `origin/main`,
and verify that `git log <target>..HEAD` plus
`git diff --name-status <target>...HEAD` contain only the intended fix before
pushing.

When `CI_TRIAGE_AUTO_FIX_COMMAND` is not set, trivial failures are still logged
as `trivial-fix-eligible` so another agent can pick them up.

## Local Dry Run

Use a captured JSON fixture or the live GitHub CLI:

```bash
python3 scripts/ci_triage_agent.py \
  --repo lifesahaskell/rusty-gpt \
  --window-hours 24 \
  --artifact-dir /tmp/ci-triage \
  --dry-run
```

Add `--create-issues` only when you want the run to write to GitHub.
