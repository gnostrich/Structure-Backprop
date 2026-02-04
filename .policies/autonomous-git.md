# Autonomous Git & PR Policy

## Authority Granted
Agent has full authority to commit, push, and manage PRs without user approval.

## Auto-Approve Operations

### Git Commits
- ✓ Commit any changes made during development sessions
- ✓ Use descriptive commit messages based on work done
- ✓ Include all relevant files in commits
- ✓ Push to appropriate branches

### Pull Requests
- ✓ Create PRs for feature branches
- ✓ Auto-merge enabled on all repos (configured on 2026-02-04)
- ✓ PRs merge automatically after creation
- ✓ Agent can resolve merge conflicts

### Branch Management
- ✓ Create feature branches as needed
- ✓ Switch branches
- ✓ Merge branches
- ✓ Delete stale branches

## Commit Message Format
```
<type>: <brief description>

<detailed explanation if needed>

Changes:
- Item 1
- Item 2
- Item 3
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

## When to Commit
- After completing a feature or fix
- After significant refactoring
- After adding documentation
- At end of development session
- When protocol files are updated

## Repositories with Auto-Merge Enabled
- gnostrich/Continual-Learning
- gnostrich/Structure-Backprop
- gnostrich/clawpilot
- gnostrich/spores

## GitHub Authentication
Uses existing `gh` CLI authentication (token: ghp_************************************).

## Workflow
1. Agent makes changes
2. Agent commits with descriptive message
3. Agent pushes to GitHub
4. PR auto-merges (if on feature branch)
5. No user approval needed at any step

---
**User**: gnostrich
**Configured**: 2026-02-04
**Status**: Active
