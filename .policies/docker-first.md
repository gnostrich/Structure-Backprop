# Global Docker-First Development Protocol

## Rule
**ALL code execution and development work must be done inside Docker containers.**
**Files can be edited on host (they are mounted) but all running/testing must be in Docker.**

## Workflow

### Host-Level Operations (OK - files are mounted)
- ✓ Edit files using host text editors (they sync to Docker via mount)
- ✓ View/read files from host
- ✓ Git operations (commit/push) from host
- ✓ GitHub CLI commands from host

### Docker-Bound Operations (Must use docker exec)
- ✓ `docker exec <container> python` - run Python code
- ✓ `docker exec <container> bash -c "..."` - run commands
- ✓ `docker exec <container> npm/node` - run JS
- ✓ `docker exec <container> pytest` - run tests
- ✓ `docker exec <container> git <cmd>` - git inside container

### Auto-Redirect to Docker (Agent should self-correct)
Agent must rewrite these to Docker:
- ✗ `python script.py` → `docker exec <container> python script.py`
- ✗ `pip install` → `docker exec <container> pip install`
- ✗ Running tests locally → `docker exec <container> pytest`
- ✗ Code execution/testing → Always in Docker

### Request User Approval
- Container creation/startup (one-time setup)
- System configuration changes

## Container Management

Agent should:
1. Check if appropriate container exists for repo
2. Create/start container if needed
3. Use volume mounts to sync workspace
4. Execute all dev commands inside container

## Protocol Enforcement

Before executing ANY command:
1. **Check**: Does this modify code, install packages, or run code?
2. **Redirect**: If yes, rewrite to use `docker exec` automatically
3. **Proceed**: Execute without asking (unless host-only operation)

## Global Exceptions (Always require host)
- `open <file>` - OS file viewer
- `git commit/push/pull` - Git operations
- `gh` commands - GitHub CLI
- VS Code operations - Editor commands

---
**Location**: `~/.ai-dev-protocol/docker-first.md`
**Applies to**: All repositories
**Agent responsibility**: Check and follow this protocol automatically

## Shell Command Guidelines

**Avoid**:
- Heredocs in `docker exec` (causes shell corruption)
- Python multiline strings in docker commands
- Long compound commands with quotes/escaping

**Prefer**:
- Simple atomic commands
- File operations on host (mounted volumes)
- Python one-liners without nested quotes
