# Tool Timeout & Fallback Protocol

## Problem
Some tools (especially create_file) can hang for 3+ minutes, blocking progress.

## Rule
If any tool invocation takes >60 seconds:
1. **Cancel immediately**
2. **Switch to simplest alternative**:
   - create_file → printf/echo to file
   - Complex Python → simple shell commands
   - Nested tool calls → direct single-step approach

## Fallback Hierarchy
1. Direct shell: `echo "content" > file`
2. Printf: `printf '%s\n' "line1" "line2" > file`
3. Python one-liner: `python3 -c "open('f','w').write('x')"`
4. Manual user paste (last resort)

## Never
- Retry the same slow tool multiple times
- Wait >2 minutes for any single operation
- Generate content "in head" then write - write directly

## Speed Targets
- File creation: <5 seconds
- Git operations: <10 seconds
- Docker commands: <30 seconds

If exceeded, pivot immediately.
