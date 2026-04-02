# AGENTS.md

Generally speaking, you should browse the codebase to figure out what is going on.

## Task Completion Requirements

- Run smoke tests to verify proper function
- When making a change, keep code simple, readable, and maintainable.

## Project Snapshot

This project's goal is to determine whether the multifractal analysis spectrum and structure functions contain information relevant to predicting internet traffic.

This repository is WIP. Proposing sweeping changes that improve long-term maintainability is encouraged.

## Core Priorities

1. Performance first.
2. Reliability first.

If a tradeoff is required, choose correctness and robustness over short-term convenience.

Do not add excessive fallbacks. Errors like missing env are critical and shouldn't be masked by fallbacks. Logic should be simple, with reasonable expecations, don't `try except` everything. Use the smallest possible diff. Then think of how to make it smaller. Don't add helpers. Do not use fallbacks with ternaries or the || operator. No typeof checks. No backwards compatability. Smallest possible set of changes to make the instructed change work.

Keep files under ~400 lines. Refactor as neeeded to meet this.

## Maintainability

Long term maintainability is a core priority. If you add new functionality, first check if there are shared logic that can be extracted to a separate module. Duplicate logic across mulitple files is a code smell and should be avoided. Don't be afraid to change existing code. Don't take shortcuts by just adding local logic to solve a problem.

## Package Roles

- `src/`: ML code location

## Expectations

- use uv for python dependency management

## References

- Original plan ./docs/project_description.md
- Skills: use ~/.agents/skills/find-skills to locate relevant skills wherever possible