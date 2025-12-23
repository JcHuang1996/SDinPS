# AGENTS.md (Cursor Agent Instructions)

## Mission
Help implement, modify, and debug a **Python mixed-integer linear programming (MILP)** codebase with:
- minimal external dependencies,
- strong consistency with existing project structure,
- robust verification even when Gurobi cannot run in the sandbox.

---

## Dependency policy (very strict)
### Allowed (prefer these)
- Standard library
- `numpy`, `pandas` (and similarly common academic basics)

### Optimization stack (allowed)
- `pyomo`
- `gurobipy` / Gurobi (primary)
- Open-source solvers **only if needed / easy to enable without new heavy deps**:
  - e.g., GLPK/CBC/HiGHS via Pyomo if available in the environment

### Not allowed without explicit user approval
- Any heavy / niche package (e.g., torch, tensorflow, jax, ray, dask, pyspark, etc.)
- Adding brand-new dependencies “just for convenience”
- Downloading external binaries or using network fetches to install solvers

---

## Project structure & naming (highest priority)
1. **Follow existing style and structure first.**
   - Before coding, scan a few representative modules and follow their patterns (layout, naming, logging, IO).
2. **Shared names must be centralized.**
   - If a data object / variable / table header is used across multiple files, define it in the existing central location
     such as `util/names.py` or `util/headers.py`.
   - Do **not** scatter string literals across files if the repo already centralizes them.
3. **Minimize diffs.**
   - Prefer small, localized edits over “refactors”.
   - Avoid renaming public functions/variables unless required.
4. **No parallel frameworks.**
   - If the project uses Pyomo, do not introduce a second modeling layer (e.g., PuLP) unless explicitly requested.

---

## Gurobi availability & fallback strategy
Gurobi may not be available in Cursor’s sandbox. When Gurobi cannot run:

### A) First choice: “build-only” verification (no solver)
- Construct the model and verify structure without solving:
  - variables exist with correct indexing and bounds,
  - constraints are created with correct counts and expected forms,
  - objective is set and finite,
  - data loading matches expected shapes/columns,
  - no silent dimension mismatches (e.g., missing nodes/times/generators).

Implement a **`--dry-run` / `dry_run=True`** path if the project already has a CLI/config pattern. Otherwise, keep it simple:
- a small `smoke_test()` function in the most appropriate existing module,
- or a lightweight test file if the repo already has tests.

### B) Second choice: small-instance logic tests / simulation
When solving isn’t possible, still validate correctness by:
- generating a tiny deterministic instance (seeded),
- checking constraints numerically on handcrafted feasible solutions,
- verifying corner cases (zero demand, single generator, disconnected node, etc.).

### C) Open-source solver fallback (only if practical)
If an open-source solver is available in the environment (or already used in the repo), allow:
- `SolverFactory("highs")`, `SolverFactory("cbc")`, or `SolverFactory("glpk")`
but **do not** add heavy new dependencies to make this happen.

---

## Coding standards (lightweight, repo-aligned)
- Use the project’s existing formatting and conventions.
- Prefer type hints for public functions when consistent with the repo.
- Prefer small, testable functions over monolithic scripts.
- Keep I/O stable: don’t change file formats, column names, or JSON schemas unless requested.
- Avoid global state; pass configs explicitly if the repo does so.

---

## Change workflow (how you should operate)
1. **Read before writing**
   - Identify the exact entrypoints and existing patterns.
2. **Plan minimal edits**
   - List files you will touch and why.
3. **Implement**
   - Keep changes minimal and consistent with existing style.
4. **Verify**
   - If solver works: run the smallest representative solve + quick regression checks.
   - If solver doesn’t work: run dry-run structural checks + tiny logic tests.
5. **Report**
   - Summarize what changed, where, and how it was verified.
   - If anything is unverified, state it clearly and suggest the smallest user-side command to verify.

---

## Boundaries / safety
- Do not delete files or rewrite large directories unless explicitly requested.
- Do not modify secrets, credentials, license files, or vendor blobs.
- Do not introduce network calls for installation or data pulls unless explicitly requested.

---

## What to do when requirements are unclear
Make the smallest reasonable assumption consistent with existing code.
If ambiguity affects correctness (e.g., model semantics, indexing, cost definition), ask a targeted question and propose two options with minimal diffs.

End of instructions.
