# Development record

This is the single record of how the current code base was developed, reviewed, fixed, and cleaned. On 2026-10-29 it replaced these files:

- `course_project_foundation_audit.md`, `independent_correctness_audit.md`, `independent_audit_fix_plan.md`;
- `codebase_refactor_development_plan.md`, `refactor_review.md`;
- `codebase_fix_plan.md`, `codebase_fix_implementation_notes.md`, `work_summary_sessions_1-3.md`.

Model and algorithm definitions are in `Formulation and Algorithm File.md`. Usage and test commands are in `README.md`.

## 1. Recovery point

The local branch `snapshot/pre-cleanup-2026-10-29` (commit `4459d7e`, parent `befd4db`) stores the full working tree as it was before the 2026-10-29 cleanup. This includes every file that was untracked at that time: all docs, all local datasets, `model/formulation_blocks.py`, `algo/runner.py`, the new tests, `logs/`, and every `output/` run. Only caches are excluded. The branch is local and has not been pushed.

To restore one path:

```bash
git checkout snapshot/pre-cleanup-2026-10-29 -- <path>
```

The working branch head `befd4db` still does not contain the untracked modules that the current code needs. Committing the current tree remains an open item (Section 10).

## 2. Provenance and AI attribution

The historical boundary is `396294f69f9041234ce850772ebba544226c1ffe`, the common ancestor of `dev` and `dev_cx`, as confirmed by the user. Five commits follow it:

| Commit | Date | Assessment |
|---|---|---|
| `7eadafb` | 2026-06-22 | Instruction-file edit. |
| `160c11b` | 2026-06-23 | Solver/configuration engineering. |
| `30616b5` | 2026-06-23 | Data refactor **and numerical behavior changes**: replaces the hardcoded big-M `1.5`, changes probability normalization, and changes scenario filtering. |
| `0d8ceab` | 2026-06-26 | Contracts, canonical subproblem, and cut-flow and insertion refactoring. It changes the algorithm implementation extensively; no historical golden test establishes behavioral equivalence. |
| `befd4db` | 2026-09-16 | Formulation/runner extraction checkpoint. It is incomplete as a standalone tracked snapshot. |

All five commits list the user's Git identity as author and committer, and none records AI co-authorship. Git therefore cannot attribute individual edits. Chats and summaries show substantial AI involvement, especially in the uncommitted P0–P9 work and in both audit rounds. The snapshot commit `4459d7e` carries a `Co-Authored-By: Claude` trailer.

All five cut families (Benders, strengthened Benders, Lagrangian, CGLP, integer optimality) already exist in `396294f`. The later work kept these families but changed parts of their implementation and of the model:

- P1 anchors cuts on solver dual bounds instead of MILP incumbents.
- P2 changes lower-bound computation and convergence under guidance weights.
- P3 weights grouped subproblem objectives and cut left-hand sides by unconditional probabilities.
- P5 adds $p^{G} \ge 0$.
- P6 extends aggregated dispatch to CGLP and integer optimality cuts.
- P8 introduced one run-level big-M; B-F4 later replaced it with bounds derived from line and voltage limits.
- P9 adds candidate generation infrastructure. The default hook does no work.
- B-F9 normalizes out-of-sample scenario weights.

A description consistent with this evidence:

> The project builds on my pre-existing MILP and decomposition code at revision `396294f`. Subsequent development used AI assistance for refactoring, solver configuration, testing, and correctness fixes. The existing cut families were retained. The maintenance also changed cut anchoring, scenario weighting, bound reporting, some variable and big-M bounds, and the out-of-sample weighting. These changes are documented separately from the new course-project contributions.

Whether this meets the course's disclosure rules depends on those rules; they were not evaluated here. Avoid the phrase "only code engineering without major changes on algorithm improvement or implementation".

## 3. Timeline

| Stage | Date | Content | Result |
|---|---|---|---|
| Refactor workstreams W1–W8 | 2026-06 | W1 verification and solver foundation; W2 data-processing cleanup; W3 algorithm data contracts; W4 canonical subproblem path; W5 cut-generation registry; W6 shared formulation blocks; W7 runner and experiment cleanup; W8 heuristic extension point | W1–W4 and W6 done; W5, W7, and W8 completed by P6, P7, and P9 |
| Refactor review | 2026-09-16 | Correctness risks R3.1–R3.10 | Addressed by P0–P9 |
| Remediation P0–P9 | 2026-09-17/18 | Harness, solver semantics, bounds, weighting, cleanup | Done |
| Audit A | 2026-10-28 | Foundation audit, findings A-F1–A-F6 | All fixed |
| Audit B | 2026-10-29 | Independent correctness audit, B-F1–B-F9, B-R1–B-R4, B-G1 | All fixed except B-F2 (deferred) |
| Static cross-check | 2026-10-29 | Both audits checked against the code | Fixes present; no rerun |
| Cleanup | 2026-10-29 | Docs merged; data, scripts, and runs retired (Section 9) | Snapshot first |

## 4. ID glossary

Code comments and test names cite these IDs.

### Risks from the refactor review (R3.x)

| ID | Risk | Resolution |
|---|---|---|
| R3.1 | Lagrangian / strengthened-Benders inner MILP was truncated and the cut read the incumbent, an upper bound on $L(\lambda)$, so a cut could be invalid | P1: anchor on the dual bound |
| R3.2 | The default MIP gap reached the exact subproblem, which anchors integer-optimality and CGLP cuts | P1: dual-bound anchors; exact role gap 0 |
| R3.3 | A guidance weight $w>0$ invalidated the reported lower bound | P2: `bound_mode` |
| R3.4 | Inconsistent probability convention for non-singleton scenario groups | P3: weighted-theta convention |
| R3.5 | Inner minimization omits the main-problem DG rows | Intentional relaxation (valid, possibly weaker) |
| R3.6 | $p^{G}$ had no lower bound | Defect; $p^{G}\ge 0$ |
| R3.7 | The legacy decomposition stack was broken | Removed (P4) |
| R3.8 | `ModelBase` solve semantics (ignored options, fabricated values) | P1 |
| R3.9 | Branch/cut report over-reported additions | P7: `CutOutcome` |
| R3.10 | Minor hygiene (dual lookup, objective name, row removal, duplicate names, LP export guard, duplicated builders) | P8 |

### Remediation phases (P0–P9)

| ID | Content |
|---|---|
| P0 | Baseline harness B1–B6. B1 reference instance; B2 extensive-form ground truth $z^*$ with $LB \le z^*$; B3 cut validity against exact recourse over all enumerated designs; B4 lower-bound monotonicity; B5 pytest with a `requires_solver` marker; B6 golden-file regression (no golden file was ever committed) |
| P1 | Role-scoped `SolverConfigSet` (master, subproblem_lp, subproblem_exact, inner_min, lagrangian_master, cglp); `get_obj_bound()`; dual-bound cut anchors; `ModelSolveError` on unusable solves |
| P2 | `bound_mode` = `resolve_at_zero_weight` or `heuristic` for $w>0$ |
| P3 | Keep $\sum_s P_s\theta_s$ in the master; weight subproblem objectives and cut left-hand sides by $P_s$ |
| P4 | Remove `algo/TwoStageDecomposition.py`; `main.py decomposition_algo` uses the live runner |
| P5 | Formulation decisions (R3.5, R3.6) |
| P6 | `CutRequest` / `GeneratedCut` registry; one `run_iteration_cuts` driver; aggregation for all families |
| P7 | `DecompositionRunner`, `PhasePolicy`, `CutOutcome` |
| P8 | Unique named rows; real row removal; shared run-level big-M (later replaced by B-F4); output guards |
| P9 | `SolutionGenerator` hook (`NoOpSolutionGenerator` default, `StaticSolutionGenerator`) |

### Decisions

| Question | Decision |
|---|---|
| Negative $p^{G}$ (R3.6) | Defect: $p^{G} \ge 0$; $q^{G}$ stays free |
| Inner minimization without the DG rows (R3.5) | Intentional relaxation |
| Theta convention (P3) | $\sum_s P_s\theta_s$ in the master; weighted cut left-hand sides |
| Bound under $w>0$ (P2) | Configurable `bound_mode` |
| Scope | Non-singleton scenario groups are supported and covered by the harness |
| Thermal polygon | Intentional (user, 2026-10-29) |
| Reactive capability lower bound (B-F2) | Deferred by the user |

### Behavior changes users notice

- `ModelBase.solve()` raises `ModelSolveError` when a solve gives no usable solution. Pass `allow_failure=True` where failure is expected.
- A subproblem LP relaxation that is not proven optimal raises an error instead of producing a Benders cut.
- `main_model.lp`, the plot, run parameters, and `best_incumbent.json` are written only with `enable_result_output=True`.
- Pretrained restrictions are opt-in; while they are active, no global lower bound is reported.

## 5. Audit A (2026-10-28): findings A-F1 to A-F6

| ID | Finding | Fix |
|---|---|---|
| A-F1 | Pretrained restrictions could certify a bound for the original problem | Opt-in only; no global lower bound while active |
| A-F2 | The runner accepted incomplete or duplicate scenario partitions | `DecompositionRunner` validates an exact, duplicate-free partition |
| A-F3 | Bound validity followed the current weight, not the last solved objective | Any objective change discards the stored results |
| A-F4 | An unproven LP was accepted for Benders cuts | Only `SOLVED_OPT` is accepted |
| A-F5 | Generated candidates with theta crashed the strengthened-cut path; the binary check was weak | Inner model reads only the canonical design blocks; candidates are validated (keys, finiteness, bounds, binary) |
| A-F6 | The harness could print PASS without its criteria | Explicit convergence check; UNCHECKED and missing golden files fail; exact oracle; plot separates certified bounds from diagnostics |

The regression tests are in `foundation_regression_test.py`. Audit A also listed documentation mismatches; they were reconciled in `Formulation and Algorithm File.md`:

- First-time expectation has no division by $|T|$, which matches the code.
- The Lagrangian stopping rule is described as heuristic.
- CGLP separation runs in the outer iterations.
- The deterministic Lagrangian model was removed.
- The scenario generator is external to this repository.

## 6. Audit B (2026-10-29): findings B-F1 to B-F9, B-R1 to B-R4, B-G1

These IDs are separate from audit A. Test names in `independent_audit_regression_test.py` use them in lower case (for example `test_f1_…` means B-F1). B-F9 is tested in `out_sample_probability_test.py`.

| ID | Finding | Fix |
|---|---|---|
| B-F1 | Cut signatures omitted the theta side, so cuts of other scenario groups were dropped as duplicates | Signatures include the weighted theta coefficients |
| B-F2 | Reactive output has no lower capability bound; an uninstalled DG can absorb reactive power | **Deferred** by the user; behavior unchanged |
| B-F3 | The best objective was not paired with its design | Incumbent design and source are stored together; `best_incumbent.json`; the out-of-sample loader prefers it |
| B-F4 | The automatic big-M bounded only active demand | Bounds from the thermal polygon ($S$, $2S$) and $\overline{V}-\underline{V}$ |
| B-F5 | CGLP revisits kept the wrong equality anchor and stale point values | One row per point; strongest certified value kept; explicit tight point; violation checked at the anchor |
| B-F6 | The LP role did not replace the exact-subproblem solver configuration | Scoped LP-role configuration, restored afterwards |
| B-F7 | CSV gaps used the diagnostic master value | Gaps and the summary row use `valid_lower_bound` only |
| B-F8 | Unfixing variables kept a restricted-model certificate | Fix and recover discard results and restore the original state |
| B-F9 | Out-of-sample weights used raw CSV probabilities (they do not sum to one) | Weights normalized over the evaluated scenarios; design cost unscaled |
| B-R1–B-R4 | Redundant collapsed-model wrappers, unused initializer parameters, an unused CGLP helper, duplicated affine-cut builders | Removed / unified |
| B-G1 | Merged scenario groups were dropped from the per-group CSV | One populated row per group |

The heuristic Lagrangian stopping rule was an observation, not a finding, and is unchanged.

## 7. Verification status

- After the audit B fixes: **84 passed with Gurobi**; **81 passed and 3 quadratic-solver skips with HiGHS**; 9/9 extensive-form reference instances matched.
- B-F9: 10/10 cases passed with HiGHS, including an extensive-form comparison at total 2870.
- 2026-10-29 static cross-check: all claimed fixes are present in the code. It was a code reading only; no tests were rerun.
- Commands: `README.md`, "Regression checks".

## 8. Evidence for the Final Report

The Final Report's screenshots live outside the repository (`~/Library/Application Support/typora-user-images`, `~/drawio/SDinPS`). The run data behind them stays untracked in `output/`. The "Basis" column says how each mapping was established:

- **cited**: the report embeds this folder's plot;
- **exact**: the report's numbers appear in these files;
- **inferred**: run name, settings, and finish time precede the screenshot.

| Final Report item | `output/` folder | Supporting files | Basis |
|---|---|---|---|
| Cut families: classical Benders benchmark (3 similar); symmetry check (initial / perturbed data) | `20260210110218_longrun_symm`, `20260210104210_longrun_non_symm` | `iter_general_info.csv`, `iter_obj_plot.png`, `main_model.lp` | inferred |
| Strengthened Benders, 3 similar scenarios | `20260210173414_longr_ns_correct_sbd` | same | inferred |
| Strengthened Benders, 3 different; Lagrangian cuts in the last 25 iterations | `20260210202331_longr_ns_correct_sbd_lgc_135s_right` | same | inferred |
| Strengthened Benders, 6 scenarios | `20260210212238_longr_ns_correct_sbd_lgc_alls_right` | same | inferred |
| Long run, 3 similar scenarios | `20260221224357_longr_750_ns_correct_sbd` | `iter_obj_plot.png` | cited |
| Fixed-rated-power formulation: previous / current setting | previous: the 2026-02-10 runs above; current: `20260223182443_preview_s123_longrun` (similar), `20260223194331_preview_s135_longrun` (different) | plots, `iter_general_info.csv` | inferred |
| Separated vs aggregated cuts | similar: `20260223182443_…` (separated, cited), `20260224105511_preview_s123_longrun_aggr` (aggregated, inferred); different: `20260223194331_…` (separated, inferred), `20260224100835_preview_s135_longrun_aggr` (aggregated, cited) | plots, `decomp_run_parameters.txt` | cited / inferred |
| Pre-training based cuts: improved / benchmark | improved: `20260224164408_test` (s_1, s_3, s_5, about 17 min); benchmark: `20260223194331_…` | plots | inferred / cited |
| Cut repetition and redundancy | `20260323202527_test_all_cut_long` | `main_model.lp` (158 CGLP rows), `cut_repetition_report.json` | exact (158) |
| Tightness check near the reference solution | `20260324201807_dynamic_algorithm_w_ref` | `iter_general_info.csv` (rows `ref_0`, `ref_1`) | inferred |
| Out-of-sample validity table (792001, 791663, 1577885, 1025804) | `20260329192349_dynamic_algorithm_new_m` | `sample_objective_summary_ref/best/lastest/ite1.csv` | exact |
| Collapsing methods (791663 / 1285651 / 1135138) | `20260329192349_…` (`collapsed_by_time`), `20260329151400_dynamic_algorithm_new_m` (`collapsed_no_time`), `20260331192031_new_m_agg_t_n_pret` (`collapsed_by_scenarios`) | `sample_objective_summary*.csv`, `decomp_run_parameters.txt` | exact |
| Weight table, $w$ = 1 / 3 / 5 / 7 | `20260329150235_…` / `20260329143058_…` / `20260329143537_…` / `20260329144046_…` (all `_dynamic_algorithm_new_m`) | `sample_objective_summary*.csv` (OOS), `iter_general_info.csv` (lower bound, incumbent, gap) | exact |
| Weight table, reference row (402288; 3239346) | `20260329150235_…` (`sample_objective_summary_ref.csv`); `20260324201807_…` (`iter_general_info.csv`) | | exact |

Notes:

- **Historical out-of-sample values.** Every out-of-sample number above was computed before B-F9, with raw CSV probabilities. Their sum over the evaluated scenarios is 0.321 for s_1–s_6 and 0.159 for s_1, s_3, s_5. The current evaluator normalizes the weights, so a rerun divides these values by that sum. Rankings within each table do not change, but the values cannot be compared with in-sample objectives or with current evaluator output. The Final Report tables were not rewritten.
- **Weight table settings.** The $w=1$ run has 55 iterations with phase-3 weight 0; the $w=3,5,7$ runs have 30 iterations (`decomp_run_parameters.txt`).
- **The "368 duplicates" figure.** It (120 Benders, 173 strengthened, 75 Lagrangian) came from an AI-assisted reading of the LP, according to the 2026-03-23 progress notes. An exact-text recount of `20260323202527_…/main_model.lp` finds at most 230 identical rows (162 strengthened Benders, 66 Lagrangian, 2 Benders). Re-check before citing it.

## 9. Retired on 2026-10-29

All items below can be restored from the snapshot branch (Section 1).

| Item | Reason |
|---|---|
| The eight records listed at the top | Merged here |
| `Lagrangian Cut Generation.md`, `New Main Problem Model.md`, `new_main_problem_model_implementation_plan.md`, `scenario_generation_logic_report.md` | Merged into `Formulation and Algorithm File.md` |
| `CGLP_node_example.md` | Identical to the first part of `CGLP_test_example.md`; its source citation was moved there |
| `Report.md` | 2026-02 to 2026-03 progress notes, replaced by the Final Report. Unique notes kept: the pre-training "known optimal design" remark (formulation file) and the origin of the 368 figure (Section 8) |
| `language_req.md` | Moved into `AGENTS.md` ("Report writing style") |
| `unit_test/test_local_csv_file/function_test/`, `function_test_symm_broken/` | Inputs of the 2026-02 experiments: original 33-bus planning data with variable DG rating, and the copy with perturbed installation and hardening costs used for the symmetry check. Unlike `function_test_fixed_rated_p`, they have different `node.csv`, `branch.csv`, and `solution reference.csv` and no `dg_rated_power.csv`, so the current reader cannot load them |
| Sensitivity workflow: `sensitive_load/`, `sensitive_broken/` (including `origin/`), `sensitive_analysis_data_test.py`, `util/tools.py` functions `data_perturbation`, `_apply_load_perturbation`, `_apply_broken_perturbation` | Not used by the Final Report, unloadable (no `dg_rated_power.csv`), and generated with unseeded randomness. Recipe: copy `origin/` into `p_1 … p_k`. *Load:* for each scenario, scale P and Q of `node_num` random nodes (never `node_1`) by $U(1-\epsilon, 1+\epsilon)$; sets `10_1_0.1` and `10_3_0.1` (10 copies, 1 or 3 nodes, $\epsilon = 0.1$). *Broken:* for each scenario, swap the no-harden/harden state series of `line_num` random lines with a random adjacent line; set `2` (10 copies, 2 lines). The 2026-01-26/27 combined-model runs used these sets |
| `analysis/main_decision_varibility.py` | One-off analyzer: per-iteration difference ratio of the master design against the previous iteration and a reference design, from `main_result_repetition_report.pkl` |
| `scripts/write_iter_general_update.py` and `util/tools.py::write_iter_general_update_from_iter_general_path` | Re-processed old `iter_general_info.csv` files; current runs write the gap and summary row directly |
| `util/tools.py::load_warm_start` | Unused |
| `unit_test/test_local_csv_file/IEEE123bus/stochastic_preview/` | Generated preview images (`util/topo_visual.py` can regenerate them) |
| `output/`: every run not listed in Section 8, and the `decomp_validity_*.json` / `micro_bench_report.json` reports | Not cited; the tests regenerate the reports |
| `logs/` | 2026-03 run logs |

## 10. Open items

- B-F2 (reactive capability lower bound): a modeling decision is pending.
- Commit the current working tree deliberately on the working branch, with messages that separate refactoring from numerical fixes, and tag the tested baseline before course work. The snapshot branch is a recovery point, not a curated commit.
- No B6 golden file exists. A file created now records a new baseline; it cannot prove equivalence with the pre-refactor code.
