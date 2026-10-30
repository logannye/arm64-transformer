# Human review and upstream submission handoff

Use this checklist to review the patches and handle the required upstream actions. It preserves the earlier FINAL checklist (archived `outputs/upstream-integration-v2/submission/HUMAN-CHECKLIST-FINAL.md`) and version 2 (archived `outputs/upstream-integration-v2/submission/HUMAN-CHECKLIST-V2.md`). The policy links below are the retained October 29 snapshot; refresh upstream rules, heads and duplicate fixes before acting.

Paths marked `archived` refer to the verified evidence layout, not this repository. Use the [archive and recipe table](REPRODUCTION.md#finding-the-archived-files) and the final release catalog after publication.

## Review the exact technical deliverables

| Candidate | Intended source |
|---|---|
| PyTorch deterministic MPS embedding backward | Formatted patch (archived `outputs/upstream-integration-v2/pytorch/formatted-v2/changes.patch`), SHA-256 `3834dca844ea3de6b40467ce214b99bce83fa06dae2bc454623516dafc0fdd71`, base `71d2a08507ab47dd6b2fa9b66319d17ce2377de4` |
| MLX deterministic embedding gradient | Embedding patch (archived `outputs/upstream-framework-fixes/mlx-embedding/embedding-fix.patch`), base `a21f892a6e8e83afb117cef1d1ceae3e82f48c32` |
| MLX attention dispatch/resource repair | Attention patch (archived `outputs/upstream-framework-fixes/mlx-attention/attention-fix.patch`), same MLX base |

- [ ] Read the complete code and regression tests. Understand deterministic ordering, precision, attention dispatch, shape coverage and performance costs. Passing tests do not replace the contributor's understanding.
- [ ] Review the completed MLX numerical evidence (archived `outputs/upstream-integration-v2/release-integration-v2/RELEASE-INTEGRATION.md`), operator results (archived `outputs/upstream-integration-v2/operator-results-v3/README.md`), and their exact source/runtime/audit bindings. These tests were already performed on the combined two-patch Release build; this checklist does not describe them as unfinished.
- [ ] Review the closed formatted-PyTorch audit (archived `work/upstream-integration-v2/pytorch/closed-five-family-audit-v2/AUDIT-SUMMARY.md`): four targeted families passed, while the full upstream module retained 25 failures and two errors across 10,353 tests. Overall development qualification remains false. The [reproduction guide](REPRODUCTION.md#source-and-runtime-integrity-follow-through) explains the separate selected runtime, matched control and preserved failure history. The independently audited paired replay records all 27 failures in both variants, with matching failure kinds, source-bound locations and all 81 state comparisons. Neither targeted passes nor this supplemental control clears the original suite or qualifies a supported release.
- [ ] Review the completed source-byte postflight and content-identity preflight in the [reproduction guide](REPRODUCTION.md#source-and-runtime-integrity-follow-through). Source-byte identity passed; both interrupted paired attempts remain incomplete and unqualified. The fresh paired replay closed with all 54 test executions failed, and its independent evidence audit passed. An integrity pass does not turn those failed tests into passes. Review original notice/archive inventories without substituting a new identity into an old receipt.
- [ ] Keep runtime patches separate from study adapters, transport, source-authentication, header packaging and report machinery. The optional MLX DLPack test-eligibility change is a separate test-only proposal, not part of either numerical repair.

## What must be repeated for submission

- [ ] Refresh upstream heads and duplicate work, choose the actual submission base, apply/rebase into isolated branches, and review the resulting complete diff. Record the submitted commit and patch identities.
- [ ] Rerun applicable checks whenever source, dependencies, compiler, runtime or target revision changes. The formatted PyTorch binary has its own evidence; the old unformatted binary is not a substitute.
- [ ] Validate the actual MLX patch combination to be submitted. Combined embedding-plus-attention evidence does not establish independent-patch coverage or performance. Keep patches separate, declare any dependency, and test each proposed submission state.
- [ ] Include required formatting/full tests and main-versus-PR benchmarks for performance-affecting changes, with scripts, commands and visible skips/failures. Review the completed checks first. Changed or separately submitted revisions need applicable validation in the authorized environment; the existing checks do not need repeating merely because an agent produced them.

## PyTorch public actions

- [ ] Follow the retained [PR lifecycle](https://github.com/pytorch/pytorch/blob/7c67fb02433dea09fa1de93b2224b6e0309e7906/CONTRIBUTING.md#pr-lifecycle): obtain an issue labeled `actionable` or a maintainer's prior agreement unless the current write-access exception applies. The earlier permission snapshot found no exception. Existing determinism discussions are context, not approval of this implementation. The retained history records [#195404](https://github.com/pytorch/pytorch/issues/195404) as closed by the nondeterminism alert in [#195754](https://github.com/pytorch/pytorch/pull/195754), not by a deterministic embedding implementation. Obtain and link an actually qualifying issue or explicit agreement for this patch; [general determinism support](https://github.com/pytorch/pytorch/issues/97236#issuecomment-1478422035) does not satisfy that requirement.
- [ ] Write the template's **“Summary (human written only)”** yourself and answer its backward-compatibility question. Under the [AI policy](https://github.com/pytorch/pytorch/blob/7c67fb02433dea09fa1de93b2224b6e0309e7906/AI_POLICY.md), any reused AI text must be clearly labeled/contained and accompanied by your explanation; do not paste unreviewed generated replies.
- [ ] Review and approve the exact content before agent-assisted issue/PR actions or replies, under the retained policy requiring that the **“user has reviewed and explicitly approved the exact content”**, in [CLAUDE.md](https://github.com/pytorch/pytorch/blob/7c67fb02433dea09fa1de93b2224b6e0309e7906/CLAUDE.md). The [template](https://github.com/pytorch/pytorch/blob/7c67fb02433dea09fa1de93b2224b6e0309e7906/.github/PULL_REQUEST_TEMPLATE.md) asks you to open the PR as a draft and mark it ready once the description and code are ready for review. Follow the selected upstream submission workflow rather than assuming a direct push is appropriate.

## MLX public actions

- [ ] Write your own descriptions and commit messages, and disclose AI usage. The retained [AGENTS.md](https://github.com/ml-explore/mlx/blob/a21f892a6e8e83afb117cef1d1ceae3e82f48c32/AGENTS.md) forbids agents from writing these or answering reviewer comments; the [template](https://github.com/ml-explore/mlx/blob/a21f892a6e8e83afb117cef1d1ceae3e82f48c32/.github/pull_request_template.md) says it is **“strictly prohibited to use AI to write PR description”**.
- [ ] Personally push submission branches, open PRs and answer reviewers; AGENTS.md states **“Do NOT run `git push` or create a PR on behalf of the user”** and **“Do NOT respond to a comment”** on the user's behalf. These requirements apply to upstream submission, not this project's authorized evidence publication.
- [ ] Consider an issue for larger changes, as [CONTRIBUTING.md](https://github.com/ml-explore/mlx/blob/a21f892a6e8e83afb117cef1d1ceae3e82f48c32/CONTRIBUTING.md) recommends. This is not a universal prerequisite for every bug fix. Include applicable test and main-versus-PR performance evidence. Await required maintainer approval for upstream CI; an unstarted job is not a pass.

## Keep the study boundary intact

An accepted source patch, a passing private build and an eligible supported stable package are different milestones. Patched integration does not admit either standard baseline. Requalify eligible installed packages within the agreed update allowance; if PyTorch changes, refresh CPU and MPS with the same selected version and revalidate CPU anchors. The full five-engine study and checkpoint audits still require their unchanged gates. Formal fits remain **0/180**, with stream **20261104 unopened**.

Use the final catalog for own-project release `428513598` / tag `transformer-upstream-integration-v2` to verify its publication state, final source revision and complete asset inventory. Those identities are separate publication bookkeeping. See the companion [reproduction guide](REPRODUCTION.md) for evidence layers, archived recipes and verification instructions.
