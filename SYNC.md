# Upstream synchronization

Source: `bn_edge_removal`, commit `f48c6a18a259975904937aea5ab0363c9b98d912`, obtained 2026-11-05.

Copied from that commit: `src/bn_edge_removal/`, `configs/`, the three dashboard files in `notebooks/`, `tests/bn_edge_removal/`, `tests/conftest.py`, `tests/fixtures/`, and the run, verification, tuning, and marimo UI scripts listed in the synchronization request.

Public-specific additions are the WASM loader in each dashboard, `scripts/portal_index.html`, and `.github/workflows/deploy.yml`. The public project also maintains its own dependency and tooling configuration.

To resynchronize, update the upstream SHA above, use `git archive` on that commit to extract the same paths into this repository, and reapply the WASM loading code to the first cell of each dashboard. Check dependencies, run the tests and browser checks, and review the result before publishing.
