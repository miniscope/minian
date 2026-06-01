# Running unit tests

Minian uses pytest for unit testing.

From the repository root:

```bash
pdm run test
# or: pytest
```

That runs only the fast tests under `minian/test/` (preprocessing utilities, etc.).

Slow end-to-end notebook checks live in `scripts/` and are **not** collected by pytest:

```bash
pdm run test-notebooks
# or individually:
python scripts/run_pipeline_notebook_check.py
python scripts/run_cross_reg_notebook_check.py
```

CI runs notebooks in the separate **notebook tests** workflow.

# Update fixtures

Run the pipeline notebook on the demo data, then update assertions in `scripts/run_pipeline_notebook_check.py` (and cross-reg script) as needed.