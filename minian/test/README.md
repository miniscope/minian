# Running unit tests

Minian uses pytest for unit testing.

From the repository root:

```bash
pdm run test
# or: pytest
```

That runs targeted tests under `minian/test/` and skips slow notebook tests by default.

To include notebook execution tests:

```bash
pdm run test-notebooks
# or: pytest --with-notebooks
```

CI runs notebooks in a separate job via `pytest --with-notebooks`.

# Update fixtures

Run the pipeline notebook on the demo data, then update assertions in `minian/test/test_pipeline.py` (and cross-reg test) as needed.
