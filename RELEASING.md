# Releasing MiniAn to PyPI

MiniAn is published to PyPI via GitHub Actions using
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC),
so no API tokens are stored in this repo.

## One-time PyPI setup

On https://pypi.org, register `miniscope/minian` as a trusted publisher for
the `minian` project:

- **PyPI Project Name:** `minian`
- **Owner:** `miniscope`
- **Repository name:** `minian`
- **Workflow name:** `publish.yml`
- **Environment name:** `pypi`

Then in this repo's GitHub settings, create an environment named `pypi`
(optionally with required reviewers for an extra approval gate).

If `minian` is not yet claimed on PyPI, do the initial release with a
[pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
instead — same fields, but PyPI creates the project on first successful run.

## Cutting a release

1. From `master`, bump the version with commitizen — this updates
   `pyproject.toml`, `minian/__init__.py`, `minian/install.py`, the
   changelog, and creates a `vX.Y.Z` tag.

   ```bash
   pdm run cz bump
   ```

2. Push the commit and the tag.

   ```bash
   git push origin master --follow-tags
   ```

3. The `Publish to PyPI` workflow runs automatically on the tag push. It
   verifies the tag matches `pyproject.toml`, builds an sdist + a pure-Python
   wheel, runs `twine check`, and uploads to PyPI.

## Manual dry-run

Running the workflow via `workflow_dispatch` builds and uploads the artifacts
to the run summary but skips the publish step — useful for validating
metadata changes before tagging.
