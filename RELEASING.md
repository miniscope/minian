# Releasing MiniAn to PyPI

MiniAn is published to PyPI by the `Publish to PyPI` GitHub Actions workflow
(`.github/workflows/publish.yml`) using
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC),
so no API tokens are stored in this repo.

The package version is derived from the **git tag** by `pdm-backend`'s SCM
source (see `[tool.pdm.version]` in `pyproject.toml`). There is no
`version = "..."` field to keep in sync: the version that ends up on PyPI is
whatever you tag. Builds between tags get a PEP 440 dev version like
`1.3.0.dev41+g<sha>`.

## Cutting a release

From a clean `master` (or a release branch):

```bash
# Auto-bump from conventional commits, write the changelog, and create the
# tag. `version_provider = "scm"` means cz reads the current version from git
# and does NOT touch source files.
pdm run cz bump

# Push the commit and the tag; the v* tag push triggers the publish workflow.
git push origin master --follow-tags
```

To set the version manually instead (e.g. to jump over inferred versions),
tag directly:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

On a `v*` tag push the workflow builds an sdist + pure-Python wheel (version
derived from the tag), runs `twine check`, and uploads to PyPI.

## Dry run

Trigger the workflow via `workflow_dispatch` (Actions tab -> Publish to PyPI ->
Run workflow) to build the artifacts and upload them to the run summary
**without publishing**. Useful for validating metadata changes before tagging.
The resulting wheel/sdist carry a PEP 440 dev version.
