# Releasing MiniAn to PyPI

MiniAn is published to PyPI via GitHub Actions using
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC),
so no API tokens are stored in this repo.

The package version is derived from the **git tag** by `pdm-backend`'s SCM
source. There is no `version = "..."` field to keep in sync — the version
that ends up on PyPI is whatever you tag.

## First PyPI release

Legacy `v1.x.x` tags exist in the repo history but were never published to
PyPI (they're only on conda-forge). The first PyPI version number hasn't
been decided — bumping to `v2.0.0` is one option, continuing from the
latest `v1.x` tag is another. SCM-derived dev builds between tags read
off the latest tag present in the repo (e.g. builds today resolve as
`1.2.1.dev<n>+g<sha>`).

## One-time PyPI setup

On https://pypi.org, register `miniscope/minian` as a trusted publisher for
the `minian` project:

- **PyPI Project Name:** `minian`
- **Owner:** `miniscope`
- **Repository name:** `minian`
- **Workflow name:** `publish.yml`
- **Environment name:** `pypi`

If the project doesn't exist on PyPI yet, use a
[pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) —
PyPI creates the project on the first successful publish.

Then in this repo's GitHub settings, create an environment named `pypi`
(optionally with required reviewers for an extra approval gate; recommended:
restrict deployment to `v*` tags).

## Cutting a release

From a clean `master` (or release branch):

```bash
# Auto-bump from conventional commits, write the changelog, and create
# the tag. `version_provider = "scm"` means cz reads the current version
# from git and does NOT touch source files.
pdm run cz bump

# Push the commit and the tag.
git push origin master --follow-tags
```

If you need to override the version manually (e.g., to jump over
inferred versions for the first PyPI release), tag directly:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

The `Publish to PyPI` workflow runs automatically on `v*` tag pushes. It
builds an sdist + a pure-Python wheel (version derived from the tag via
SCM), runs `twine check`, and uploads to PyPI.

## Manual dry-run

Running the workflow via `workflow_dispatch` builds and uploads the artifacts
to the run summary but skips the publish step — useful for validating
metadata changes before tagging. The resulting wheel/sdist will have a
PEP 440 dev version (e.g. `1.2.1.dev41+g<sha>`).
