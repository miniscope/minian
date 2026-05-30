# Releasing MiniAn to PyPI

MiniAn is published to PyPI via GitHub Actions using
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC),
so no API tokens are stored in this repo.

The package version is derived from the **git tag** by `pdm-backend`'s SCM
source. There is no `version = "..."` field to keep in sync — the version
that ends up on PyPI is whatever you tag.

## First PyPI release

The first PyPI release is planned as `v2.0.0`, even though older `v1.x.x`
tags exist in the repo history. Those legacy tags were never published to
PyPI — they're only on conda-forge — so PyPI's version history starts fresh
at v2. SCM-derived dev builds between tags will read off the latest tag
present in the repo (e.g. builds today resolve as `1.2.1.dev<n>+g<sha>` until
`v2.0.0` is cut).

## One-time PyPI setup

On https://pypi.org, register `miniscope/minian` as a trusted publisher for
the `minian` project (currently pending the [PEP 541][pep541] transfer):

[pep541]: https://peps.python.org/pep-0541/

- **PyPI Project Name:** `minian`
- **Owner:** `miniscope`
- **Repository name:** `minian`
- **Workflow name:** `publish.yml`
- **Environment name:** `pypi`

Then in this repo's GitHub settings, create an environment named `pypi`
(optionally with required reviewers for an extra approval gate; recommended:
restrict deployment to `v*` tags).

If `minian` is not yet claimed on PyPI, do the initial release with a
[pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
instead — same fields, but PyPI creates the project on first successful run.

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

If you need to override the version manually (e.g., the first v2 release
that jumps over inferred versions), tag directly:

```bash
git tag -a v2.0.0 -m "Release v2.0.0"
git push origin v2.0.0
```

The `Publish to PyPI` workflow runs automatically on `v*` tag pushes. It
builds an sdist + a pure-Python wheel (version derived from the tag via
SCM), runs `twine check`, and uploads to PyPI.

## Manual dry-run

Running the workflow via `workflow_dispatch` builds and uploads the artifacts
to the run summary but skips the publish step — useful for validating
metadata changes before tagging. The resulting wheel/sdist will have a
PEP 440 dev version (e.g. `1.2.1.dev41+g<sha>`).
