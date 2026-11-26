# Release checklist

1. Confirm project ownership, author metadata, MIT attribution and intended repository URL. Add project URLs to pyproject.toml after the GitHub repository exists.
2. Run `python -m pytest -v` and `python examples/regime_demo.py`.
3. Run `python -m build` and `python -m twine check --strict dist/*` in a clean working tree.
4. Push to a separate GitHub repository and verify CI on Windows and Linux.
5. Configure a project-specific pending Trusted Publisher on TestPyPI and a matching GitHub environment, with deployment limited to main. Use workflow_dispatch and separate build and publishing jobs, with id-token: write only on the publishing job. Verify the package in a clean environment, installing NumPy from official PyPI first and OnsetKit from TestPyPI with --no-deps.
6. Configure a separate official PyPI publisher and environment. Publish only after tests and metadata review. Verify normal pip installation in another clean environment.
7. Tag the published commit and document the release. Do not upload the same version twice. Do not commit credentials.
