# Standalone distribution

This directory implements the standalone fork requested by the project owner.
Preserve upstream agent, gateway, persistence and authorization semantics.

- Build tools, bootstrap and tests in this directory must run with the Python
  standard library alone. Do not install dependencies through pip, uv or PM.
- For this directory, run `python3 -I -S standalone/check.py`. It is the
  standalone distribution's test runner; the upstream pytest runner remains
  applicable to changes in upstream runtime modules.
- A packaging or diagnostic milestone is not a working Hermes agent. Keep
  capability reports and documentation explicit about what has been tested.
- Do not access the user's Hermes configuration, credentials or database in
  tests. Use temporary directories and deterministic local fixtures.
- Ship source, resources, distribution metadata and license notices. Reject
  third-party executable and native extension payloads.
- Keep dependency versions and download hashes pinned. Build-time fetching is
  explicit; runtime commands must never install or download dependencies.
- Keep generated artifacts and downloaded build inputs out of Git.
