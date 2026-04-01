# Runtime Boundary

This directory is reserved for generated runtime output.

Target contents:

- artifacts
- metrics
- reports
- local state
- temporary databases

Rule:

- new generated output should prefer this area over product source trees
- only minimal regression fixtures should stay in Git
