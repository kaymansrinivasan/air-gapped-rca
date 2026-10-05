# Jetson LlamaIndex changes awaiting publication approval

Target public repository: kaymansrinivasan/air-gapped-rca
Proposed branch: feature/llamaindex-jetson-ui
Base GitHub commit: fd8b3e6828c928e3ef4444d1ae13acf4435e803b

No GitHub commit or branch was created. Automatic approval review blocked
creating a public commit because approval for the exact code and environment
details was required. This package is for review. Do not treat it as a complete
checkout or as an on-board validated release.

## Proposed milestones

1. Track demonstrated Jetson LlamaIndex retrieval and integration status.
2. Add a local browser investigation form, evidence-ID selection, source-owned
   text/citations, input checks and fail-closed output handling.

## Validation performed

- 14 standard-library tests passed, including actual scenario-source linkage,
  the rejected draft, source-type/ID checks, source tampering, current/historical
  DUT separation, unrelated-pin choices and HTTP form handling.
- Python compilation and JavaScript syntax checks passed.
- No GPU inference, new prompt run, diagnostic benchmark or physical offline
  test was performed in this workspace. Those require the user's Jetson.
- Model selections can still be semantically wrong. Membership checks are
  not causal confirmation; the system must retain uncertainty and undergo
  engineer review and evaluation.

## Publication contents

The package contains source code, tests, dependency declarations and docs.
Documentation includes observed test results and default Jetson filesystem
paths/user name. No model weights, credentials, virtual environment, Chroma
DB, generated device log collection or private runtime logs are included.
Existing public synthetic case records are read by tests but are not modified.

Exact changed paths:

- .gitignore
- README.md
- docs/jetson-progress.md
- docs/local-app.md
- notes/decisions.md
- requirements-jetson.txt
- src/rca_local/__init__.py
- src/rca_local/__main__.py
- src/rca_local/core.py
- src/rca_local/index.html
- src/rca_local/pipeline.py
- src/rca_local/web.py
- src/rca_local/workers.py
- src/retrieve_case_evidence_llamaindex.py
- tests/test_rca_local.py
- tests/test_rca_web.py
