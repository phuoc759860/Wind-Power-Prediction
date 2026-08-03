"""Model leakage audit (thin re-export of app/validation/leakage_audit.py).

Reviewer Step 12: the leakage audit is reachable under app/evaluation/. The
implementation lives in app/validation/leakage_audit.py; this module only
re-exports it so all existing imports keep working.
"""

from __future__ import annotations

from app.validation.leakage_audit import (  # noqa: F401
    audit_model,
    build_full_leakage_audit,
)

__all__ = ["audit_model", "build_full_leakage_audit"]
