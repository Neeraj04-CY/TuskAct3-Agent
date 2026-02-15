from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from eikon_engine.config_loader import load_settings
from eikon_engine.approval.models import ApprovalRequest, ApprovalState, UTC


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve pending approval requests")
    sub = parser.add_subparsers(dest="command", required=True)

    approve = sub.add_parser("approve", help="Approve an approval request")
    approve.add_argument("approval_id", help="Approval id to approve")
    approve.add_argument("--reason", default="approved_by_cli", help="Reason for approval")

    reject = sub.add_parser("reject", help="Reject an approval request")
    reject.add_argument("approval_id", help="Approval id to reject")
    reject.add_argument("--reason", default="rejected_by_cli", help="Reason for rejection")

    return parser.parse_args()


def _artifact_root() -> Path:
    settings = load_settings()
    logging_cfg = settings.get("logging", {}) if isinstance(settings, dict) else {}
    root = logging_cfg.get("artifact_root") or "artifacts"
    return Path(root).resolve()


def _locate_request(approval_id: str, root: Path) -> Optional[Path]:
    for path in root.rglob("approval_request.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if str(payload.get("approval_id")) == approval_id:
                return path
        except Exception:
            continue
    return None


def _update(path: Path, state: ApprovalState, reason: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = ApprovalRequest.from_payload(payload)
    request.state = state
    request.resolution_reason = reason
    request.resolved_at = datetime.now(UTC)
    request.resolved_by = "cli"
    request.approved_by_human = state == "approved"
    path.write_text(json.dumps(request.to_payload(), indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    root = _artifact_root()
    path = _locate_request(args.approval_id, root)
    if not path:
        print(f"Approval id {args.approval_id} not found under {root}")
        return 1
    if args.command == "approve":
        _update(path, "approved", args.reason)
        print(f"Approval {args.approval_id} approved")
    elif args.command == "reject":
        _update(path, "rejected", args.reason)
        print(f"Approval {args.approval_id} rejected")
    else:
        print("Unknown command")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
