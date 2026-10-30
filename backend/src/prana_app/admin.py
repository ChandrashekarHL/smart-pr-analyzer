"""Operator commands.

    python -m prana_app.admin failed            # list failed runs (the dead letters)
    python -m prana_app.admin replay <run_id>   # reopen a failed run and enqueue it
    python -m prana_app.admin init-db           # create tables (also done at start-up)
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("failed")
    r = sub.add_parser("replay")
    r.add_argument("run_id", type=int)
    sub.add_parser("init-db")
    args = ap.parse_args()

    from prana_app.services import services
    svc = services()
    if args.cmd == "init-db":
        svc.store.create_all()
        print("ok")
    elif args.cmd == "failed":
        rows = [r for r in svc.store.recent_runs(500) if r["status"] == "failed"]
        print(json.dumps(rows, indent=2, default=str))
    elif args.cmd == "replay":
        run = svc.store.get_run(args.run_id)
        if run is None:
            raise SystemExit(f"run {args.run_id} not found")
        if run.status != "failed":
            raise SystemExit(f"run {args.run_id} is {run.status}, not failed")
        rid = svc.store.open_run(run.installation_id, run.repo, run.pr_number, run.head_sha)
        from prana_app.tasks import enqueue
        enqueue(rid)
        print(f"run {rid} re-enqueued")


if __name__ == "__main__":
    main()
