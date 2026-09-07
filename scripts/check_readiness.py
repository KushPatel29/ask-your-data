"""Read-only release preflight for this checkout, not a serving-process health endpoint.

    python scripts/check_readiness.py
    python scripts/check_readiness.py --production
    python scripts/check_readiness.py --voice

The voice flag can download the approved models into their normal cache. No
questions, SQL, credentials, result cells, or audio are written to the report.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import demo_mode
from engine.access import AccessScope
from engine.deployment import validate_deployment
from engine.semantics import Layer
from engine.warehouse import build_warehouse, table_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--voice", action="store_true")
    args = parser.parse_args()
    env = dict(os.environ)
    if args.production:
        env["ASK_DEPLOYMENT_MODE"] = "production"
    report = {"target": "checkout preflight", "checks": [], "ok": False}
    con = None
    try:
        mode = validate_deployment(env)
        report["checks"].append({"check": "configuration", "status": "passed", "mode": mode})
        con = build_warehouse()
        layer = Layer(con)
        if not layer.tables:
            raise ValueError("semantic layer is empty")
        report["checks"].append({"check": "warehouse and semantic layer", "status": "passed",
                                 "tables": len(table_names(con))})
        cases = demo_mode.load_golden_questions()
        scope = AccessScope.demo("release-preflight")
        passed = sum(demo_mode.answer(con, case, access=scope).matches_contract for case in cases)
        report["checks"].append({"check": "reference queries", "passed": passed,
                                 "total": len(cases), "status": "passed"
                                 if passed == len(cases) else "failed"})
        if passed != len(cases):
            raise ValueError("reference query contract drift")
        if args.voice:
            from engine.local_voice import LocalVoice

            client = LocalVoice()
            audio = client.synthesize("There are 876 denied claims.")
            transcript = client.transcribe(audio.audio, language="en")
            words = transcript.text.lower()
            if "876" not in words or "claims" not in words:
                raise ValueError("speech roundtrip did not preserve the test fact")
            report["checks"].append({"check": "male speech roundtrip", "status": "passed",
                                     "audio_bytes": len(audio.audio)})
        report["ok"] = True
    except Exception as exc:
        report["checks"].append({"check": "preflight", "status": "failed",
                                 "error_type": type(exc).__name__})
    finally:
        if con is not None:
            con.close()
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
