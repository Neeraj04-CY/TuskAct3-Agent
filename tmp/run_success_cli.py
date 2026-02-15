import json
import subprocess

cmd = [
    "python",
    "-m",
    "eikon_engine.missions.cli",
    "--mission",
    "Navigate to https://example.com and report success",
    "--artifacts-dir",
    "artifacts/random_success",
    "--budget-max-steps",
    "20",
    "--autonomy-budget",
    json.dumps({"max_risk_score": 0.9}),
]

subprocess.run(cmd, check=True)
