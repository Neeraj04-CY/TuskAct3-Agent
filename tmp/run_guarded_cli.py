import json
import subprocess

cmd = [
    "python",
    "-m",
    "eikon_engine.missions.cli",
    "--mission",
    "...goal...",
    "--budget-max-steps",
    "10",
    "--safety-contract",
    json.dumps({"blocked_actions": ["download_file"]}),
    "--ask-on-uncertainty",
]

subprocess.run(cmd, check=True)
