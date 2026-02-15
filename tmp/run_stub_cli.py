import os
import subprocess

env = dict(os.environ)
env["EIKON_MISSION_TEST_EXECUTOR"] = "eikon_engine.tests.missions.cli_stub:build_executor"
cmd = [
    "python",
    "-m",
    "eikon_engine.missions.cli",
    "--mission",
    "Demo guardrail success",
    "--artifacts-dir",
    "artifacts/random_stub",
]
subprocess.run(cmd, check=True, env=env)
