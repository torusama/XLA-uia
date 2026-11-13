import subprocess
import sys


def test_help_and_doctor():
    for command in ("--help", "doctor"):
        result = subprocess.run([sys.executable, "-m", "xla", command], capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
