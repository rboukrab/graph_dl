"""Backward-compatible wrapper around run_gl.py for the GLAMA algorithm."""

import subprocess
import sys

if __name__ == "__main__":
    sys.argv[1:1] = ["--algorithm", "glama"]
    subprocess.run([sys.executable, "run_gl.py", *sys.argv[1:]], check=False)
