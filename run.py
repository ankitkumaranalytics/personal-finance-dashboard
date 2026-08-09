"""Entry point to launch the Personal Finance Dashboard."""

import subprocess
import sys


def main():
    """Run the Streamlit dashboard."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    main()
