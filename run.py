"""Start the dashboard: run this file (e.g. PyCharm's Run button) and it opens in your browser."""

import sys
from pathlib import Path

from streamlit.web import cli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py"))]
    sys.exit(cli.main())
