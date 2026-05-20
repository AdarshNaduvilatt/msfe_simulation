"""Command-line helpers for launching the MSFE Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit application installed with the package."""
    from streamlit.web import cli as stcli

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
