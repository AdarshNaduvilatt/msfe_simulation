import streamlit.web.cli as stcli
import sys
from pathlib import Path

app = Path(__file__).parent / "app.py"

sys.argv = [
    "streamlit",
    "run",
    str(app),
]

stcli.main()