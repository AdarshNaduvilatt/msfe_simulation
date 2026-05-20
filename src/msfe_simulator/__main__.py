def main():
    import sys
    from pathlib import Path
    from streamlit.web import cli as stcli

    app_path = Path(__file__).parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    stcli.main()