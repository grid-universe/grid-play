import os
import sys
from streamlit.web import cli as stcli

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

def main():
    sys.argv = ["streamlit", "run", os.path.join(SCRIPT_PATH, "main.py")]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()