import sys
from pathlib import Path

# Ensure root repository directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Launch unified Streamlit application
import streamlit_app  # noqa: F401
