"""
Run this file to launch the DRSA interface.
Usage: python run_app.py
"""
import subprocess
import sys
import os

app_path = os.path.join(os.path.dirname(__file__), "drsa", "interface", "app.py")
subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
