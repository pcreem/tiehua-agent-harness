#!/usr/bin/env python3
"""
Tiehua Tourism Agent - Evaluation Dashboard
Hugging Face Space Entry Point
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Check required environment variables
if not os.getenv("OPENROUTER_API_KEY"):
    print("WARNING: OPENROUTER_API_KEY not set. LLM Judge will not work.")
    print("Set OPENROUTER_API_KEY in Space settings.")

# Run Streamlit app
if __name__ == "__main__":
    import subprocess
    import sys
    
    # Run streamlit app
    result = subprocess.run([
        sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ])
    sys.exit(result.returncode)