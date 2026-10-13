"""
Colab-compatible training launcher.
Run from repo root: !python train_colab.py
This avoids the 'No module named src' error when using python -m src.train
"""
import sys
import os

# Ensure the repo root is on sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Also set working directory to repo root
os.chdir(repo_root)

# Now import and run train main
from src.train import main
main()
