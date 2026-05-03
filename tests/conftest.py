import os
import sys
import pathlib

# put project root on sys.path so tests can import birefnet_api/utils/etc. without install
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Avoid loading the project's Config() at import time when tests are collected;
# config.py reads train.sh and probes /workspace/datasets which doesn't exist
# in CI. Tests that need the model will set this env var to allow it.
os.environ.setdefault("BIREFNET_TESTS_NO_CONFIG", "1")
