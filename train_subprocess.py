"""
Subprocess script for running AFLCP training.
Called by backend/main.py to avoid TensorFlow threading issues.
"""
import sys
import json

if __name__ == "__main__":
    config_json = sys.argv[1]
    config = json.loads(config_json)

    from aflcp_core import train_aflcp
    train_aflcp(config)
