import json
from pathlib import Path

_dir = Path(__file__).parent

with open(_dir / "role_based_pro_social.json") as f:
    role_based_pro_social = json.load(f)

with open(_dir / "instructed_self_serving.json") as f:
    instructed_self_serving = json.load(f)

with open(_dir / "instrumental_self_serving.json") as f:
    instrumental_self_serving = json.load(f)

DATASET = role_based_pro_social + instructed_self_serving + instrumental_self_serving
