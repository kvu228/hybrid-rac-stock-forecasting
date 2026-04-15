import sys
from pathlib import Path

# Ensure repo root is on sys.path for `etl.*` imports when running `pytest`
# without installing the project as a package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

