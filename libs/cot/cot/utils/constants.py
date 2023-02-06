import json
import pkgutil
from types import SimpleNamespace
from typing import Dict


def _load_json_licenses() -> Dict[str, str]:
    """
    Load all licenses from JSON file.
    Amend names to be valid variable names
    """
    licenses = {k: v for k, v in json.loads(pkgutil.get_data(__name__, "licenses.json")).items()}

    return licenses


Licenses = SimpleNamespace(**_load_json_licenses())
