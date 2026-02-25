from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


def append_jsonl(path: str, obj: Any) -> Optional[str]:
  if not path:
    return None
  try:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = obj
    if isinstance(obj, dict) and 'ts' not in obj:
      payload = {**obj, 'ts': time.time()}
    with target.open('a', encoding='utf-8') as f:
      f.write(json.dumps(payload, ensure_ascii=False) + '\n')
    return str(target)
  except Exception:
    return None

