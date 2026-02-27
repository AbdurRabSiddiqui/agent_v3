from __future__ import annotations

import json
import time
import hashlib
from pathlib import Path
from typing import Any, Optional


def prompt_fingerprint(text: str, *, preview_chars: int = 160) -> dict:
  raw = (text or '').strip()
  preview = raw[:max(int(preview_chars), 0)]
  sha = hashlib.sha256(raw.encode('utf-8', errors='ignore')).hexdigest()
  return {
    'prompt_preview': preview,
    'prompt_sha256': sha
  }


def write_json(path: str, obj: Any) -> Optional[str]:
  """
  Write a small JSON file atomically (best-effort).
  Used for live power-label sharing between processes.
  """
  if not path:
    return None
  try:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
    tmp.replace(target)
    return str(target)
  except Exception:
    return None


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

