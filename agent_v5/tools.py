from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool

from rate_tools import RATE_TOOLS


def _repo_root() -> Path:
  return Path(__file__).resolve().parent


def _resolve_in_repo(user_path: str) -> Path:
  if not user_path or not isinstance(user_path, str):
    raise ValueError('path must be a non-empty string')

  root = _repo_root()
  candidate = (root / user_path).resolve()

  try:
    candidate.relative_to(root)
  except Exception as err:
    raise ValueError('path escapes repo root') from err

  return candidate


@tool
def get_time_utc() -> str:
  """Get the current time in UTC as ISO-8601."""
  return datetime.now(timezone.utc).isoformat()


@tool
def calculator(expression: str) -> str:
  """
  Safely evaluate a simple math expression.

  Allowed:
  - numbers
  - + - * / ** ( ) and whitespace
  - selected math functions: sqrt, sin, cos, tan, log, exp
  - constants: pi, e
  """
  if not expression or not isinstance(expression, str):
    return 'Error: expression must be a non-empty string'

  if len(expression) > 200:
    return 'Error: expression too long'

  allowed = re.compile(r'^[0-9\.\+\-\*\/\(\)\s,eipnqrtasclogx\^]+$')
  if not allowed.match(expression.lower().replace('**', '^')):
    return 'Error: expression contains unsupported characters'

  safe_globals = {
    '__builtins__': {},
    'pi': math.pi,
    'e': math.e,
    'sqrt': math.sqrt,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'log': math.log,
    'exp': math.exp
  }

  try:
    # allow caret as exponent for convenience
    expr = expression.replace('^', '**')
    value = eval(expr, safe_globals, {})
  except Exception as err:
    return f'Error: {err}'

  return str(value)


@tool
def list_dir(path: str = '.') -> str:
  """List files/directories at `path` (restricted to this repo)."""
  try:
    target = _resolve_in_repo(path)
  except Exception as err:
    return f'Error: {err}'

  if not target.exists():
    return 'Error: path does not exist'
  if not target.is_dir():
    return 'Error: path is not a directory'

  entries = []
  for p in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
    kind = 'dir' if p.is_dir() else 'file'
    rel = p.relative_to(_repo_root())
    entries.append(f'{kind}\t{rel}')

  return '\n'.join(entries) if entries else '(empty)'


@tool
def read_text_file(path: str) -> str:
  """Read a UTF-8 text file at `path` (restricted to this repo)."""
  try:
    target = _resolve_in_repo(path)
  except Exception as err:
    return f'Error: {err}'

  if not target.exists():
    return 'Error: file does not exist'
  if not target.is_file():
    return 'Error: path is not a file'

  # keep it simple + safe
  if target.stat().st_size > 50_000:
    return 'Error: file too large (max 50KB)'

  try:
    return target.read_text(encoding='utf-8')
  except Exception as err:
    return f'Error: {err}'


@tool
def write_text_file(path: str, content: str) -> str:
  """Write `content` to `path` (overwrites). Restricted to this repo."""
  if content is None:
    return 'Error: content is required'

  try:
    target = _resolve_in_repo(path)
  except Exception as err:
    return f'Error: {err}'

  if len(content) > 50_000:
    return 'Error: content too large (max 50KB)'

  try:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')
  except Exception as err:
    return f'Error: {err}'

  rel = target.relative_to(_repo_root())
  return f'Wrote {rel}'


@tool
def append_text_file(path: str, content: str) -> str:
  """Append `content` to `path` (creates if missing). Restricted to this repo."""
  if content is None:
    return 'Error: content is required'

  try:
    target = _resolve_in_repo(path)
  except Exception as err:
    return f'Error: {err}'

  if len(content) > 50_000:
    return 'Error: content too large (max 50KB)'

  try:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a', encoding='utf-8') as f:
      f.write(content)
  except Exception as err:
    return f'Error: {err}'

  rel = target.relative_to(_repo_root())
  return f'Appended to {rel}'


TOOLS = [
  get_time_utc,
  calculator,
  list_dir,
  read_text_file,
  write_text_file,
  append_text_file
]

TOOLS.extend(RATE_TOOLS)

