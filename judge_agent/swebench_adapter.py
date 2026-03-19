from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Optional


def _iter_jsonl(path: str) -> Iterable[Any]:
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      line = (line or '').strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except Exception:
        continue


def _load_records(path: str) -> list[dict]:
  p = Path(path)
  if not p.exists():
    raise SystemExit(f'Error: input not found: {path}')

  if p.suffix.lower() == '.jsonl':
    out: list[dict] = []
    for obj in _iter_jsonl(path):
      if isinstance(obj, dict):
        out.append(obj)
    return out

  obj = json.loads(p.read_text(encoding='utf-8'))
  if isinstance(obj, list):
    return [x for x in obj if isinstance(x, dict)]

  if isinstance(obj, dict):
    for key in ['results', 'instances', 'data', 'items']:
      v = obj.get(key)
      if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    # if it's a single record, wrap it
    return [obj]

  return []


def _to_bool(value: Any) -> Optional[bool]:
  if value is None:
    return None
  if isinstance(value, bool):
    return value
  if isinstance(value, (int, float)):
    # common: 1/0, or score
    return bool(value)
  s = str(value).strip().lower()
  if s in ('1', 'true', 'yes', 'y', 'on', 'passed', 'pass', 'success', 'resolved'):
    return True
  if s in ('0', 'false', 'no', 'n', 'off', 'failed', 'fail', 'error', 'unresolved'):
    return False
  return None


def _pick_instance_id(rec: dict) -> str:
  for k in ['instance_id', 'id', 'task_id', 'problem_id', 'name']:
    v = rec.get(k)
    if v:
      return str(v)
  return ''


def _pick_correct(rec: dict) -> Optional[bool]:
  for k in ['resolved', 'is_resolved', 'passed', 'pass', 'success', 'correct']:
    v = _to_bool(rec.get(k))
    if v is not None:
      return v
  # numeric scores
  if 'score' in rec:
    try:
      return float(rec.get('score')) >= 1.0
    except Exception:
      pass
  return None


def _pick_duration_ms(rec: dict) -> int:
  for k in ['duration_ms', 'runtime_ms', 'time_ms']:
    v = rec.get(k)
    if v is None:
      continue
    try:
      return int(v)
    except Exception:
      continue
  for k in ['duration_s', 'runtime_s', 'time_s', 'elapsed_s']:
    v = rec.get(k)
    if v is None:
      continue
    try:
      return int(float(v) * 1000)
    except Exception:
      continue
  return 0


def _pick_model(rec: dict) -> str:
  for k in ['model', 'llm', 'selected_model']:
    v = rec.get(k)
    if v:
      return str(v)
  return ''


def main() -> int:
  parser = argparse.ArgumentParser(description='Convert SWE-bench results into a standard JSONL format.')
  parser.add_argument('--input', required=True, help='Input SWE-bench results file (.json or .jsonl)')
  parser.add_argument('--output', default='logs/swebench.jsonl', help='Output JSONL path (default: logs/swebench.jsonl)')
  parser.add_argument('--run-id', default='', help='Optional run id (default: unix timestamp)')
  parser.add_argument('--start-prompt-idx', type=int, default=1, help='Starting prompt_idx to assign (default: 1)')
  args = parser.parse_args()

  run_id = (args.run_id or '').strip() or str(int(time.time()))
  records = _load_records(args.input)

  out_path = Path(args.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  prompt_idx = int(args.start_prompt_idx)
  written = 0

  with out_path.open('w', encoding='utf-8') as f:
    for rec in records:
      if not isinstance(rec, dict):
        continue
      item_id = _pick_instance_id(rec)
      if not item_id:
        continue

      correct = _pick_correct(rec)
      if correct is None:
        # default to false if unknown
        correct = False

      payload = {
        'event': 'swebench.item',
        'run_id': run_id,
        'item_id': item_id,
        'prompt_idx': prompt_idx,
        'model': _pick_model(rec),
        'duration_ms': _pick_duration_ms(rec),
        'correct': bool(correct),
        'correct_int': 1 if correct else 0,
        # keep some optional info if present
        'status': rec.get('status') or rec.get('result') or rec.get('outcome'),
        'error': rec.get('error') or rec.get('exception')
      }
      f.write(json.dumps(payload, ensure_ascii=False) + '\n')
      written += 1
      prompt_idx += 1

  print(f'Wrote {written} SWE-bench items to "{out_path}"')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

