from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from training.math500_tasks import load_math500_tasks


def _iter_jsonl(path: str):
  if not path or not Path(path).exists():
    return
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      line = (line or '').strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except Exception:
        continue


def _extract_boxed(text: str) -> Optional[str]:
  if not text:
    return None
  last = (text or '').rfind('\\boxed{')
  if last < 0:
    return None
  i = last + len('\\boxed{')
  depth = 1
  out = []
  while i < len(text):
    ch = text[i]
    if ch == '{':
      depth += 1
      out.append(ch)
    elif ch == '}':
      depth -= 1
      if depth == 0:
        return ''.join(out).strip()
      out.append(ch)
    else:
      out.append(ch)
    i += 1
  return None


def _normalize_answer(s: str) -> str:
  if s is None:
    return ''
  t = str(s).strip().replace('$', '')
  t = t.replace('\\dfrac', '\\frac')
  t = t.replace('\\left', '')
  t = t.replace('\\right', '')
  t = t.replace('\\,', '')
  t = re.sub(r'\\s+', '', t)
  return t


def _compare(pred: str, gold: str) -> bool:
  pn = _normalize_answer(pred)
  gn = _normalize_answer(gold)
  if pn == gn:
    return True
  pb = _extract_boxed(pred)
  gb = _extract_boxed(gold)
  if pb and gb:
    return _normalize_answer(pb) == _normalize_answer(gb)
  if pb and _normalize_answer(pb) == gn:
    return True
  return False


def _to_int(v: Any, default: int = 0) -> int:
  try:
    return int(v)
  except Exception:
    return int(default)


def _to_float(v: Any, default: float = 0.0) -> float:
  try:
    return float(v)
  except Exception:
    return float(default)


def main() -> int:
  parser = argparse.ArgumentParser(description='Build an offline dataset from existing MATH-500 logs.')
  parser.add_argument('--limit', type=int, default=400)
  parser.add_argument('--eval-jsonl', default='logs/math500.jsonl', help='Per-item MATH-500 results JSONL')
  parser.add_argument('--agent-trace', default='logs/agent_trace.jsonl', help='Agent trace JSONL (draft/judge events)')
  parser.add_argument('--gpu-csv', default='logs/gpu_energy.csv', help='GPU CSV with prompt_idx labels (optional)')
  parser.add_argument('--out-jsonl', default='logs/math500_offline_dataset.jsonl', help='Output JSONL path')
  args = parser.parse_args()

  tasks = load_math500_tasks(limit=args.limit)
  gold_by_idx: Dict[int, str] = {}
  prompt_by_idx: Dict[int, str] = {}
  for t in tasks:
    i = _to_int(t.get('idx'), -1)
    if i < 0:
      continue
    gold_by_idx[i] = str(t.get('gold') or '')
    prompt_by_idx[i] = str(t.get('prompt') or '')

  # Load per-prompt energy stats (optional)
  power_by_prompt: Dict[int, Dict[str, float]] = {}
  if args.gpu_csv and Path(args.gpu_csv).exists():
    try:
      from eval_report import load_power_stats_from_gpu_csv  # type: ignore

      ps = load_power_stats_from_gpu_csv(args.gpu_csv)
      for k, v in ps.items():
        power_by_prompt[int(k)] = {
          'gpu_effective_energy_j': float(getattr(v, 'effective_energy_j', 0.0) or 0.0),
          'gpu_total_energy_j': float(getattr(v, 'total_energy_j', 0.0) or 0.0),
          'gpu_avg_effective_power_w': float(getattr(v, 'avg_effective_power_w', 0.0) or 0.0),
          'gpu_duration_s': float(getattr(v, 'duration_s', 0.0) or 0.0)
        }
    except Exception:
      power_by_prompt = {}

  # Index trace drafts by prompt_idx and draft_n
  drafts_by_prompt: Dict[int, Dict[int, Dict[str, Any]]] = {}
  for ev in _iter_jsonl(args.agent_trace):
    if not isinstance(ev, dict):
      continue
    if ev.get('event') != 'agent.select.draft':
      continue
    prompt_idx = ev.get('prompt_idx')
    if not isinstance(prompt_idx, int) or prompt_idx <= 0:
      continue
    phase = str(ev.get('phase') or '')
    m = re.search(r'draft_(\\d+)', phase)
    if not m:
      continue
    n = _to_int(m.group(1), -1)
    if n < 0:
      continue
    drafts_by_prompt.setdefault(prompt_idx, {})[n] = ev

  Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
  wrote = 0
  with open(args.out_jsonl, 'w', encoding='utf-8') as f:
    for ev in _iter_jsonl(args.eval_jsonl):
      if not isinstance(ev, dict):
        continue
      if ev.get('event') != 'math500.item':
        continue
      idx = _to_int(ev.get('idx'), -1)
      if idx < 0 or idx >= int(args.limit):
        continue

      prompt_idx = _to_int(ev.get('prompt_idx') or (idx + 1), 0)
      gold = gold_by_idx.get(idx, '')
      prompt = prompt_by_idx.get(idx, '')

      drafts = drafts_by_prompt.get(prompt_idx) or {}
      draft0 = drafts.get(0) or {}
      draft0_out = str(draft0.get('output_trunc') or '')
      draft0_model = str(draft0.get('model') or '')
      draft0_correct = 1 if (gold and draft0_out and _compare(draft0_out, gold)) else 0

      out = {
        'event': 'math500.offline_item',
        'idx': idx,
        'prompt_idx': prompt_idx,
        'prompt_preview': prompt[:160],
        'selected_model': str(ev.get('model') or ''),
        'selected_reason': str(ev.get('reason') or ''),
        'duration_ms': _to_int(ev.get('duration_ms')),
        'correct_int': _to_int(ev.get('correct_int'), 1 if ev.get('correct') else 0),
        'draft0_model': draft0_model,
        'draft0_output_trunc': draft0_out,
        'draft0_correct_int': int(draft0_correct)
      }
      if prompt_idx in power_by_prompt:
        out.update(power_by_prompt[prompt_idx])

      f.write(json.dumps(out, ensure_ascii=False) + '\n')
      wrote += 1

  print(f'Wrote {wrote} items to "{args.out_jsonl}"')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

