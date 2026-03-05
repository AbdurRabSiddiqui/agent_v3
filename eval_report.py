from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class PowerStats:
  prompt_idx: int
  sample_count: int
  duration_s: float
  total_energy_j: float
  effective_energy_j: float
  avg_total_power_w: float
  avg_effective_power_w: float


def _iter_jsonl(path: str):
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      line = (line or '').strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except Exception:
        continue


def _to_int(value: Any, default: int = 0) -> int:
  try:
    return int(value)
  except Exception:
    return int(default)


def _to_float(value: Any, default: float = 0.0) -> float:
  try:
    return float(value)
  except Exception:
    return float(default)


def load_power_stats_from_gpu_csv(csv_path: str) -> dict[int, PowerStats]:
  """
  Uses prompt_idx labels in the GPU CSV to compute per-prompt energy and average power.
  Requires columns: prompt_idx, elapsed_s, total_power_w, effective_power_w, total_energy_j, effective_energy_j
  """
  per: dict[int, dict] = {}
  with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
      idx_raw = (row.get('prompt_idx') or '').strip()
      if not idx_raw:
        continue
      idx = _to_int(idx_raw, default=-1)
      if idx <= 0:
        continue

      elapsed = _to_float(row.get('elapsed_s'))
      total_power = _to_float(row.get('total_power_w'))
      eff_power = _to_float(row.get('effective_power_w'))
      total_energy = _to_float(row.get('total_energy_j'))
      eff_energy = _to_float(row.get('effective_energy_j'))

      bucket = per.get(idx)
      if bucket is None:
        per[idx] = {
          'first_elapsed': elapsed,
          'last_elapsed': elapsed,
          'first_total_energy': total_energy,
          'last_total_energy': total_energy,
          'first_eff_energy': eff_energy,
          'last_eff_energy': eff_energy,
          'sum_total_power': total_power,
          'sum_eff_power': eff_power,
          'n': 1
        }
        continue

      bucket['last_elapsed'] = elapsed
      bucket['last_total_energy'] = total_energy
      bucket['last_eff_energy'] = eff_energy
      bucket['sum_total_power'] += total_power
      bucket['sum_eff_power'] += eff_power
      bucket['n'] += 1

  out: dict[int, PowerStats] = {}
  for idx, b in per.items():
    n = int(b['n']) if b['n'] else 0
    duration = max(float(b['last_elapsed']) - float(b['first_elapsed']), 0.0)
    total_energy_j = max(float(b['last_total_energy']) - float(b['first_total_energy']), 0.0)
    eff_energy_j = max(float(b['last_eff_energy']) - float(b['first_eff_energy']), 0.0)
    avg_total = (float(b['sum_total_power']) / n) if n else 0.0
    avg_eff = (float(b['sum_eff_power']) / n) if n else 0.0
    out[idx] = PowerStats(
      prompt_idx=idx,
      sample_count=n,
      duration_s=duration,
      total_energy_j=total_energy_j,
      effective_energy_j=eff_energy_j,
      avg_total_power_w=avg_total,
      avg_effective_power_w=avg_eff
    )
  return out


def load_trace_stats(*, trace_jsonl: str) -> dict[int, dict]:
  """
  Return per-prompt stats derived from agent_trace.jsonl.
  Keys we track:
    - selected_model (agent.select.final)
    - judge_model_used (agent.select.judge)
    - timeouts_count (any event with timed_out=true)
    - retries_count (draft_failed + judge_failed + tools.escalate)
  """
  if not trace_jsonl or not Path(trace_jsonl).exists():
    return {}

  per: dict[int, dict] = {}
  for ev in _iter_jsonl(trace_jsonl):
    if not isinstance(ev, dict):
      continue
    prompt_idx = ev.get('prompt_idx')
    if not isinstance(prompt_idx, int) or prompt_idx <= 0:
      continue

    bucket = per.get(prompt_idx)
    if bucket is None:
      bucket = {
        'selected_model': '',
        'judge_model_used': '',
        'timeouts_count': 0,
        'retries_count': 0
      }
      per[prompt_idx] = bucket

    event = str(ev.get('event') or '')
    if event == 'agent.select.final':
      bucket['selected_model'] = str(ev.get('model') or bucket.get('selected_model') or '')
    if event == 'agent.select.judge':
      bucket['judge_model_used'] = str(ev.get('judge_model') or bucket.get('judge_model_used') or '')

    if bool(ev.get('timed_out')):
      bucket['timeouts_count'] = int(bucket.get('timeouts_count') or 0) + 1

    if event in ('agent.select.draft_failed', 'agent.select.judge_failed', 'agent.select.tools.escalate'):
      bucket['retries_count'] = int(bucket.get('retries_count') or 0) + 1

  return per


def main() -> int:
  parser = argparse.ArgumentParser(description='Aggregate eval JSONL + labeled GPU CSV into results.csv/results.jsonl.')
  parser.add_argument('--benchmark', default='math500', help='Benchmark name (default: math500)')
  parser.add_argument('--eval-jsonl', default='logs/math500.jsonl', help='Input eval JSONL path (default: logs/math500.jsonl)')
  parser.add_argument('--agent-trace', default='logs/agent_trace.jsonl', help='Optional agent trace JSONL path (default: logs/agent_trace.jsonl)')
  parser.add_argument('--gpu-csv', default='logs/gpu_energy.csv', help='Input GPU CSV path (default: logs/gpu_energy.csv)')
  parser.add_argument('--out-jsonl', default='logs/results.jsonl', help='Output JSONL path (default: logs/results.jsonl)')
  parser.add_argument('--out-csv', default='logs/results.csv', help='Output CSV path (default: logs/results.csv)')
  parser.add_argument('--run-id', default='', help='Optional run identifier (default: unix timestamp)')
  args = parser.parse_args()

  run_id = (args.run_id or '').strip() or str(int(time.time()))
  benchmark = (args.benchmark or 'math500').strip()

  if not Path(args.eval_jsonl).exists():
    raise SystemExit(f'Error: eval JSONL not found: {args.eval_jsonl}')
  if not Path(args.gpu_csv).exists():
    raise SystemExit(f'Error: GPU CSV not found: {args.gpu_csv}')

  power = load_power_stats_from_gpu_csv(args.gpu_csv)
  trace_stats = load_trace_stats(trace_jsonl=args.agent_trace)

  rows_out: list[dict] = []
  for ev in _iter_jsonl(args.eval_jsonl):
    if not isinstance(ev, dict):
      continue
    if ev.get('event') != 'math500.item' and benchmark == 'math500':
      continue
    if benchmark == 'swebench' and ev.get('event') != 'swebench.item':
      continue

    item_id = str(ev.get('idx') if benchmark == 'math500' else ev.get('item_id'))
    prompt_idx = _to_int(ev.get('prompt_idx') or (_to_int(ev.get('idx'), 0) + 1))
    correct_int = _to_int(ev.get('correct_int'), 1 if ev.get('correct') else 0)

    ps: Optional[PowerStats] = power.get(prompt_idx)
    ts = trace_stats.get(prompt_idx) or {}

    row = {
      'run_id': run_id,
      'benchmark': benchmark,
      'item_id': item_id,
      'prompt_idx': prompt_idx,
      'model': str(ev.get('model') or ts.get('selected_model') or ''),
      'selected_model': str(ts.get('selected_model') or ''),
      'judge_model_used': str(ts.get('judge_model_used') or ''),
      'timeouts_count': _to_int(ts.get('timeouts_count')),
      'retries_count': _to_int(ts.get('retries_count')),
      'temperature': _to_float(ev.get('temperature')),
      'duration_ms': _to_int(ev.get('duration_ms')),
      'correct_int': int(correct_int),
      'level': ev.get('level'),
      'type': ev.get('type'),
      'reason': str(ev.get('reason') or '')
    }

    if ps is None:
      row.update({
        'gpu_sample_count': 0,
        'gpu_duration_s': 0.0,
        'gpu_total_energy_j': 0.0,
        'gpu_effective_energy_j': 0.0,
        'gpu_avg_total_power_w': 0.0,
        'gpu_avg_effective_power_w': 0.0
      })
    else:
      row.update({
        'gpu_sample_count': ps.sample_count,
        'gpu_duration_s': ps.duration_s,
        'gpu_total_energy_j': ps.total_energy_j,
        'gpu_effective_energy_j': ps.effective_energy_j,
        'gpu_avg_total_power_w': ps.avg_total_power_w,
        'gpu_avg_effective_power_w': ps.avg_effective_power_w
      })

    rows_out.append(row)

  Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
  with open(args.out_jsonl, 'w', encoding='utf-8') as f:
    for r in rows_out:
      f.write(json.dumps(r, ensure_ascii=False) + '\n')

  fieldnames: list[str] = []
  if rows_out:
    # stable order: start with a preferred list, then append remaining keys
    preferred = [
      'run_id', 'benchmark', 'item_id', 'prompt_idx',
      'model', 'selected_model', 'judge_model_used', 'timeouts_count', 'retries_count',
      'temperature', 'duration_ms', 'correct_int',
      'gpu_sample_count', 'gpu_duration_s', 'gpu_total_energy_j', 'gpu_effective_energy_j',
      'gpu_avg_total_power_w', 'gpu_avg_effective_power_w',
      'level', 'type', 'reason'
    ]
    seen = set()
    for k in preferred:
      if k in rows_out[0]:
        fieldnames.append(k)
        seen.add(k)
    for k in rows_out[0].keys():
      if k not in seen:
        fieldnames.append(k)

  Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
  with open(args.out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames or ['run_id'])
    writer.writeheader()
    for r in rows_out:
      writer.writerow(r)

  print(f'Wrote {len(rows_out)} rows to "{args.out_jsonl}" and "{args.out_csv}"')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

