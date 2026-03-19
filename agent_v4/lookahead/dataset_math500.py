from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from lookahead.math500 import build_math500_prompt, is_correct_math500
from lookahead.power_attribution import attribute_effective_energy


@dataclass(frozen=True)
class DraftRow:
  prompt_idx: int
  item_idx: int
  model: str
  prompt: str
  gold: str
  output: str
  correct_int: int
  duration_ms: int
  energy_j: float


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


def build_dataset(
  *,
  agent_trace_path: str,
  gpu_csv_path: Optional[str],
  limit: int = 0
) -> list[DraftRow]:
  ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
  total = len(ds) if limit <= 0 else min(limit, len(ds))

  prompt_idx_to_item: dict[int, int] = {}
  prompt_idx_to_selected: dict[int, str] = {}
  drafts: list[dict] = []

  for ev in _iter_jsonl(agent_trace_path):
    if not isinstance(ev, dict):
      continue
    pidx = ev.get('prompt_idx')
    if not isinstance(pidx, int) or pidx <= 0:
      continue

    if ev.get('event') == 'math500.item':
      item_idx = ev.get('idx')
      if isinstance(item_idx, int):
        prompt_idx_to_item[pidx] = item_idx
      model = ev.get('model')
      if isinstance(model, str):
        prompt_idx_to_selected[pidx] = model

    if ev.get('event') == 'agent.select.draft':
      model = str(ev.get('model') or '').strip()
      out = str(ev.get('output_trunc') or '').strip()
      duration_ms = int(ev.get('duration_ms') or 0)
      phase = str(ev.get('phase') or '')
      if model and out:
        drafts.append({
          'prompt_idx': pidx,
          'model': model,
          'output': out,
          'duration_ms': duration_ms,
          'phase': phase
        })

  energy_map = {}
  if gpu_csv_path and Path(gpu_csv_path).exists():
    energy_map = attribute_effective_energy(gpu_csv_path=gpu_csv_path)

  rows: list[DraftRow] = []
  for d in drafts:
    pidx = int(d['prompt_idx'])
    item_idx = int(prompt_idx_to_item.get(pidx, -1))
    if item_idx < 0 or item_idx >= total:
      continue
    problem = ds[item_idx]['problem']
    gold = ds[item_idx]['answer']
    prompt = build_math500_prompt(problem)
    output = str(d['output'])
    correct_int = 1 if is_correct_math500(output, gold) else 0
    model = str(d['model'])
    duration_ms = int(d.get('duration_ms') or 0)

    # Attribute energy for draft phases of this model.
    # We sum any slices whose phase starts with "draft_" and matches phase_model == model.
    energy_j = 0.0
    for (idx, phase, phase_model), sl in energy_map.items():
      if idx != pidx:
        continue
      if phase_model != model:
        continue
      if not str(phase or '').startswith('draft_'):
        continue
      energy_j += float(sl.energy_j)

    rows.append(DraftRow(
      prompt_idx=pidx,
      item_idx=item_idx,
      model=model,
      prompt=prompt,
      gold=gold,
      output=output,
      correct_int=correct_int,
      duration_ms=duration_ms,
      energy_j=energy_j
    ))

  return rows


def main() -> int:
  ap = argparse.ArgumentParser(description='Build Lookahead training dataset from existing agent traces (MATH-500).')
  ap.add_argument('--agent-trace', default='logs/agent_trace.jsonl', help='Path to agent trace JSONL')
  ap.add_argument('--gpu-csv', default='logs/gpu_energy.csv', help='Path to GPU CSV (optional; energy_j will be 0 if missing)')
  ap.add_argument('--limit', type=int, default=0, help='Limit to first N items (0=all)')
  ap.add_argument('--out-jsonl', default='logs/lookahead_math500_dataset.jsonl', help='Output JSONL path')
  args = ap.parse_args()

  rows = build_dataset(
    agent_trace_path=args.agent_trace,
    gpu_csv_path=args.gpu_csv if args.gpu_csv else None,
    limit=int(args.limit or 0)
  )

  Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
  with open(args.out_jsonl, 'w', encoding='utf-8') as f:
    for r in rows:
      f.write(json.dumps(r.__dict__, ensure_ascii=False) + '\n')

  print(f'Wrote {len(rows)} draft rows to "{args.out_jsonl}"')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

