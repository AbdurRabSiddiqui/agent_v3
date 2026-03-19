from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
  parser = argparse.ArgumentParser(description='Build simple model->avg_effective_power_w priors from labeled GPU CSV.')
  parser.add_argument('--gpu-csv', default='logs/gpu_energy.csv', help='Input GPU CSV with phase_model/effective_power_w columns')
  parser.add_argument('--out-json', default='logs/model_power_priors.json', help='Output JSON mapping model->avg_effective_power_w')
  args = parser.parse_args()

  sums = defaultdict(float)
  counts = defaultdict(int)

  with open(args.gpu_csv, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
      model = str((row.get('phase_model') or '')).strip()
      if not model:
        continue
      try:
        p = float(row.get('effective_power_w') or 0.0)
      except Exception:
        continue
      sums[model] += p
      counts[model] += 1

  out = {}
  for model, n in counts.items():
    if n <= 0:
      continue
    out[model] = {
      'avg_effective_power_w': sums[model] / n,
      'sample_count': int(n)
    }

  Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
  with open(args.out_json, 'w', encoding='utf-8') as f:
    f.write(json.dumps(out, ensure_ascii=False, indent=2))
    f.write('\n')

  print(f'Wrote {len(out)} models to "{args.out_json}"')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

