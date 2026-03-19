from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EnergySlice:
  prompt_idx: int
  phase: str
  phase_model: str
  energy_j: float
  duration_s: float


def _to_int(value, default: int = 0) -> int:
  try:
    return int(value)
  except Exception:
    return int(default)


def _to_float(value, default: float = 0.0) -> float:
  try:
    return float(value)
  except Exception:
    return float(default)


def attribute_effective_energy(
  *,
  gpu_csv_path: str,
  use_unix_ts: bool = True
) -> dict[tuple[int, str, str], EnergySlice]:
  """
  Attribute *effective* energy (J) to (prompt_idx, phase, phase_model) by integrating per-row power over time.

  This avoids relying on cumulative energy columns, which are global; instead we compute:
    E += effective_power_w * dt
  where dt is the time since the previous sample.
  """
  rows: list[dict] = []
  with open(gpu_csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
      rows.append(r)

  if not rows:
    return {}

  def row_time(row: dict) -> float:
    if use_unix_ts and row.get('unix_ts'):
      return _to_float(row.get('unix_ts'), 0.0)
    return _to_float(row.get('elapsed_s'), 0.0)

  rows_sorted = sorted(rows, key=row_time)

  out: dict[tuple[int, str, str], dict] = {}
  prev_t: Optional[float] = None
  prev_eff_p: float = 0.0
  prev_key: Optional[tuple[int, str, str]] = None

  for row in rows_sorted:
    t = row_time(row)
    eff_p = _to_float(row.get('effective_power_w'), 0.0)
    idx_raw = (row.get('prompt_idx') or '').strip()
    phase = str(row.get('phase') or '').strip()
    phase_model = str(row.get('phase_model') or '').strip()

    key = None
    idx = _to_int(idx_raw, default=-1) if idx_raw else -1
    if idx > 0 and (phase or phase_model):
      key = (idx, phase, phase_model)

    if prev_t is not None and prev_key is not None:
      dt = max(t - prev_t, 0.0)
      bucket = out.get(prev_key)
      if bucket is None:
        out[prev_key] = {'energy_j': prev_eff_p * dt, 'duration_s': dt}
      else:
        bucket['energy_j'] += prev_eff_p * dt
        bucket['duration_s'] += dt

    prev_t = t
    prev_eff_p = eff_p
    prev_key = key

  result: dict[tuple[int, str, str], EnergySlice] = {}
  for (idx, phase, phase_model), b in out.items():
    result[(idx, phase, phase_model)] = EnergySlice(
      prompt_idx=idx,
      phase=phase,
      phase_model=phase_model,
      energy_j=float(b.get('energy_j') or 0.0),
      duration_s=float(b.get('duration_s') or 0.0)
    )
  return result

