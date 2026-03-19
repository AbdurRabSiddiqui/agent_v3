from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional, Union

from langchain_core.tools import tool


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


EnergyType = Literal['flat', 'tou']
NetMeteringType = Literal['none', 'retail', 'wholesale']


def _validate_24(values: list[float], *, name: str) -> list[float]:
  if not isinstance(values, list):
    raise ValueError(f'{name} must be a list of 24 numbers')
  if len(values) != 24:
    raise ValueError(f'{name} must have length 24')
  clean: list[float] = []
  for v in values:
    try:
      fv = float(v)
    except Exception as err:
      raise ValueError(f'{name} contains non-numeric values') from err
    if fv < 0:
      raise ValueError(f'{name} cannot contain negative values')
    clean.append(fv)
  return clean


def _load_plans(path: str) -> dict:
  target = _resolve_in_repo(path)
  if not target.exists():
    raise ValueError('plans file does not exist')
  data = json.loads(target.read_text(encoding='utf-8'))
  if not isinstance(data, dict) or 'plans' not in data or not isinstance(data['plans'], list):
    raise ValueError('plans file must be an object with a "plans" array')
  return data


def _energy_rate_for_hour(plan: dict, hour: int) -> float:
  energy = plan.get('energy') or {}
  if energy.get('type') == 'flat':
    return float(energy.get('rate_per_kwh'))
  if energy.get('type') == 'tou':
    peak_hours = energy.get('peak_hours_local') or []
    is_peak = hour in peak_hours
    return float(energy.get('peak_rate_per_kwh' if is_peak else 'offpeak_rate_per_kwh'))
  raise ValueError('unsupported energy plan type')


def _compute_daily_bill(plan: dict, usage: list[float], solar: list[float], *, has_solar: bool) -> dict:
  net = plan.get('net_metering') or {'type': 'none'}
  nm_type: str = net.get('type') or 'none'

  customer_charge = float(plan.get('customer_charge_per_day') or 0)
  import_cost = 0.0
  export_credit = 0.0
  imported_kwh = 0.0
  exported_kwh = 0.0

  for hour in range(24):
    rate = _energy_rate_for_hour(plan, hour)
    load = usage[hour]
    gen = solar[hour] if has_solar else 0.0
    net_kwh = load - gen

    if net_kwh >= 0:
      imported_kwh += net_kwh
      import_cost += net_kwh * rate
      continue

    exported = abs(net_kwh)
    exported_kwh += exported

    if nm_type == 'none':
      continue
    if nm_type == 'retail':
      export_credit += exported * rate
      continue
    if nm_type == 'wholesale':
      export_rate = float(net.get('export_rate_per_kwh') or 0)
      export_credit += exported * export_rate
      continue

    raise ValueError('unsupported net metering type')

  total = customer_charge + import_cost - export_credit
  return {
    'currency': plan.get('currency') or 'USD',
    'customer_charge': round(customer_charge, 4),
    'imported_kwh': round(imported_kwh, 4),
    'exported_kwh': round(exported_kwh, 4),
    'import_cost': round(import_cost, 4),
    'export_credit': round(export_credit, 4),
    'total': round(total, 4)
  }


@tool
def rate_list_plans(plans_path: str = 'rate_plans.sample.json') -> str:
  """List available rate plans from a JSON file in this repo."""
  try:
    data = _load_plans(plans_path)
    plans = data['plans']
    lines = []
    for p in plans:
      lines.append(f"{p.get('id')}\t{p.get('name')}")
    return '\n'.join(lines) if lines else '(no plans)'
  except Exception as err:
    return f'Error: {err}'


@tool
def rate_estimate_daily_bill(
  plan_id: str,
  usage_hourly_kwh: Union[List[float], str],
  has_solar: bool = False,
  solar_hourly_kwh: Optional[Union[List[float], str]] = None,
  plans_path: str = 'rate_plans.sample.json'
) -> str:
  """
  Estimate daily bill for a given plan and typical-day hourly usage (kWh).
  Optionally include solar generation (kWh) and net metering rules.
  """
  try:
    usage_raw = json.loads(usage_hourly_kwh) if isinstance(usage_hourly_kwh, str) else usage_hourly_kwh
    solar_raw = (
      json.loads(solar_hourly_kwh)
      if isinstance(solar_hourly_kwh, str)
      else (solar_hourly_kwh or [0] * 24)
    )
    usage = _validate_24(usage_raw, name='usage_hourly_kwh')
    solar = _validate_24(solar_raw, name='solar_hourly_kwh')
    data = _load_plans(plans_path)
    plan = next((p for p in data['plans'] if p.get('id') == plan_id), None)
    if not plan:
      return 'Error: unknown plan_id'
    result = _compute_daily_bill(plan, usage, solar, has_solar=bool(has_solar))
    return json.dumps({'plan_id': plan_id, 'plan_name': plan.get('name'), **result}, indent=2)
  except Exception as err:
    return f'Error: {err}'


@tool
def rate_recommend_plan(
  usage_hourly_kwh: Union[List[float], str],
  has_solar: bool = False,
  solar_hourly_kwh: Optional[Union[List[float], str]] = None,
  plans_path: str = 'rate_plans.sample.json'
) -> str:
  """
  Evaluate all plans in `plans_path` and recommend the lowest-cost option
  for the given typical-day usage/solar profile.
  """
  try:
    usage_raw = json.loads(usage_hourly_kwh) if isinstance(usage_hourly_kwh, str) else usage_hourly_kwh
    solar_raw = (
      json.loads(solar_hourly_kwh)
      if isinstance(solar_hourly_kwh, str)
      else (solar_hourly_kwh or [0] * 24)
    )
    usage = _validate_24(usage_raw, name='usage_hourly_kwh')
    solar = _validate_24(solar_raw, name='solar_hourly_kwh')
    data = _load_plans(plans_path)
    scored = []
    for plan in data['plans']:
      bill = _compute_daily_bill(plan, usage, solar, has_solar=bool(has_solar))
      scored.append({
        'plan_id': plan.get('id'),
        'plan_name': plan.get('name'),
        'total': bill['total'],
        'currency': bill['currency']
      })
    scored.sort(key=lambda x: x['total'])
    return json.dumps({'recommendation': scored[0] if scored else None, 'all': scored}, indent=2)
  except Exception as err:
    return f'Error: {err}'


RATE_TOOLS = [
  rate_list_plans,
  rate_estimate_daily_bill,
  rate_recommend_plan
]

