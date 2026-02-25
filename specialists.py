from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from rate_tools import rate_estimate_daily_bill, rate_recommend_plan
from tools import calculator


def _extract_first_json_list(text: str) -> Optional[str]:
  if not text:
    return None
  m = re.search(r'\[[^\]]+\]', text, flags=re.S)
  return m.group(0) if m else None


def _parse_24_floats(list_text: str) -> Optional[List[float]]:
  try:
    data = json.loads(list_text)
  except Exception:
    return None
  if not isinstance(data, list) or len(data) != 24:
    return None
  out: List[float] = []
  for v in data:
    try:
      fv = float(v)
    except Exception:
      return None
    if fv < 0:
      return None
    out.append(fv)
  return out


def try_power_recommendation(user_input: str) -> Optional[str]:
  """
  Deterministic solver for power plan selection.
  If the user includes a 24-value list, call the rate plan tool directly.
  """
  list_text = _extract_first_json_list(user_input)
  if not list_text:
    return None

  usage = _parse_24_floats(list_text)
  if not usage:
    return None

  # invoke tool directly to avoid LLM hallucinating prices
  return rate_recommend_plan.invoke({'usage_hourly_kwh': usage})


def try_math_expression(user_input: str) -> Optional[str]:
  """
  If the input looks like a simple arithmetic expression, evaluate it directly.
  """
  if not user_input:
    return None

  # allow "what is ..." wrapper
  text = user_input.strip()
  text = re.sub(r'^(what is|calculate|compute)\s+', '', text, flags=re.I).strip()

  # keep it conservative: only evaluate when it looks like pure math
  if not re.fullmatch(r'[\d\.\s\+\-\*\/\^\(\)]+', text):
    return None

  # calculator is a LangChain tool; invoke with dict
  return calculator.invoke({'expression': text})

