from __future__ import annotations

import re
from typing import Optional


def build_math500_prompt(problem: str) -> str:
  return '\n'.join([
    'You are solving a MATH-500 competition problem.',
    'Think privately; do NOT show steps, reasoning, or explanations.',
    r'Output ONLY the final answer in the form: \boxed{...}',
    'Do not output anything else (no words, no punctuation outside the box).',
    r'Use simplest exact form. For rationals use \frac{a}{b}. Avoid decimals unless unavoidable.',
    '',
    f'Problem: {problem}'
  ])


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


def _canonicalize_latex(s: str) -> str:
  t = str(s or '')
  t = t.replace('\\dfrac', '\\frac')
  t = t.replace('\\left', '')
  t = t.replace('\\right', '')
  t = t.replace('\\,', '')
  return t


def normalize_math500_answer(s: str) -> str:
  if s is None:
    return ''
  s = str(s).strip()
  s = s.replace('$', '')
  s = _canonicalize_latex(s)
  s = re.sub(r'\s+', '', s)
  return s


def is_correct_math500(pred: str, gold: str) -> bool:
  pred_n = normalize_math500_answer(pred)
  gold_n = normalize_math500_answer(gold)
  if pred_n == gold_n:
    return True

  pred_boxed = _extract_boxed(pred)
  gold_boxed = _extract_boxed(gold)
  if pred_boxed and gold_boxed:
    if normalize_math500_answer(pred_boxed) == normalize_math500_answer(gold_boxed):
      return True

  if pred_boxed and normalize_math500_answer(pred_boxed) == gold_n:
    return True

  def to_float(x: str) -> Optional[float]:
    try:
      return float(x)
    except Exception:
      return None

  pf = to_float(pred_n)
  gf = to_float(gold_n)
  if pf is not None and gf is not None:
    return abs(pf - gf) <= 1e-6

  return False

