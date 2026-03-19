from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class VerifyResult:
  decision: str  # accept | escalate | unsure
  confidence: float
  reason: str
  verifier_model: str = ''
  raw: str = ''
  parsed: Optional[Dict[str, Any]] = None


def deterministic_checks(*, requires_boxed: bool, answer: str) -> Optional[VerifyResult]:
  text = (answer or '').strip()
  if not text:
    return VerifyResult(decision='escalate', confidence=1.0, reason='empty_answer')
  if requires_boxed:
    if not re.fullmatch(r'\s*\\boxed\{[\s\S]*\}\s*', text):
      return VerifyResult(decision='escalate', confidence=1.0, reason='format:not_boxed_only')
  return None


def build_verify_prompt(*, user_input: str, answer: str, requires_boxed: bool) -> str:
  rules = [
    'You are a strict verifier for an answer produced by another model.',
    'Decide whether the answer is good enough to return to the user, or whether we should escalate to a stronger model.',
    '',
    'Return STRICT JSON only with this schema:',
    '{"accept": <bool>, "confidence": <0..1>, "reason": "<short>"}',
    '',
    'Rules:',
    '- Output MUST be valid JSON only (no markdown, no prose).',
    '- confidence must be a number between 0 and 1.',
    '- Be conservative: if you are not confident the answer is correct/complete, set accept=false.',
    '- If the answer violates output constraints, set accept=false with high confidence.'
  ]
  if requires_boxed:
    rules.extend([
      '',
      'Constraint: output must be ONLY a single LaTeX expression of the form \\boxed{...}, with nothing else.'
    ])
  return '\n'.join([
    *rules,
    '',
    f'User request:\n{user_input}',
    '',
    f'Candidate answer:\n{(answer or "").strip()}'
  ])


def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
  try:
    v = json.loads(raw)
    return v if isinstance(v, dict) else None
  except Exception:
    m = re.search(r'({[\\s\\S]*})', raw or '')
    if not m:
      return None
    try:
      v = json.loads(m.group(1))
      return v if isinstance(v, dict) else None
    except Exception:
      return None


def try_parse_verify_json(raw: str) -> Optional[Tuple[bool, float, str, Dict[str, Any]]]:
  parsed = _try_parse_json(raw)
  if not parsed:
    return None
  accept = parsed.get('accept')
  confidence = parsed.get('confidence')
  reason = parsed.get('reason')
  if not isinstance(accept, bool):
    return None
  try:
    conf = float(confidence)
  except Exception:
    return None
  if conf < 0:
    conf = 0.0
  if conf > 1:
    conf = 1.0
  return accept, conf, str(reason or ''), parsed


def decision_from_verify(
  *,
  accept: bool,
  confidence: float,
  accept_threshold: float,
  reject_threshold: float
) -> str:
  if accept and confidence >= accept_threshold:
    return 'accept'
  if (not accept) and confidence >= reject_threshold:
    return 'escalate'
  return 'unsure'

