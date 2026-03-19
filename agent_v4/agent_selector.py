from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent import build_llm
from ollama_utils import env as _env
from ollama_utils import get_installed_ollama_models, is_truthy
from trace_utils import append_jsonl, prompt_fingerprint, write_json
from lookahead.policy import LookaheadPolicy


@dataclass(frozen=True)
class SelectionResult:
  prompt_idx: int
  model: str
  temperature: float
  system_prompt: str
  reason: str


_PROMPT_COUNTER = 0


def _power_label_path() -> str:
  return os.environ.get('POWER_LABEL_PATH') or os.environ.get('AGENT_POWER_LABEL_PATH') or 'logs/power_label.json'


def _set_power_label(*, prompt_idx: int, phase: str, model: str, prompt_text: str):
  path = _power_label_path()
  payload = {
    'ts': time.time(),
    'prompt_idx': int(prompt_idx),
    'phase': str(phase or ''),
    'phase_model': str(model or ''),
    **prompt_fingerprint(prompt_text)
  }
  write_json(path, payload)


def _invoke_chat_with_timeout(
  *,
  model: str,
  temperature: float,
  system_prompt: str,
  chat_history: List[BaseMessage],
  user_input: str,
  timeout_s: float
) -> dict:
  """
  Slim build: synchronous invoke (no subprocess timeout).
  Returns dict: { ok, text, duration_ms, timed_out, error? }
  """
  start = time.time()
  try:
    llm = build_llm(model=model, temperature=temperature)
    messages: List[BaseMessage] = []
    if system_prompt:
      messages.append(SystemMessage(content=system_prompt))
    messages.extend(list(chat_history))
    messages.append(HumanMessage(content=user_input))
    res = llm.invoke(messages)
    text = (getattr(res, 'content', None) or str(res)).strip()
    return {
      'ok': True,
      'timed_out': False,
      'duration_ms': int((time.time() - start) * 1000),
      'text': text
    }
  except Exception as err:
    return {
      'ok': False,
      'timed_out': False,
      'duration_ms': int((time.time() - start) * 1000),
      'error': str(err)
    }


## Tool-agent execution has been removed in the MATH-500-only slimmed version of this repo.


def _requires_boxed(user_input: str) -> bool:
  text = user_input or ''
  return '\\boxed' in text or 'Output ONLY the final answer in the form' in text


def _is_math500_prompt(user_input: str) -> bool:
  text = user_input or ''
  return 'MATH-500 competition problem' in text and _requires_boxed(text)


def _extract_boxed_balanced(text: str) -> Optional[str]:
  """
  Extract the LAST \\boxed{...} content with balanced braces.
  Returns the inner content (without \\boxed{ }).
  """
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


def _canonicalize_latex_math(expr: str) -> str:
  s = str(expr or '')
  s = s.replace('\\dfrac', '\\frac')
  s = s.replace('\\left', '')
  s = s.replace('\\right', '')
  s = s.replace('\\,', '')
  s = re.sub(r'\s+', '', s)
  return s


def _canonicalize_math500_answer(text: str) -> Optional[str]:
  boxed = _extract_boxed_balanced(text or '')
  if not boxed:
    return None
  inner = _canonicalize_latex_math(boxed)
  return f'\\boxed{{{inner}}}'


def _draft_format_score(user_input: str, answer: str) -> Tuple[int, int, int]:
  """
  Higher is better.
  - If prompt requires boxed, prefer answers with \\boxed and with ONLY boxed.
  - Otherwise prefer shorter answers (slight).
  """
  ans = (answer or '').strip()
  if not _requires_boxed(user_input):
    return (0, 0, -len(ans))

  has_boxed = 1 if '\\boxed{' in ans else 0
  only_boxed = 0
  if has_boxed:
    # allow whitespace/newlines around; require entire output be a single boxed expression
    if re.fullmatch(r'\s*\\boxed\{[\s\S]*\}\s*', ans):
      only_boxed = 1
  return (has_boxed, only_boxed, -len(ans))


def _pick_best_draft(user_input: str, drafts: List[Dict[str, Any]]) -> Dict[str, Any]:
  if not drafts:
    return {}
  return sorted(
    drafts,
    key=lambda d: (_draft_format_score(user_input, str(d.get('output') or '')), _parse_model_size_b(str(d.get('model') or ''))),
    reverse=True
  )[0]


def _try_parse_judge_json(raw: str) -> Optional[Dict[str, Any]]:
  """
  Parse judge output. Return dict only if it contains an int winner.
  """
  parsed: Dict[str, Any] = {}
  try:
    parsed = json.loads(raw)
  except Exception:
    m = re.search(r'({[\s\S]*})', raw or '')
    if not m:
      return None
    try:
      parsed = json.loads(m.group(1))
    except Exception:
      return None

  winner = parsed.get('winner')
  if not isinstance(winner, int):
    return None
  return parsed


def _parse_model_size_b(model: str) -> float:
  if not model:
    return 0.0
  m = re.search(r':(\d+(?:\.\d+)?)b', model)
  if m:
    try:
      return float(m.group(1))
    except Exception:
      return 0.0
  if ':mini' in model:
    return 1.0
  return 0.0


def _best_by_size(models: List[str], *, prefer_regex: Optional[str] = None) -> Optional[str]:
  if not models:
    return None
  pool = models
  if prefer_regex:
    preferred = [m for m in models if re.search(prefer_regex, m)]
    if preferred:
      pool = preferred
  pool_sorted = sorted(pool, key=_parse_model_size_b, reverse=True)
  return pool_sorted[0] if pool_sorted else None


def _first_existing(candidates: List[str], installed: List[str]) -> Optional[str]:
  installed_set = set(installed)
  for c in candidates:
    if c and c in installed_set:
      return c
  return None


def _clamp_temperature(value: str, default: float) -> float:
  try:
    v = float(value)
  except Exception:
    return float(default)
  if v < 0:
    return 0.0
  if v > 1:
    return 1.0
  return v


def classify_intent(user_input: str) -> str:
  text = (user_input or '').lower()

  code_markers = ['```', 'stack trace', 'traceback', 'exception', 'error:', 'compile', 'typescript', 'javascript']
  if any(m in text for m in code_markers):
    return 'code'
  if re.search(r'\b(def|class|import|from)\b', text) and re.search(r'[:\(\)]', text):
    return 'code'

  power_keywords = [
    'rate plan', 'tou', 'time-of-use', 'time of use', 'kwh',
    'net metering', 'export', 'import', 'solar', 'panel', 'utility',
    'bill', 'tariff', 'demand charge', 'off-peak', 'on-peak', 'peak'
  ]
  if any(k in text for k in power_keywords):
    return 'general'

  math_keywords = [
    'find', 'determine', 'evaluate', 'compute', 'solve', 'simplify',
    'prove', 'show that',
    'integral', 'derivative', 'equation', 'factor',
    'probability', 'geometry', 'mod', 'congruent', 'polynomial', 'matrix',
    'log', 'sin', 'cos', 'tan',
    'value of', 'how many', 'greatest', 'least', 'integer', 'real number', 'positive'
  ]
  if any(k in text for k in math_keywords):
    return 'math'

  if any(k in text for k in ['\\frac', '\\sqrt', '\\cdot', '\\pi', '\\theta', '\\angle', '\\triangle']):
    return 'math'

  if re.search(r'\b(x|y|z|n|k|m)\b', text) and re.search(r'[=\^\+\-\*\/\(\)]', text):
    return 'math'

  if re.search(r'[\d][\d\s\+\-\*\/\^\(\)\.]+', text):
    return 'math'

  return 'general'


def _needs_tools(user_input: str) -> bool:
  return False


def _select_temperature(intent: str, *, is_tool: bool) -> float:
  if is_tool:
    return _clamp_temperature(_env('AGENT_TEMP_TOOL', _env('ROUTER_TEMP_TOOL', '0.1')), 0.1)
  if intent == 'math':
    return _clamp_temperature(_env('AGENT_TEMP_MATH', _env('ROUTER_TEMP_MATH', '0.2')), 0.2)
  if intent == 'code':
    return _clamp_temperature(_env('AGENT_TEMP_CODE', _env('ROUTER_TEMP_CODE', '0.2')), 0.2)
  return _clamp_temperature(_env('AGENT_TEMP_GENERAL', _env('ROUTER_TEMP_GENERAL', '0.3')), 0.3)


def _select_system_prompt(intent: str, *, is_tool: bool) -> str:
  if is_tool:
    return '\n'.join([
      'You are a reliable local tool-using assistant running via Ollama.',
      'Goal: complete the user task correctly with minimal cost/latency.',
      'Use tools when they are the fastest/safest path to truth (files/time/calculation).',
      'Never invent tool outputs. If a tool errors, adjust inputs and retry.',
      'Minimize tool calls: plan first, then call the smallest set of tools.',
      'If user intent is unclear, ask one targeted question and stop.'
    ])
  if intent == 'math':
    return '\n'.join([
      'Solve the problem carefully.',
      'Return only the final answer (no steps).',
      'Prefer exact forms (fractions, radicals) over decimals.'
    ])
  if intent == 'code':
    return '\n'.join([
      'You are a senior software engineer.',
      'Return correct, runnable code or precise steps.',
      'Be concise. Avoid unnecessary explanations.',
      'If you are uncertain, ask one clarifying question.'
    ])
  return 'Answer concisely and directly.'


def _filter_installed(models: List[str]) -> List[str]:
  out: List[str] = []
  for m in models:
    name = (m or '').strip()
    if not name:
      continue
    lower = name.lower()
    if any(k in lower for k in ['embed', 'embedding', 'rerank', 'reranker']):
      continue
    out.append(name)
  return out


def _pick_candidates(user_input: str, installed: List[str], *, k: int) -> Tuple[List[str], List[Dict[str, Any]]]:
  """
  Deterministic candidate selection so traces are stable.
  Returns (candidates, reasons[]).
  """
  installed = _filter_installed(installed)
  intent = classify_intent(user_input)

  pref_tiny = _env('OLLAMA_TINY_MODEL', 'phi3:mini')
  pref_fast = _env('OLLAMA_FAST_MODEL', 'gemma3:4b')
  pref_general = _env('OLLAMA_GENERAL_MODEL', 'deepseek-r1:8b')
  pref_strong = _env('OLLAMA_STRONG_MODEL', 'deepseek-r1:32b') or pref_general

  reasons: List[Dict[str, Any]] = []
  picked: List[str] = []

  # Quality candidate: largest available (or strong pref if installed).
  quality = _first_existing([pref_strong, pref_general], installed) or _best_by_size(installed)
  if quality:
    picked.append(quality)
    reasons.append({'model': quality, 'why': 'quality:largest_or_strong_pref'})

  # Speed candidate: tiny/fast/general smallest among prefs that exist.
  speed_pool = [m for m in [_first_existing([pref_tiny], installed), _first_existing([pref_fast], installed), _first_existing([pref_general], installed)] if m]
  speed = sorted(speed_pool, key=_parse_model_size_b)[0] if speed_pool else None
  if speed and speed not in picked:
    picked.append(speed)
    reasons.append({'model': speed, 'why': 'speed:smallest_pref'})

  # Specialist candidate for code.
  if intent == 'code':
    coder = _best_by_size(installed, prefer_regex=r'coder|code')
    if coder and coder not in picked:
      picked.append(coder)
      reasons.append({'model': coder, 'why': 'specialist:prefer_coder'})

  # Fill remaining slots by descending size, to keep deterministic.
  if len(picked) < max(int(k), 1):
    for m in sorted(installed, key=_parse_model_size_b, reverse=True):
      if m in picked:
        continue
      picked.append(m)
      reasons.append({'model': m, 'why': 'fill:next_largest'})
      if len(picked) >= max(int(k), 1):
        break

  return picked[:max(int(k), 1)], reasons


def _build_judge_prompt(user_input: str, drafts: List[Dict[str, Any]]) -> str:
  is_math500 = _is_math500_prompt(user_input)
  answers = []
  for idx, d in enumerate(drafts):
    text = (d.get('output') or '').strip()
    if is_math500:
      answers.append(f'Answer_{idx} (model={str(d.get("model") or "")}):\n{text}')
    else:
      answers.append(f'Answer_{idx}:\n{text}')
  notes = []
  if is_math500:
    notes = [
      'Notes (MATH-500):',
      '- Candidate answers are expected to already satisfy the output format constraint.',
      '- Choose based on mathematical correctness/simplest exact form.',
      '- If you are genuinely unsure between answers, prefer the one from the largest/most capable model.',
      ''
    ]
  return '\n\n'.join([
    'You are grading candidate answers from different local models.',
    'Pick the best answer for the user request.',
    '',
    'Rubric:',
    '- Correctness first',
    '- Completeness (meets constraints)',
    '- Clarity and concision',
    '- No hallucinated tool results',
    '- Strictly follow any output formatting constraints in the user request',
    '',
    *notes,
    'Return STRICT JSON only, with this schema:',
    '{"winner": <int>, "reason": "<short>", "scores": {"correctness": <0-10>, "completeness": <0-10>, "clarity": <0-10>}}',
    'Rules:',
    '- winner MUST be an integer index referring to one Answer_i.',
    '- Output MUST be valid JSON only (no markdown, no prose).',
    '',
    f'User request:\n{user_input}',
    '',
    'Candidate answers:',
    *answers
  ])


def _trace(path: str, obj: Dict[str, Any], *, prompt_idx: Optional[int] = None, prompt_text: Optional[str] = None) -> None:
  if not path:
    return
  payload = dict(obj or {})
  if prompt_idx is not None and 'prompt_idx' not in payload:
    payload['prompt_idx'] = int(prompt_idx)
  if prompt_text is not None and isinstance(payload, dict):
    if 'prompt_preview' not in payload or 'prompt_sha256' not in payload:
      payload.update(prompt_fingerprint(prompt_text))
  append_jsonl(path, payload)


def answer_with_selection(
  user_input: str,
  chat_history: List[BaseMessage],
  *,
  force_model: Optional[str] = None,
  trace_path: Optional[str] = None,
  selection_k: Optional[int] = None
) -> Tuple[str, SelectionResult]:
  """
  Agent-owned model selection (no router):
  - deterministic fast-path for simple math expressions
  - tool-needed: run tool-agent and escalate across candidate models
  - no-tools: draft answers from candidates and judge-pick best
  Returns (output, SelectionResult)
  """
  trace_path = trace_path if trace_path is not None else os.environ.get('AGENT_TRACE_PATH', '')
  intent = classify_intent(user_input)
  global _PROMPT_COUNTER
  _PROMPT_COUNTER += 1
  prompt_idx = _PROMPT_COUNTER
  _set_power_label(prompt_idx=prompt_idx, phase='select.start', model='', prompt_text=user_input)

  installed = get_installed_ollama_models(
    ttl_s=float(_env('MODEL_LIST_TTL_S', _env('ROUTER_MODELS_TTL_S', '5')) or '5')
  )

  if force_model:
    selected = SelectionResult(
      prompt_idx=prompt_idx,
      model=force_model,
      temperature=_select_temperature(intent, is_tool=_needs_tools(user_input)),
      system_prompt=_select_system_prompt(intent, is_tool=_needs_tools(user_input)),
      reason='forced_model'
    )
    _set_power_label(prompt_idx=prompt_idx, phase='forced', model=selected.model, prompt_text=user_input)
    _trace(trace_path, {
      'event': 'agent.select.forced',
      'intent': intent,
      'model': selected.model,
      'temperature': selected.temperature
    }, prompt_idx=prompt_idx, prompt_text=user_input)
    output = _run_with_selected(user_input, chat_history, selected, trace_path=trace_path, prompt_idx=prompt_idx)
    _trace(trace_path, {'event': 'agent.select.final', 'model': selected.model, 'output_chars': len(output or '')}, prompt_idx=prompt_idx, prompt_text=user_input)
    return output, selected

  k = int(selection_k or _env('MODEL_SELECT_CANDIDATES', '3') or '3')
  candidates, reasons = _pick_candidates(user_input, installed, k=k)

  _trace(trace_path, {
    'event': 'agent.select.start',
    'intent': intent,
    'needs_tools': bool(_needs_tools(user_input)),
    'installed': [{'name': m, 'size_b': _parse_model_size_b(m)} for m in installed],
    'candidates': candidates,
    'candidate_reasons': reasons
  }, prompt_idx=prompt_idx, prompt_text=user_input)

  # Lookahead policy path: choose a single model without multi-draft + judge.
  # Default to MATH-500 prompts only, since that's what the dataset builder trains on.
  lookahead = LookaheadPolicy.from_env()
  if lookahead and _is_math500_prompt(user_input) and not _needs_tools(user_input):
    chosen_model, scored = lookahead.select_model(prompt=user_input, candidates=list(candidates))
    if chosen_model:
      _set_power_label(prompt_idx=prompt_idx, phase='lookahead.select', model=chosen_model, prompt_text=user_input)
      _trace(trace_path, {
        'event': 'agent.policy.lookahead.select',
        'phase': 'lookahead.select',
        'candidates': list(candidates),
        'chosen_model': chosen_model,
        'scored': lookahead.to_json(scored, chosen=chosen_model)
      }, prompt_idx=prompt_idx, prompt_text=user_input)

      sys_prompt = _select_system_prompt(intent, is_tool=False)
      temp = _select_temperature(intent, is_tool=False)
      draft_timeout_s = float(_env('MODEL_SELECT_DRAFT_TIMEOUT_S', '180') or '180')

      _set_power_label(prompt_idx=prompt_idx, phase='lookahead.run', model=chosen_model, prompt_text=user_input)
      r = _invoke_chat_with_timeout(
        model=chosen_model,
        temperature=temp,
        system_prompt=sys_prompt,
        chat_history=list(chat_history),
        user_input=user_input,
        timeout_s=draft_timeout_s
      )
      if r.get('ok'):
        text = str(r.get('text') or '').strip()
        canon = _canonicalize_math500_answer(text)
        if canon:
          text = canon
          _trace(trace_path, {
            'event': 'agent.policy.lookahead.run',
            'phase': 'lookahead.run',
            'model': chosen_model,
            'duration_ms': int(r.get('duration_ms') or 0),
            'output_chars': len(text or '')
          }, prompt_idx=prompt_idx, prompt_text=user_input)
          _set_power_label(prompt_idx=prompt_idx, phase='final', model=chosen_model, prompt_text=user_input)
          _trace(trace_path, {'event': 'agent.select.final', 'model': chosen_model, 'output_chars': len(text or ''), 'phase': 'final', 'reason': 'lookahead'}, prompt_idx=prompt_idx, prompt_text=user_input)
          return text, SelectionResult(prompt_idx=prompt_idx, model=chosen_model, temperature=temp, system_prompt=sys_prompt, reason='lookahead')

      _trace(trace_path, {
        'event': 'agent.policy.lookahead.fallback',
        'phase': 'lookahead.fallback',
        'chosen_model': chosen_model,
        'ok': bool(r.get('ok')),
        'timed_out': bool(r.get('timed_out')),
        'error': str(r.get('error') or ''),
        'duration_ms': int(r.get('duration_ms') or 0)
      }, prompt_idx=prompt_idx, prompt_text=user_input)

  # Non-tool: draft + judge.
  sys_prompt = _select_system_prompt(intent, is_tool=False)
  temp = _select_temperature(intent, is_tool=False)

  max_chars = int(_env('MODEL_SELECT_MAX_OUTPUT_CHARS', '2000') or '2000')
  drafts: List[Dict[str, Any]] = []
  # Always try drafting from smallest -> largest so progress happens even if a big model hangs.
  draft_models = sorted(list(candidates), key=_parse_model_size_b)
  draft_timeout_s = float(_env('MODEL_SELECT_DRAFT_TIMEOUT_S', '180') or '180')

  for i, m in enumerate(draft_models):
    _set_power_label(prompt_idx=prompt_idx, phase=f'draft_{i}', model=m, prompt_text=user_input)
    _trace(trace_path, {'event': 'agent.select.draft_attempt', 'phase': f'draft_{i}', 'model': m}, prompt_idx=prompt_idx, prompt_text=user_input)
    r = _invoke_chat_with_timeout(
      model=m,
      temperature=temp,
      system_prompt=sys_prompt,
      chat_history=list(chat_history),
      user_input=user_input,
      timeout_s=draft_timeout_s
    )
    if not r.get('ok'):
      _trace(trace_path, {
        'event': 'agent.select.draft_failed',
        'phase': f'draft_{i}',
        'model': m,
        'duration_ms': int(r.get('duration_ms') or 0),
        'timed_out': bool(r.get('timed_out')),
        'error': str(r.get('error') or '')
      }, prompt_idx=prompt_idx, prompt_text=user_input)
      continue

    text = str(r.get('text') or '').strip()
    if _is_math500_prompt(user_input):
      canon = _canonicalize_math500_answer(text)
      if not canon:
        _trace(trace_path, {
          'event': 'agent.select.draft_failed',
          'phase': f'draft_{i}',
          'model': m,
          'duration_ms': int(r.get('duration_ms') or 0),
          'timed_out': False,
          'error': 'missing_boxed'
        }, prompt_idx=prompt_idx, prompt_text=user_input)
        continue
      text = canon
    draft = {
      'model': m,
      'duration_ms': int(r.get('duration_ms') or 0),
      'output': text,
      'output_chars': len(text or ''),
      'output_trunc': (text or '')[:max_chars]
    }
    drafts.append(draft)
    _trace(trace_path, {
      'event': 'agent.select.draft',
      'phase': f'draft_{i}',
      **{k: v for k, v in draft.items() if k != 'output'}
    }, prompt_idx=prompt_idx, prompt_text=user_input)

  if not drafts:
    _set_power_label(prompt_idx=prompt_idx, phase='final', model='', prompt_text=user_input)
    _trace(trace_path, {'event': 'agent.select.final', 'model': '', 'output_chars': 0, 'phase': 'final', 'reason': 'all_drafts_failed'}, prompt_idx=prompt_idx, prompt_text=user_input)
    return '', SelectionResult(prompt_idx=prompt_idx, model='', temperature=temp, system_prompt=sys_prompt, reason='all_drafts_failed')

  judge_temp = _clamp_temperature(_env('MODEL_SELECT_JUDGE_TEMP', '0.0'), 0.0)
  judge_timeout_s = float(_env('MODEL_SELECT_JUDGE_TIMEOUT_S', '90') or '90')

  # Judge failover: if the heaviest judge hangs, try the next-largest judge.
  preferred_judge = (_env('OLLAMA_JUDGE_MODEL', '') or '').strip()
  judge_pool = sorted(_filter_installed(list(installed or [])), key=_parse_model_size_b, reverse=True)
  judge_models: List[str] = []
  if preferred_judge:
    judge_models.append(preferred_judge)
  for jm in judge_pool:
    if jm not in judge_models:
      judge_models.append(jm)

  judge_prompt = _build_judge_prompt(user_input, drafts)
  judged_raw = ''
  judged_parsed: Dict[str, Any] = {}
  judged_model_used = ''
  judged_duration_ms = 0

  for jm in judge_models[: max(len(judge_models), 1)]:
    _set_power_label(prompt_idx=prompt_idx, phase='judge', model=jm, prompt_text=user_input)
    for attempt in range(2):
      system_prompt = 'Return strict JSON only.'
      if attempt == 1:
        system_prompt = '\n'.join([
          'Return VALID JSON only.',
          'No markdown, no prose.',
          'winner MUST be an integer index that refers to one Answer_i.'
        ])

      _trace(trace_path, {
        'event': 'agent.select.judge_attempt',
        'phase': 'judge',
        'judge_model': jm,
        'attempt': attempt + 1
      }, prompt_idx=prompt_idx, prompt_text=user_input)

      rj = _invoke_chat_with_timeout(
        model=jm,
        temperature=judge_temp,
        system_prompt=system_prompt,
        chat_history=[],
        user_input=judge_prompt,
        timeout_s=judge_timeout_s
      )
      judged_duration_ms = int(rj.get('duration_ms') or 0)

      if not rj.get('ok'):
        _trace(trace_path, {
          'event': 'agent.select.judge_failed',
          'phase': 'judge',
          'judge_model': jm,
          'attempt': attempt + 1,
          'duration_ms': judged_duration_ms,
          'timed_out': bool(rj.get('timed_out')),
          'error': str(rj.get('error') or '')
        }, prompt_idx=prompt_idx, prompt_text=user_input)
        break

      judged_raw = str(rj.get('text') or '').strip()
      parsed = _try_parse_judge_json(judged_raw)
      if parsed is None:
        _trace(trace_path, {
          'event': 'agent.select.judge_failed',
          'phase': 'judge',
          'judge_model': jm,
          'attempt': attempt + 1,
          'duration_ms': judged_duration_ms,
          'timed_out': False,
          'error': 'invalid_judge_json'
        }, prompt_idx=prompt_idx, prompt_text=user_input)
        continue

      winner = int(parsed.get('winner') or 0)
      if winner < 0 or winner >= len(drafts):
        _trace(trace_path, {
          'event': 'agent.select.judge_failed',
          'phase': 'judge',
          'judge_model': jm,
          'attempt': attempt + 1,
          'duration_ms': judged_duration_ms,
          'timed_out': False,
          'error': 'winner_out_of_range'
        }, prompt_idx=prompt_idx, prompt_text=user_input)
        continue

      judged_parsed = parsed
      judged_model_used = jm
      break

    if judged_model_used:
      break

  if not judged_model_used:
    # No judge succeeded: pick the best draft by format/quality heuristic.
    best = _pick_best_draft(user_input, drafts) or drafts[0]
    chosen_model = str(best.get('model') or '')
    output = str(best.get('output') or '').strip()
    _set_power_label(prompt_idx=prompt_idx, phase='final', model=chosen_model, prompt_text=user_input)
    _trace(trace_path, {
      'event': 'agent.select.judge',
      'phase': 'judge',
      'judge_model': '',
      'duration_ms': 0,
      'judge_raw': '',
      'judge_parsed': {},
      'winner': 0,
      'chosen_model': chosen_model,
      'reason': 'judge_all_failed_pick_best_draft'
    }, prompt_idx=prompt_idx, prompt_text=user_input)
    _trace(trace_path, {'event': 'agent.select.final', 'model': chosen_model, 'output_chars': len(output or ''), 'phase': 'final'}, prompt_idx=prompt_idx, prompt_text=user_input)
    return output, SelectionResult(prompt_idx=prompt_idx, model=chosen_model, temperature=temp, system_prompt=sys_prompt, reason='judge_all_failed_pick_best_draft')

  winner = int(judged_parsed.get('winner') or 0)
  if winner < 0 or winner >= len(drafts):
    winner = 0

  chosen = drafts[winner]
  chosen_model = str(chosen.get('model') or '')
  reason = str(judged_parsed.get('reason') or 'judge_pick').strip()

  # Guardrail: if formatting constraints exist (e.g., \\boxed) and judge chose a worse-formatted draft,
  # override to the best formatted draft.
  best = _pick_best_draft(user_input, drafts)
  if best:
    best_model = str(best.get('model') or '')
    chosen_score = _draft_format_score(user_input, str(chosen.get('output') or ''))
    best_score = _draft_format_score(user_input, str(best.get('output') or ''))
    if best_score > chosen_score and best_model != chosen_model:
      _trace(trace_path, {
        'event': 'agent.select.override',
        'phase': 'override',
        'from_model': chosen_model,
        'to_model': best_model,
        'from_score': list(chosen_score),
        'to_score': list(best_score),
        'reason': 'format_guardrail'
      }, prompt_idx=prompt_idx, prompt_text=user_input)
      chosen = best
      chosen_model = best_model
      reason = (reason + ' | override:format_guardrail').strip()

  _trace(trace_path, {
    'event': 'agent.select.judge',
    'phase': 'judge',
    'judge_model': judged_model_used,
    'duration_ms': judged_duration_ms,
    'judge_raw': judged_raw[:max_chars],
    'judge_parsed': judged_parsed,
    'winner': winner,
    'chosen_model': chosen_model,
    'reason': reason
  }, prompt_idx=prompt_idx, prompt_text=user_input)

  output = str(chosen.get('output') or '').strip()
  _set_power_label(prompt_idx=prompt_idx, phase='final', model=chosen_model, prompt_text=user_input)
  _trace(trace_path, {'event': 'agent.select.final', 'model': chosen_model, 'output_chars': len(output or ''), 'phase': 'final'}, prompt_idx=prompt_idx, prompt_text=user_input)
  return output, SelectionResult(prompt_idx=prompt_idx, model=chosen_model, temperature=temp, system_prompt=sys_prompt, reason=reason)


def _run_with_selected(
  user_input: str,
  chat_history: List[BaseMessage],
  selected: SelectionResult,
  *,
  trace_path: str,
  prompt_idx: Optional[int] = None
) -> str:
  """
  Call the chat model directly (tool-agent removed in slim build).
  """
  llm = build_llm(model=selected.model, temperature=selected.temperature)
  messages: List[BaseMessage] = [
    SystemMessage(content=selected.system_prompt),
    *list(chat_history),
    HumanMessage(content=user_input)
  ]
  start = time.time()
  res = llm.invoke(messages)
  out = (getattr(res, 'content', None) or str(res)).strip()
  _trace(trace_path, {
    'event': 'agent.run.chat',
    'phase': 'chat.run',
    'model': selected.model,
    'duration_ms': int((time.time() - start) * 1000),
    'output_chars': len(out or '')
  }, prompt_idx=prompt_idx, prompt_text=user_input)
  return out

