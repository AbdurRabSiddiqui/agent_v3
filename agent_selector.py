from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent import build_agent_executor, build_llm
from ollama_utils import env as _env
from ollama_utils import get_installed_ollama_models, is_truthy
from specialists import try_math_expression
from trace_utils import append_jsonl, prompt_fingerprint, write_json


@dataclass(frozen=True)
class SelectionResult:
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
  text = (user_input or '').lower()
  if any(k in text for k in ['list files', 'list_dir', 'read file', 'read_text_file', 'write file', 'append file']):
    return True
  if any(k in text for k in ['what time', 'time utc', 'get_time_utc']):
    return True
  if re.search(r'\bcalculate\b|\bcompute\b|\bevaluate\b', text) and re.search(r'[\d\+\-\*\/\^\(\)]', text):
    return True
  if any(k in text for k in ['rate plan', 'kwh', 'tariff', 'net metering']):
    return True
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


def _judge_pick(user_input: str, drafts: List[Dict[str, Any]], *, judge_model: str, temperature: float) -> Dict[str, Any]:
  """
  Ask a judge model to pick the best draft. Returns parsed JSON with winner index.
  """
  # Anonymize model names in the judge prompt.
  answers = []
  for idx, d in enumerate(drafts):
    text = (d.get('output') or '').strip()
    answers.append(f'Answer_{idx}:\n{text}')

  prompt = '\n\n'.join([
    'You are grading candidate answers from different local models.',
    'Pick the best answer for the user request.',
    '',
    'Rubric:',
    '- Correctness first',
    '- Completeness (meets constraints)',
    '- Clarity and concision',
    '- No hallucinated tool results',
    '',
    'Return STRICT JSON only, with this schema:',
    '{"winner": <int>, "reason": "<short>", "scores": {"correctness": <0-10>, "completeness": <0-10>, "clarity": <0-10>}}',
    '',
    f'User request:\n{user_input}',
    '',
    'Candidate answers:',
    *answers
  ])

  start = time.time()
  llm = build_llm(model=judge_model, temperature=temperature)
  res = llm.invoke([SystemMessage(content='Return strict JSON only.'), HumanMessage(content=prompt)])
  raw = (getattr(res, 'content', None) or str(res)).strip()
  duration_ms = int((time.time() - start) * 1000)

  parsed: Dict[str, Any] = {}
  try:
    parsed = json.loads(raw)
  except Exception:
    # last-resort: try to extract a JSON object from the text
    m = re.search(r'({[\s\S]*})', raw)
    if m:
      try:
        parsed = json.loads(m.group(1))
      except Exception:
        parsed = {}

  winner = parsed.get('winner')
  if not isinstance(winner, int):
    parsed['winner'] = 0
    parsed['reason'] = (parsed.get('reason') or 'invalid_judge_json_fallback').strip()

  return {'raw': raw, 'parsed': parsed, 'duration_ms': duration_ms}


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

  if intent == 'math':
    fast = try_math_expression(user_input)
    if fast is not None:
      _set_power_label(prompt_idx=prompt_idx, phase='fast_path', model='', prompt_text=user_input)
      _trace(trace_path, {
        'event': 'agent.select.fast_path',
        'intent': intent,
        'kind': 'math_expression',
        'model': None
      }, prompt_idx=prompt_idx, prompt_text=user_input)
      _trace(trace_path, {
        'event': 'agent.select.final',
        'model': None,
        'output_chars': len(fast or '')
      }, prompt_idx=prompt_idx, prompt_text=user_input)
      return fast, SelectionResult(model='', temperature=0.0, system_prompt='', reason='fast_path:math_expression')

  installed = get_installed_ollama_models(
    ttl_s=float(_env('MODEL_LIST_TTL_S', _env('ROUTER_MODELS_TTL_S', '5')) or '5')
  )

  if force_model:
    selected = SelectionResult(
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

  if _needs_tools(user_input):
    # Prefer smaller first, then larger: escalate on failure/stops.
    by_size = sorted(candidates, key=_parse_model_size_b)
    last_error = ''
    for m in by_size:
      _set_power_label(prompt_idx=prompt_idx, phase='tools.attempt', model=m, prompt_text=user_input)
      selected = SelectionResult(
        model=m,
        temperature=_select_temperature(intent, is_tool=True),
        system_prompt=_select_system_prompt(intent, is_tool=True),
        reason='tools:attempt'
      )
      _trace(trace_path, {'event': 'agent.select.tools.attempt', 'model': m, 'phase': 'tools.attempt'}, prompt_idx=prompt_idx, prompt_text=user_input)
      out = _run_with_selected(user_input, chat_history, selected, trace_path=trace_path, prompt_idx=prompt_idx)
      if out and 'Agent stopped due to iteration limit or time limit.' not in out and 'Agent stopped:' not in out:
        _set_power_label(prompt_idx=prompt_idx, phase='final', model=m, prompt_text=user_input)
        _trace(trace_path, {'event': 'agent.select.tools.success', 'model': m, 'output_chars': len(out or ''), 'phase': 'tools.success'}, prompt_idx=prompt_idx, prompt_text=user_input)
        _trace(trace_path, {'event': 'agent.select.final', 'model': m, 'output_chars': len(out or '')}, prompt_idx=prompt_idx, prompt_text=user_input)
        return out, SelectionResult(
          model=m,
          temperature=selected.temperature,
          system_prompt=selected.system_prompt,
          reason='tools:success'
        )
      last_error = out or last_error
      _trace(trace_path, {'event': 'agent.select.tools.escalate', 'model': m, 'stop_output': (out or '')[:400], 'phase': 'tools.escalate'}, prompt_idx=prompt_idx, prompt_text=user_input)

    # Fall back to a non-tool tournament if tools kept failing.
    _trace(trace_path, {'event': 'agent.select.tools.fallback_to_qa', 'last': (last_error or '')[:400], 'phase': 'tools.fallback_to_qa'}, prompt_idx=prompt_idx, prompt_text=user_input)

  # Non-tool: draft + judge.
  sys_prompt = _select_system_prompt(intent, is_tool=False)
  temp = _select_temperature(intent, is_tool=False)

  max_chars = int(_env('MODEL_SELECT_MAX_OUTPUT_CHARS', '2000') or '2000')
  drafts: List[Dict[str, Any]] = []
  for i, m in enumerate(candidates):
    _set_power_label(prompt_idx=prompt_idx, phase=f'draft_{i}', model=m, prompt_text=user_input)
    start = time.time()
    llm = build_llm(model=m, temperature=temp)
    messages: List[BaseMessage] = [SystemMessage(content=sys_prompt), *list(chat_history), HumanMessage(content=user_input)]
    res = llm.invoke(messages)
    text = (getattr(res, 'content', None) or str(res)).strip()
    duration_ms = int((time.time() - start) * 1000)
    draft = {
      'model': m,
      'duration_ms': duration_ms,
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

  judge_model = _env('OLLAMA_JUDGE_MODEL', '') or _best_by_size(installed) or (candidates[0] if candidates else '')
  judge_temp = _clamp_temperature(_env('MODEL_SELECT_JUDGE_TEMP', '0.0'), 0.0)

  _set_power_label(prompt_idx=prompt_idx, phase='judge', model=judge_model, prompt_text=user_input)
  judged = _judge_pick(user_input, drafts, judge_model=judge_model, temperature=judge_temp)
  parsed = judged.get('parsed') or {}
  winner = int(parsed.get('winner') or 0)
  if winner < 0 or winner >= len(drafts):
    winner = 0

  chosen = drafts[winner] if drafts else {'model': candidates[0] if candidates else ''}
  chosen_model = chosen.get('model') or (candidates[0] if candidates else '')
  reason = str(parsed.get('reason') or 'judge_pick').strip()

  _trace(trace_path, {
    'event': 'agent.select.judge',
    'phase': 'judge',
    'judge_model': judge_model,
    'duration_ms': int(judged.get('duration_ms') or 0),
    'judge_raw': (judged.get('raw') or '')[:max_chars],
    'judge_parsed': parsed,
    'winner': winner,
    'chosen_model': chosen_model,
    'reason': reason
  }, prompt_idx=prompt_idx, prompt_text=user_input)

  output = (drafts[winner].get('output') or '').strip() if drafts else ''
  _set_power_label(prompt_idx=prompt_idx, phase='final', model=chosen_model, prompt_text=user_input)
  _trace(trace_path, {'event': 'agent.select.final', 'model': chosen_model, 'output_chars': len(output or ''), 'phase': 'final'}, prompt_idx=prompt_idx, prompt_text=user_input)
  return output, SelectionResult(model=chosen_model, temperature=temp, system_prompt=sys_prompt, reason=reason)


def _run_with_selected(
  user_input: str,
  chat_history: List[BaseMessage],
  selected: SelectionResult,
  *,
  trace_path: str,
  prompt_idx: Optional[int] = None
) -> str:
  """
  If tools are needed, run the tool-agent loop. Otherwise call the chat model directly.
  """
  if _needs_tools(user_input):
    executor = build_agent_executor(model=selected.model, temperature=selected.temperature, verbose=False)
    start = time.time()
    result = executor.invoke({'input': user_input, 'chat_history': chat_history})
    out = (result.get('output', '') or '').strip()
    _trace(trace_path, {
      'event': 'agent.run.tools',
      'phase': 'tools.run',
      'model': selected.model,
      'duration_ms': int((time.time() - start) * 1000),
      'output_chars': len(out or '')
    }, prompt_idx=prompt_idx, prompt_text=user_input)
    return out

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

