from __future__ import annotations

import json
import os
import re
import time
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

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


def _serialize_history(chat_history: List[BaseMessage]) -> list[dict]:
  out: list[dict] = []
  for m in list(chat_history or []):
    content = getattr(m, 'content', None)
    if not isinstance(content, str):
      content = str(content)
    role = (getattr(m, 'type', None) or m.__class__.__name__ or '').lower()
    if 'human' in role:
      kind = 'human'
    elif 'ai' in role:
      kind = 'ai'
    elif 'system' in role:
      kind = 'system'
    else:
      kind = 'other'
    out.append({'kind': kind, 'content': content})
  return out


def _invoke_chat_worker(payload: dict, out_q):
  """
  Separate process worker so we can timeout/terminate safely if Ollama hangs.
  Payload keys: model, temperature, system_prompt, history(list), user_input
  """
  try:
    # Import inside worker for spawn safety.
    from agent import build_llm as _build_llm  # type: ignore
    from langchain_core.messages import SystemMessage as _SystemMessage  # type: ignore
    from langchain_core.messages import HumanMessage as _HumanMessage  # type: ignore
    from langchain_core.messages import AIMessage as _AIMessage  # type: ignore

    model = str(payload.get('model') or '')
    temperature = float(payload.get('temperature') or 0.2)
    system_prompt = str(payload.get('system_prompt') or '')
    history = payload.get('history') or []
    user_input = str(payload.get('user_input') or '')

    llm = _build_llm(model=model, temperature=temperature)
    messages = [_SystemMessage(content=system_prompt)] if system_prompt else []
    for h in history:
      if not isinstance(h, dict):
        continue
      kind = str(h.get('kind') or '')
      content = str(h.get('content') or '')
      if kind == 'human':
        messages.append(_HumanMessage(content=content))
      elif kind == 'ai':
        messages.append(_AIMessage(content=content))
      elif kind == 'system':
        messages.append(_SystemMessage(content=content))
    messages.append(_HumanMessage(content=user_input))

    res = llm.invoke(messages)
    text = (getattr(res, 'content', None) or str(res)).strip()
    out_q.put({'ok': True, 'text': text})
  except Exception as err:
    out_q.put({'ok': False, 'error': str(err), 'traceback': traceback.format_exc()})


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
  Returns dict: { ok, text, duration_ms, timed_out, error? }
  """
  start = time.time()
  ctx = mp.get_context('spawn')
  q = ctx.Queue(maxsize=1)
  p = ctx.Process(
    target=_invoke_chat_worker,
    args=({
      'model': model,
      'temperature': temperature,
      'system_prompt': system_prompt,
      'history': _serialize_history(chat_history),
      'user_input': user_input
    }, q)
  )
  p.start()
  p.join(timeout=max(float(timeout_s), 0.1))
  if p.is_alive():
    try:
      p.terminate()
    except Exception:
      pass
    p.join(timeout=2)
    return {
      'ok': False,
      'timed_out': True,
      'duration_ms': int((time.time() - start) * 1000),
      'error': f'timeout_after_{timeout_s}s'
    }
  try:
    msg = q.get_nowait()
  except Exception:
    msg = {'ok': False, 'error': 'no_result_from_worker'}
  duration_ms = int((time.time() - start) * 1000)
  if not isinstance(msg, dict) or not msg.get('ok'):
    return {
      'ok': False,
      'timed_out': False,
      'duration_ms': duration_ms,
      'error': str((msg or {}).get('error') or 'invoke_failed')
    }
  return {
    'ok': True,
    'timed_out': False,
    'duration_ms': duration_ms,
    'text': str(msg.get('text') or '')
  }


def _invoke_tool_worker(payload: dict, out_q):
  """
  Separate process worker so we can timeout/terminate safely if the tool-agent hangs.
  Payload keys: model, temperature, history(list), user_input
  """
  try:
    from agent import build_agent_executor as _build_agent_executor  # type: ignore
    from langchain_core.messages import HumanMessage as _HumanMessage  # type: ignore
    from langchain_core.messages import AIMessage as _AIMessage  # type: ignore
    from langchain_core.messages import SystemMessage as _SystemMessage  # type: ignore

    model = str(payload.get('model') or '')
    temperature = float(payload.get('temperature') or 0.2)
    history = payload.get('history') or []
    user_input = str(payload.get('user_input') or '')

    chat_history: list = []
    for h in history:
      if not isinstance(h, dict):
        continue
      kind = str(h.get('kind') or '')
      content = str(h.get('content') or '')
      if kind == 'human':
        chat_history.append(_HumanMessage(content=content))
      elif kind == 'ai':
        chat_history.append(_AIMessage(content=content))
      elif kind == 'system':
        chat_history.append(_SystemMessage(content=content))

    executor = _build_agent_executor(model=model, temperature=temperature, verbose=False)
    result = executor.invoke({'input': user_input, 'chat_history': chat_history})
    out = (result.get('output', '') or '').strip()
    out_q.put({'ok': True, 'text': out})
  except Exception as err:
    out_q.put({'ok': False, 'error': str(err), 'traceback': traceback.format_exc()})


def _invoke_tool_with_timeout(
  *,
  model: str,
  temperature: float,
  chat_history: List[BaseMessage],
  user_input: str,
  timeout_s: float
) -> dict:
  """
  Returns dict: { ok, text, duration_ms, timed_out, error? }
  """
  start = time.time()
  ctx = mp.get_context('spawn')
  q = ctx.Queue(maxsize=1)
  p = ctx.Process(
    target=_invoke_tool_worker,
    args=({
      'model': model,
      'temperature': temperature,
      'history': _serialize_history(chat_history),
      'user_input': user_input
    }, q)
  )
  p.start()
  p.join(timeout=max(float(timeout_s), 0.1))
  if p.is_alive():
    try:
      p.terminate()
    except Exception:
      pass
    p.join(timeout=2)
    return {
      'ok': False,
      'timed_out': True,
      'duration_ms': int((time.time() - start) * 1000),
      'error': f'tool_timeout_after_{timeout_s}s'
    }
  try:
    msg = q.get_nowait()
  except Exception:
    msg = {'ok': False, 'error': 'no_result_from_tool_worker'}
  duration_ms = int((time.time() - start) * 1000)
  if not isinstance(msg, dict) or not msg.get('ok'):
    return {
      'ok': False,
      'timed_out': False,
      'duration_ms': duration_ms,
      'error': str((msg or {}).get('error') or 'tool_invoke_failed')
    }
  return {
    'ok': True,
    'timed_out': False,
    'duration_ms': duration_ms,
    'text': str(msg.get('text') or '')
  }


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


def _build_judge_prompt(user_input: str, drafts: List[Dict[str, Any]]) -> str:
  answers = []
  for idx, d in enumerate(drafts):
    text = (d.get('output') or '').strip()
    answers.append(f'Answer_{idx}:\n{text}')
  return '\n\n'.join([
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


def _parse_judge_json(raw: str) -> Dict[str, Any]:
  parsed: Dict[str, Any] = {}
  try:
    parsed = json.loads(raw)
  except Exception:
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
  return parsed


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
      if out and 'tool_timeout_after_' not in out and 'Agent stopped due to iteration limit or time limit.' not in out and 'Agent stopped:' not in out:
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
    return '', SelectionResult(model='', temperature=temp, system_prompt=sys_prompt, reason='all_drafts_failed')

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
    _trace(trace_path, {'event': 'agent.select.judge_attempt', 'phase': 'judge', 'judge_model': jm}, prompt_idx=prompt_idx, prompt_text=user_input)
    rj = _invoke_chat_with_timeout(
      model=jm,
      temperature=judge_temp,
      system_prompt='Return strict JSON only.',
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
        'duration_ms': judged_duration_ms,
        'timed_out': bool(rj.get('timed_out')),
        'error': str(rj.get('error') or '')
      }, prompt_idx=prompt_idx, prompt_text=user_input)
      continue

    judged_raw = str(rj.get('text') or '').strip()
    judged_parsed = _parse_judge_json(judged_raw)
    judged_model_used = jm
    break

  if not judged_model_used:
    # No judge succeeded: pick the largest successful draft as a deterministic fallback.
    best = sorted(drafts, key=lambda d: _parse_model_size_b(str(d.get('model') or '')), reverse=True)[0]
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
      'reason': 'judge_all_failed_pick_largest'
    }, prompt_idx=prompt_idx, prompt_text=user_input)
    _trace(trace_path, {'event': 'agent.select.final', 'model': chosen_model, 'output_chars': len(output or ''), 'phase': 'final'}, prompt_idx=prompt_idx, prompt_text=user_input)
    return output, SelectionResult(model=chosen_model, temperature=temp, system_prompt=sys_prompt, reason='judge_all_failed_pick_largest')

  winner = int(judged_parsed.get('winner') or 0)
  if winner < 0 or winner >= len(drafts):
    winner = 0

  chosen = drafts[winner]
  chosen_model = str(chosen.get('model') or '')
  reason = str(judged_parsed.get('reason') or 'judge_pick').strip()

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
    timeout_s = float(_env('MODEL_SELECT_TOOL_TIMEOUT_S', '240') or '240')
    r = _invoke_tool_with_timeout(
      model=selected.model,
      temperature=selected.temperature,
      chat_history=list(chat_history),
      user_input=user_input,
      timeout_s=timeout_s
    )
    out = str(r.get('text') or '').strip()
    _trace(trace_path, {
      'event': 'agent.run.tools',
      'phase': 'tools.run',
      'model': selected.model,
      'duration_ms': int(r.get('duration_ms') or 0),
      'timed_out': bool(r.get('timed_out')),
      'ok': bool(r.get('ok')),
      'output_chars': len(out or ''),
      'error': str(r.get('error') or '')
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

