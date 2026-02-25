from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ollama_utils import env as _env
from ollama_utils import get_installed_ollama_models
from ollama_utils import is_truthy as _is_truthy
from trace_utils import append_jsonl


@dataclass(frozen=True)
class SelectedModel:
  provider: str
  model: str
  temperature: float = 0.2
  system_prompt: str = ''
  reason: str = ''


def _get_installed_ollama_models() -> List[str]:
  return get_installed_ollama_models(ttl_s=float(_env('ROUTER_MODELS_TTL_S', '5') or '5'))


def _parse_model_size_b(model: str) -> float:
  """
  Heuristic: parse parameter count from model tag like:
  - deepseek-r1:8b -> 8
  - gemma3:27b -> 27
  - deepseek-v3.1:671b-cloud -> 671
  - gpt-oss:120b-cloud -> 120
  - phi3:mini -> 1
  """
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


def classify_intent(user_input: str) -> str:
  text = (user_input or '').lower()

  code_markers = ['```', 'stack trace', 'traceback', 'exception', 'error:', 'compile', 'typescript', 'javascript']
  if any(m in text for m in code_markers):
    return 'code'
  if re.search(r'\b(def|class|import|from)\b', text) and re.search(r'[:\(\)]', text):
    return 'code'

  # Detect common 24-hour usage profiles (rate plan tasks) to avoid misclassifying as math.
  if re.search(r'\[[\s\d\.,]+\]', text) and text.count(',') >= 15:
    if any(k in text for k in ['kwh', 'solar', 'rate plan', 'tariff', 'net metering', 'off-peak', 'on-peak']):
      return 'general'

  power_keywords = [
    'rate plan', 'tou', 'time-of-use', 'time of use', 'kwh',
    'net metering', 'export', 'import', 'solar', 'panel', 'utility',
    'bill', 'tariff', 'demand charge', 'off-peak', 'on-peak', 'peak'
  ]
  if any(k in text for k in power_keywords):
    return 'general'

  math_keywords = [
    # common math-problem verbs
    'find', 'determine', 'evaluate', 'compute', 'solve', 'simplify',
    'prove', 'show that',
    # common math terms
    'solve', 'simplify', 'integral', 'derivative', 'equation', 'factor',
    'probability', 'geometry', 'mod', 'congruent', 'polynomial', 'matrix',
    'log', 'sin', 'cos', 'tan',
    # common MATH benchmark phrasing
    'value of', 'how many', 'greatest', 'least', 'integer', 'real number', 'positive'
  ]
  if any(k in text for k in math_keywords):
    return 'math'

  # latex-ish patterns common in MATH-500
  if any(k in text for k in ['\\frac', '\\sqrt', '\\cdot', '\\pi', '\\theta', '\\angle', '\\triangle']):
    return 'math'

  # variable/algebra patterns even without digits
  if re.search(r'\b(x|y|z|n|k|m)\b', text) and re.search(r'[=\^\+\-\*\/\(\)]', text):
    return 'math'

  if re.search(r'[\d][\d\s\+\-\*\/\^\(\)\.]+', text):
    return 'math'

  return 'general'


def _needs_tools(user_input: str) -> bool:
  """
  Heuristic: does the task likely require tools (files/time/calculator/rate tools)?
  This helps pick a stronger "tool model" only when necessary.
  """
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


def _estimate_complexity(user_input: str) -> int:
  """
  0=trivial, 1=small, 2=medium, 3=hard
  Used for power-efficient "smallest sufficient model" selection.
  """
  text = user_input or ''
  lower = text.lower()
  score = 0

  if len(text) >= 900:
    score += 2
  elif len(text) >= 350:
    score += 1

  if lower.count('\n') >= 12:
    score += 1

  if 'prove' in lower or 'show that' in lower:
    score += 2

  if 'benchmark' in lower or 'evaluate on' in lower or 'math-500' in lower:
    score += 1

  code_fences = lower.count('```')
  if code_fences >= 2:
    score += 1

  # lots of symbols often means harder reasoning / longer context
  symbol_hits = len(re.findall(r'[\{\}\[\]\(\)=<>\\\^_]', text))
  if symbol_hits >= 80:
    score += 1

  if score <= 0:
    return 0
  if score == 1:
    return 1
  if score == 2:
    return 2
  return 3


def _pick_smallest_sufficient(installed: List[str], candidates: List[str]) -> Optional[str]:
  if not installed:
    return None
  installed_set = set(installed)
  pool = [c for c in candidates if c and c in installed_set]
  if not pool:
    return None
  # sort ascending: smallest first for power efficiency
  return sorted(pool, key=_parse_model_size_b)[0]


def _pick_tool_model(user_input: str, installed: List[str], tool_pref: str, strong_pref: str) -> tuple[str, str]:
  """
  Pick a tool model for the agent loop.
  - trivial/simple tool use -> prefer smaller (tool/tiny/fast)
  - complex multi-step tool use -> prefer tool/strong
  Returns (model, reason)
  """
  complexity = _estimate_complexity(user_input)
  needs_tools = _needs_tools(user_input)

  if not needs_tools and complexity <= 1:
    model = _pick_smallest_sufficient(installed, [tool_pref, _env('OLLAMA_FAST_MODEL', ''), _env('OLLAMA_TINY_MODEL', '')]) or tool_pref
    return model, f'tool:low (complexity={complexity}, needs_tools={needs_tools})'

  if complexity >= 2:
    model = _first_existing([tool_pref, strong_pref], installed) or strong_pref
    return model, f'tool:high (complexity={complexity}, needs_tools={needs_tools})'

  model = _first_existing([tool_pref, _env('OLLAMA_FAST_MODEL', ''), strong_pref], installed) or tool_pref
  return model, f'tool:mid (complexity={complexity}, needs_tools={needs_tools})'


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


def _select_temperature(intent: str, *, is_tool: bool) -> float:
  if is_tool:
    return _clamp_temperature(_env('ROUTER_TEMP_TOOL', '0.1'), 0.1)
  if intent == 'math':
    return _clamp_temperature(_env('ROUTER_TEMP_MATH', '0.2'), 0.2)
  if intent == 'code':
    return _clamp_temperature(_env('ROUTER_TEMP_CODE', '0.2'), 0.2)
  return _clamp_temperature(_env('ROUTER_TEMP_GENERAL', '0.3'), 0.3)


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


def _is_hard_math(user_input: str) -> bool:
  text = (user_input or '').lower()
  if len(text) > 200:
    return True
  hard_markers = ['proof', 'show that', 'let', 'suppose', 'for all', 'exists', 'therefore']
  if any(m in text for m in hard_markers):
    return True
  if any(m in text for m in ['integral', 'derivative', 'system of equations', 'diophantine']):
    return True
  return False


def _estimate_math500_difficulty(user_input: str) -> int:
  """
  Heuristic difficulty score for MATH-500 style questions.
  Higher score => prefer larger math model.
  """
  text = (user_input or '').lower()
  score = 0

  if len(text) >= 260:
    score += 2
  elif len(text) >= 180:
    score += 1

  proof_markers = [
    'prove', 'proof', 'show that', 'for all', 'exists', 'let ', 'suppose', 'therefore'
  ]
  if any(m in text for m in proof_markers):
    score += 2

  hard_topics = [
    'functional equation', 'inequality', 'diophantine', 'number theory', 'congruent',
    'expected value', 'variance', 'combinatorics', 'bijection', 'invariant',
    'generating function', 'recurrence', 'determinant', 'eigenvalue'
  ]
  if any(m in text for m in hard_topics):
    score += 1

  # Dense LaTeX / algebra often correlates with harder items.
  latex_hits = len(re.findall(r'\\(frac|sqrt|sum|prod|int|binom|cdot|geq|leq|neq)', text))
  if latex_hits >= 6:
    score += 1

  return score


def _pick_deepseek_r1_math_model(user_input: str, installed: List[str]) -> Optional[str]:
  """
  Prefer DeepSeek-R1 variants for MATH-500, switching by question difficulty.
  - easy: deepseek-r1:8b
  - medium: deepseek-r1:14b (if installed)
  - hard: deepseek-r1:32b
  """
  small = _env('OLLAMA_MATH_SMALL_MODEL', 'deepseek-r1:8b')
  medium = _env('OLLAMA_MATH_MEDIUM_MODEL', 'deepseek-r1:14b')
  large = _env('OLLAMA_MATH_LARGE_MODEL', 'deepseek-r1:32b')

  if not installed:
    score = _estimate_math500_difficulty(user_input)
    if score >= 3:
      return large or medium or small
    if score >= 1:
      return medium or small or large
    return small or medium or large

  has_any_r1 = any(m.startswith('deepseek-r1') for m in installed)
  if not has_any_r1:
    return None

  score = _estimate_math500_difficulty(user_input)
  if score >= 3:
    return _first_existing([large, medium, small], installed) or _best_by_size(
      [m for m in installed if m.startswith('deepseek-r1')]
    )
  if score >= 1:
    return _first_existing([medium, small, large], installed) or _best_by_size(
      [m for m in installed if m.startswith('deepseek-r1')]
    )
  return _first_existing([small, medium, large], installed) or _best_by_size(
    [m for m in installed if m.startswith('deepseek-r1')]
  )


def _is_hard_general(user_input: str) -> bool:
  text = (user_input or '')
  if len(text) > 300:
    return True
  return False


def _is_code_request(user_input: str) -> bool:
  text = (user_input or '').lower()
  keywords = [
    'write code', 'code for', 'implement', 'create a', 'build a',
    'java', 'python', 'javascript', 'typescript', 'react', 'spring',
    'api', 'backend', 'frontend', 'class ', 'function '
  ]
  return any(k in text for k in keywords)


def select_model(user_input: str, *, is_tool: bool = False) -> SelectedModel:
  intent = classify_intent(user_input)

  installed = _get_installed_ollama_models()

  # Env “preferences” (router will still consider all installed models)
  tiny = _env('OLLAMA_TINY_MODEL', 'phi3:mini')
  fast = _env('OLLAMA_FAST_MODEL', 'gemma3:4b')
  general = _env('OLLAMA_GENERAL_MODEL', 'deepseek-r1:8b')
  tool = _env('OLLAMA_TOOL_MODEL', general)
  strong = _env('OLLAMA_STRONG_MODEL', 'deepseek-r1:32b') or general
  ultra = _env('OLLAMA_ULTRA_MODEL', strong) or strong

  # If we can see installed models, snap preferences to installed ones.
  if installed:
    tiny = _first_existing([tiny, fast, general], installed) or tiny
    fast = _first_existing([fast, general, tiny], installed) or fast
    general = _first_existing([general, strong, fast, tiny], installed) or general
    tool = _first_existing([tool, general, strong, fast], installed) or tool
    strong = _first_existing([strong, general, ultra], installed) or strong
    ultra = _first_existing([ultra, strong, general], installed) or ultra

  if is_tool:
    picked, why = _pick_tool_model(user_input, installed, tool, strong)
    return SelectedModel(
      provider='ollama',
      model=picked,
      temperature=_select_temperature(intent, is_tool=True),
      system_prompt=_select_system_prompt(intent, is_tool=True),
      reason=why
    )

  # Ollama-only (no API keys)
  if intent == 'math':
    r1_math = _pick_deepseek_r1_math_model(user_input, installed)
    if _is_truthy(_env('ROUTER_MATH_ALWAYS_STRONG', '0')):
      return SelectedModel(
        provider='ollama',
        model=strong,
        temperature=_select_temperature(intent, is_tool=is_tool),
        system_prompt=_select_system_prompt(intent, is_tool=is_tool),
        reason='math:forced_strong'
      )
    if _is_hard_math(user_input):
      # Prefer DeepSeek-R1 large for hard MATH-500 when available.
      if r1_math:
        return SelectedModel(
          provider='ollama',
          model=r1_math,
          temperature=_select_temperature(intent, is_tool=is_tool),
          system_prompt=_select_system_prompt(intent, is_tool=is_tool),
          reason=f'math:hard -> {r1_math}'
        )
      # Otherwise prefer the strongest available math-ish model.
      if installed:
        best_math = _best_by_size(installed, prefer_regex=r'(deepseek|gemma3|gpt-oss)')
        return SelectedModel(
          provider='ollama',
          model=best_math or strong,
          temperature=_select_temperature(intent, is_tool=is_tool),
          system_prompt=_select_system_prompt(intent, is_tool=is_tool),
          reason=f'math:hard -> {best_math or strong}'
        )
      return SelectedModel(
        provider='ollama',
        model=strong,
        temperature=_select_temperature(intent, is_tool=is_tool),
        system_prompt=_select_system_prompt(intent, is_tool=is_tool),
        reason='math:hard -> strong'
      )
    if r1_math:
      return SelectedModel(
        provider='ollama',
        model=r1_math,
        temperature=_select_temperature(intent, is_tool=is_tool),
        system_prompt=_select_system_prompt(intent, is_tool=is_tool),
        reason=f'math:tiered -> {r1_math}'
      )
    return SelectedModel(
      provider='ollama',
      model=general,
      temperature=_select_temperature(intent, is_tool=is_tool),
      system_prompt=_select_system_prompt(intent, is_tool=is_tool),
      reason='math:default_general'
    )

  # general Q&A
  if _is_code_request(user_input):
    if installed:
      # Prefer coder models first, then biggest general model.
      best_coder = _best_by_size(installed, prefer_regex=r'coder|code')
      if best_coder:
        return SelectedModel(
          provider='ollama',
          model=best_coder,
          temperature=_select_temperature(intent, is_tool=is_tool),
          system_prompt=_select_system_prompt(intent, is_tool=is_tool),
          reason=f'code:prefer_coder -> {best_coder}'
        )
      best_any = _best_by_size(installed)
      return SelectedModel(
        provider='ollama',
        model=best_any or strong,
        temperature=_select_temperature(intent, is_tool=is_tool),
        system_prompt=_select_system_prompt(intent, is_tool=is_tool),
        reason=f'code:largest_available -> {best_any or strong}'
      )
    return SelectedModel(
      provider='ollama',
      model=strong,
      temperature=_select_temperature(intent, is_tool=is_tool),
      system_prompt=_select_system_prompt(intent, is_tool=is_tool),
      reason='code:fallback_strong'
    )
  if _is_hard_general(user_input):
    if installed:
      best_any = _best_by_size(installed)
      return SelectedModel(
        provider='ollama',
        model=best_any or ultra,
        temperature=_select_temperature(intent, is_tool=is_tool),
        system_prompt=_select_system_prompt(intent, is_tool=is_tool),
        reason=f'general:hard -> {best_any or ultra}'
      )
    return SelectedModel(
      provider='ollama',
      model=ultra,
      temperature=_select_temperature(intent, is_tool=is_tool),
      system_prompt=_select_system_prompt(intent, is_tool=is_tool),
      reason='general:hard -> ultra'
    )
  if len(user_input or '') <= 160:
    return SelectedModel(
      provider='ollama',
      model=tiny,
      temperature=_select_temperature(intent, is_tool=is_tool),
      system_prompt=_select_system_prompt(intent, is_tool=is_tool),
      reason='general:short -> tiny'
    )
  return SelectedModel(
    provider='ollama',
    model=fast,
    temperature=_select_temperature(intent, is_tool=is_tool),
    system_prompt=_select_system_prompt(intent, is_tool=is_tool),
    reason='general:default -> fast'
  )


def build_chat_llm(selected: SelectedModel):
  from agent import build_llm  # keep single Ollama loader/fallback
  return build_llm(model=selected.model, temperature=selected.temperature)


def answer_general(user_input: str, chat_history: List[BaseMessage], selected: SelectedModel) -> str:
  llm = build_chat_llm(selected)
  system_prompt = selected.system_prompt or _select_system_prompt(classify_intent(user_input), is_tool=False)
  messages: List[BaseMessage] = [
    SystemMessage(content=system_prompt),
    *chat_history,
    HumanMessage(content=user_input)
  ]
  start = __import__('time').time()
  res = llm.invoke(messages)
  out = (getattr(res, 'content', None) or str(res)).strip()

  trace_path = _env('ROUTER_TRACE_PATH', '')
  if trace_path:
    append_jsonl(trace_path, {
      'event': 'router.answer_general',
      'intent': classify_intent(user_input),
      'model': selected.model,
      'temperature': selected.temperature,
      'reason': selected.reason,
      'duration_ms': int((__import__('time').time() - start) * 1000),
      'input_chars': len(user_input or ''),
      'output_chars': len(out or '')
    })

  return out

