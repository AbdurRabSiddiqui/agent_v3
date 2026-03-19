from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import agentlightning as agl
from datasets import load_dataset
from openai import OpenAI

from agent_selector import answer_with_selection
from lookahead.policy import LookaheadPolicy
from lookahead.math500 import build_math500_prompt, is_correct_math500


@dataclass(frozen=True)
class Math500Task:
  idx: int
  problem: str
  answer: str


def _controller_action_schema() -> str:
  return '\n'.join([
    'Return STRICT JSON only. No markdown, no prose.',
    'Schema:',
    '{"action": "draft", "model": "<one of candidates>"}',
    'Optional (one-step cascade):',
    '{"action": "escalate", "model": "<one of candidates>"}',
    '{"action": "stop"}'
  ])


def _build_controller_prompt(*, prompt: str, candidates: list[str], lookahead_hint: Optional[dict]) -> str:
  hint = json.dumps(lookahead_hint, ensure_ascii=False) if lookahead_hint else ''
  return '\n'.join([
    'You are a model-selection controller for a local multi-model system.',
    'Goal: choose the model that will solve the problem correctly with minimal energy/latency.',
    'You do NOT solve the math problem yourself; you only choose which model should answer.',
    '',
    'Candidates:',
    *[f'- {c}' for c in candidates],
    '',
    'Optional lookahead scores (predicted correctness/cost):',
    hint or '(none)',
    '',
    _controller_action_schema(),
    '',
    'Problem prompt:',
    prompt
  ])


def _try_parse_action(raw: str) -> Optional[dict]:
  try:
    obj = json.loads(raw)
  except Exception:
    return None
  if not isinstance(obj, dict):
    return None
  action = str(obj.get('action') or '').strip()
  if action not in ('draft', 'escalate', 'stop'):
    return None
  if action in ('draft', 'escalate'):
    model = str(obj.get('model') or '').strip()
    if not model:
      return None
  return obj


class LitMathRouterAgent(agl.LitAgent[Math500Task]):
  """
  Agent Lightning RL wrapper that trains a controller LLM to choose a local model.

  - The controller LLM is provided by VERL as resources['main_llm'].
  - The chosen answer model is executed via your existing local Ollama stack in agent_selector.py.
  - Reward is correctness - lambda_t * latency_ms - lambda_e * energy_proxy.

  Note: true GPU energy measurement is hard under parallel rollouts; by default we use a proxy based on model size.
  """

  def __init__(
    self,
    *,
    candidates: list[str],
    lambda_t: float,
    lambda_e: float,
    max_steps: int = 2
  ):
    super().__init__()
    self.candidates = list(candidates)
    self.lambda_t = float(lambda_t)
    self.lambda_e = float(lambda_e)
    self.max_steps = int(max_steps)

    self.lookahead = LookaheadPolicy.from_env()

  def rollout(self, task: Math500Task, resources: agl.NamedResources, rollout: agl.Rollout) -> float | None:
    proxy_llm = resources['main_llm']
    base_url = proxy_llm.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id)
    client = OpenAI(base_url=base_url, api_key=os.environ.get('OPENAI_API_KEY') or 'agentlightning')

    prompt = build_math500_prompt(task.problem)
    lookahead_hint = None
    if self.lookahead:
      chosen, scored = self.lookahead.select_model(prompt=prompt, candidates=self.candidates)
      lookahead_hint = self.lookahead.to_json(scored, chosen=chosen)

    chosen_model = ''
    start = time.time()
    steps = 0
    while steps < self.max_steps and not chosen_model:
      steps += 1
      ctrl_prompt = _build_controller_prompt(prompt=prompt, candidates=self.candidates, lookahead_hint=lookahead_hint)
      resp = client.chat.completions.create(
        model=proxy_llm.model,
        messages=[{'role': 'system', 'content': 'Return strict JSON only.'}, {'role': 'user', 'content': ctrl_prompt}],
        temperature=0.0
      )
      raw = (resp.choices[0].message.content or '').strip()
      act = _try_parse_action(raw)
      if not act:
        continue
      if act['action'] == 'stop':
        break
      m = str(act.get('model') or '').strip()
      if m in self.candidates:
        chosen_model = m

    if not chosen_model:
      chosen_model = self.candidates[-1]

    # Execute exactly one answer run with the chosen model (no tournament).
    answer, _sel = answer_with_selection(prompt, [], force_model=chosen_model, trace_path=os.environ.get('AGENT_TRACE_PATH', ''))
    duration_ms = int((time.time() - start) * 1000)
    correct_int = 1 if is_correct_math500(answer, task.answer) else 0

    # Energy proxy: model size in billions (cheap heuristic). RL can still learn policies that avoid big models when not needed.
    # If you want true energy, run single-runner rollouts and add NVML integration here.
    def parse_b(m: str) -> float:
      import re
      mt = re.search(r':(\\d+(?:\\.\\d+)?)b', m)
      if mt:
        try:
          return float(mt.group(1))
        except Exception:
          return 0.0
      return 0.0

    energy_proxy = parse_b(chosen_model)
    reward = float(correct_int - (self.lambda_t * duration_ms) - (self.lambda_e * energy_proxy))
    return reward


def load_math500_tasks(*, limit: int = 0) -> list[Math500Task]:
  ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
  total = len(ds) if limit <= 0 else min(int(limit), len(ds))
  out: list[Math500Task] = []
  for i in range(total):
    out.append(Math500Task(idx=i, problem=ds[i]['problem'], answer=ds[i]['answer']))
  return out

