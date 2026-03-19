from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv

from agent_selector import _invoke_chat_with_timeout  # type: ignore
from verifier import decision_from_verify, try_parse_verify_json
from training.math500_tasks import load_math500_tasks


class Math500Task(TypedDict):
  idx: str
  prompt: str
  gold: str


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


def _normalize_answer(s: str) -> str:
  if s is None:
    return ''
  t = str(s).strip().replace('$', '')
  t = t.replace('\\dfrac', '\\frac')
  t = t.replace('\\left', '')
  t = t.replace('\\right', '')
  t = t.replace('\\,', '')
  t = re.sub(r'\\s+', '', t)
  return t


def _compare(pred: str, gold: str) -> bool:
  pn = _normalize_answer(pred)
  gn = _normalize_answer(gold)
  if pn == gn:
    return True
  pb = _extract_boxed(pred)
  gb = _extract_boxed(gold)
  if pb and gb:
    return _normalize_answer(pb) == _normalize_answer(gb)
  if pb and _normalize_answer(pb) == gn:
    return True
  return False


def _call_ollama(*, model: str, prompt: str, timeout_s: float) -> tuple[bool, str, int]:
  r = _invoke_chat_with_timeout(
    model=model,
    temperature=0.2,
    system_prompt='',
    chat_history=[],
    user_input=prompt,
    timeout_s=timeout_s
  )
  return bool(r.get('ok')), str(r.get('text') or '').strip(), int(r.get('duration_ms') or 0)


def _call_verifier(*, model: str, prompt: str, timeout_s: float) -> tuple[str, float, str, int]:
  ok, raw, ms = _call_ollama(model=model, prompt=prompt, timeout_s=timeout_s)
  if not ok:
    return 'unsure', 0.0, 'verify_failed', int(ms or 0)
  parsed = try_parse_verify_json(raw)
  if not parsed:
    return 'unsure', 0.0, 'invalid_verify_json', int(ms or 0)
  accept, conf, reason, _ = parsed
  decision = decision_from_verify(
    accept=accept,
    confidence=conf,
    accept_threshold=float(os.environ.get('MODEL_SELECT_ACCEPT_THRESHOLD', '0.78')),
    reject_threshold=float(os.environ.get('MODEL_SELECT_REJECT_THRESHOLD', '0.78'))
  )
  return decision, float(conf), str(reason or ''), int(ms or 0)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument('--limit', type=int, default=400)
  parser.add_argument('--train-n', type=int, default=320)
  parser.add_argument('--out-jsonl', default='logs/agentlightning_rollouts.jsonl')
  parser.add_argument('--power-priors', default='', help='Optional JSON from training/build_cost_priors.py')
  args = parser.parse_args()

  load_dotenv()

  # NOTE: This script is a minimal “policy experimentation” harness that is compatible with Agent Lightning’s
  # rollout-based dataset APIs. Full RL via VERL is configured via Agent Lightning’s VERL algorithm and requires
  # Linux + compatible PyTorch + vLLM stack.
  try:
    import agentlightning as agl
    from agentlightning.algorithm.fast import Baseline
  except Exception as err:
    raise SystemExit(
      'Agent Lightning is not installed. Install with `pip install agentlightning` '
      '(or `pip install agentlightning[verl]` for RL).'
    ) from err

  tasks_raw = load_math500_tasks(limit=args.limit)
  tasks: list[Math500Task] = [Math500Task(idx=t['idx'], prompt=t['prompt'], gold=t['gold']) for t in tasks_raw]
  train = tasks[: max(int(args.train_n), 0)]
  val = tasks[max(int(args.train_n), 0):]

  small_model = os.environ.get('OLLAMA_MATH_SMALL_MODEL') or os.environ.get('OLLAMA_GENERAL_MODEL') or 'deepseek-r1:8b'
  strong_model = os.environ.get('OLLAMA_MATH_LARGE_MODEL') or os.environ.get('OLLAMA_STRONG_MODEL') or small_model
  verify_model = os.environ.get('OLLAMA_VERIFY_MODEL') or os.environ.get('OLLAMA_TINY_MODEL') or small_model

  draft_timeout_s = float(os.environ.get('MODEL_SELECT_DRAFT_TIMEOUT_S', '180') or '180')
  verify_timeout_s = float(os.environ.get('MODEL_SELECT_VERIFY_TIMEOUT_S', '25') or '25')

  lambda_strong = float(os.environ.get('RL_LAMBDA_STRONG', '0.05') or '0.05')
  lambda_time = float(os.environ.get('RL_LAMBDA_TIME', '0.0') or '0.0')
  lambda_energy = float(os.environ.get('RL_LAMBDA_ENERGY', '0.0') or '0.0')

  power_priors = {}
  if (args.power_priors or '').strip():
    p = Path(args.power_priors)
    if p.exists():
      try:
        raw = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(raw, dict):
          for model, v in raw.items():
            if isinstance(v, dict) and 'avg_effective_power_w' in v:
              try:
                power_priors[str(model)] = float(v['avg_effective_power_w'])
              except Exception:
                continue
      except Exception:
        power_priors = {}

  def est_energy_j(model: str, duration_ms: int) -> float:
    p = float(power_priors.get(model) or 0.0)
    return p * (max(int(duration_ms), 0) / 1000.0)

  default_template = '\n'.join([
    'You are a strict verifier for an answer produced by another model.',
    'Return STRICT JSON only with this schema:',
    '{"accept": <bool>, "confidence": <0..1>, "reason": "<short>"}',
    'Be conservative. If unsure, accept=false.',
    '',
    'User request:',
    '{user_input}',
    '',
    'Candidate answer:',
    '{answer}'
  ])

  @agl.rollout
  def early_stop_policy(task: Math500Task, prompt_template: agl.PromptTemplate) -> float:
    start = time.time()

    ok_small, small_ans, ms_small = _call_ollama(model=small_model, prompt=task['prompt'], timeout_s=draft_timeout_s)
    if not ok_small or not _extract_boxed(small_ans):
      ok_strong, strong_ans, ms_strong = _call_ollama(model=strong_model, prompt=task['prompt'], timeout_s=draft_timeout_s)
      pred = strong_ans if ok_strong else ''
      correct = 1.0 if _compare(pred, task['gold']) else 0.0
      energy = est_energy_j(small_model, ms_small) + est_energy_j(strong_model, ms_strong)
      reward = correct - lambda_strong - (lambda_time * (time.time() - start)) - (lambda_energy * energy)
      _append_rollout(args.out_jsonl, task, used_strong=True, pred=pred, reward=reward)
      return float(reward)

    verify_prompt = prompt_template.format(user_input=task['prompt'], answer=small_ans)
    decision, _conf, _why, ms_verify = _call_verifier(model=verify_model, prompt=verify_prompt, timeout_s=verify_timeout_s)

    used_strong = False
    pred = small_ans
    ms_strong = 0
    if decision != 'accept':
      ok_strong, strong_ans, ms_strong = _call_ollama(model=strong_model, prompt=task['prompt'], timeout_s=draft_timeout_s)
      used_strong = True
      pred = strong_ans if ok_strong else small_ans

    correct = 1.0 if _compare(pred, task['gold']) else 0.0
    energy = est_energy_j(small_model, ms_small) + est_energy_j(verify_model, ms_verify) + est_energy_j(strong_model, ms_strong)
    reward = correct - (lambda_strong if used_strong else 0.0) - (lambda_time * (time.time() - start)) - (lambda_energy * energy)
    _append_rollout(args.out_jsonl, task, used_strong=used_strong, pred=pred, reward=reward)
    return float(reward)

  def _append_rollout(path: str, task: Math500Task, *, used_strong: bool, pred: str, reward: float):
    if not path:
      return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
      'ts': time.time(),
      'idx': task['idx'],
      'used_strong': bool(used_strong),
      'reward': float(reward),
      'pred': str(pred or '')[:500]
    }
    with open(path, 'a', encoding='utf-8') as f:
      f.write(json.dumps(payload, ensure_ascii=False) + '\n')

  algorithm = Baseline(n_epochs=1, train_split=0.8, span_verbosity='keys')
  initial_resources = {'prompt_template': agl.PromptTemplate(template=default_template, engine='f-string')}
  agl.Trainer(algorithm=algorithm, initial_resources=initial_resources).fit(agent=early_stop_policy, train_dataset=train, val_dataset=val)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

