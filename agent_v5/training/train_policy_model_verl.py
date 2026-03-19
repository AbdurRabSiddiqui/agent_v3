from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional, TypedDict

from dotenv import load_dotenv
from openai import OpenAI

import agentlightning as agl
from agentlightning.algorithm.verl import VERL

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
  t = ''.join(t.split())
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


def _policy_prompt(*, user_input: str, answer: str) -> str:
  return '\n'.join([
    'You are a strict verifier policy.',
    'Decide whether the candidate answer is good enough to return, or whether to escalate to a stronger model.',
    'Return STRICT JSON only with schema:',
    '{"accept": <bool>, "confidence": <0..1>, "reason": "<short>"}',
    '',
    f'User request:\n{user_input}',
    '',
    f'Candidate answer:\n{(answer or "").strip()}'
  ])


def main() -> int:
  parser = argparse.ArgumentParser(description='Train a verifier/policy model with Agent Lightning + VERL.')
  parser.add_argument('--limit', type=int, default=400)
  parser.add_argument('--train-n', type=int, default=320)
  parser.add_argument('--out-jsonl', default='logs/verl_policy_rollouts.jsonl')
  args = parser.parse_args()

  load_dotenv()

  tasks_raw = load_math500_tasks(limit=args.limit)
  tasks: list[Math500Task] = [Math500Task(idx=t['idx'], prompt=t['prompt'], gold=t['gold']) for t in tasks_raw]
  train = tasks[: max(int(args.train_n), 0)]
  val = tasks[max(int(args.train_n), 0):]

  small_model = os.environ.get('OLLAMA_MATH_SMALL_MODEL') or os.environ.get('OLLAMA_GENERAL_MODEL') or 'deepseek-r1:8b'
  strong_model = os.environ.get('OLLAMA_MATH_LARGE_MODEL') or os.environ.get('OLLAMA_STRONG_MODEL') or small_model
  draft_timeout_s = float(os.environ.get('MODEL_SELECT_DRAFT_TIMEOUT_S', '180') or '180')

  lambda_strong = float(os.environ.get('RL_LAMBDA_STRONG', '0.05') or '0.05')

  # VERL will provide an OpenAI-compatible proxy endpoint for the policy model.
  @agl.rollout
  def policy_agent(task: Math500Task, llm: agl.LLM) -> float:
    ok_small, small_ans, _ms_small = _call_ollama(model=small_model, prompt=task['prompt'], timeout_s=draft_timeout_s)
    if (not ok_small) or (not _extract_boxed(small_ans)):
      ok_strong, strong_ans, _ms_strong = _call_ollama(model=strong_model, prompt=task['prompt'], timeout_s=draft_timeout_s)
      pred = strong_ans if ok_strong else ''
      correct = 1.0 if _compare(pred, task['gold']) else 0.0
      reward = correct - lambda_strong
      _append_rollout(args.out_jsonl, task, used_strong=True, correct=correct, reward=reward)
      return float(reward)

    prompt = _policy_prompt(user_input=task['prompt'], answer=small_ans)
    client = OpenAI(base_url=llm.endpoint, api_key=llm.api_key or 'dummy-key')
    resp = client.chat.completions.create(
      model=llm.model,
      messages=[{'role': 'user', 'content': prompt}]
    )
    raw = (resp.choices[0].message.content or '').strip()
    parsed = try_parse_verify_json(raw)
    if not parsed:
      # invalid JSON => treat as unsure and escalate (conservative)
      ok_strong, strong_ans, _ms_strong = _call_ollama(model=strong_model, prompt=task['prompt'], timeout_s=draft_timeout_s)
      pred = strong_ans if ok_strong else small_ans
      correct = 1.0 if _compare(pred, task['gold']) else 0.0
      reward = correct - lambda_strong
      _append_rollout(args.out_jsonl, task, used_strong=True, correct=correct, reward=reward)
      return float(reward)

    accept, conf, _reason, _parsed_obj = parsed
    decision = decision_from_verify(
      accept=accept,
      confidence=float(conf),
      accept_threshold=float(os.environ.get('MODEL_SELECT_ACCEPT_THRESHOLD', '0.78')),
      reject_threshold=float(os.environ.get('MODEL_SELECT_REJECT_THRESHOLD', '0.78'))
    )

    used_strong = False
    pred = small_ans
    if decision != 'accept':
      ok_strong, strong_ans, _ms_strong = _call_ollama(model=strong_model, prompt=task['prompt'], timeout_s=draft_timeout_s)
      used_strong = True
      pred = strong_ans if ok_strong else small_ans

    correct = 1.0 if _compare(pred, task['gold']) else 0.0
    reward = correct - (lambda_strong if used_strong else 0.0)
    _append_rollout(args.out_jsonl, task, used_strong=used_strong, correct=correct, reward=reward)
    return float(reward)

  def _append_rollout(path: str, task: Math500Task, *, used_strong: bool, correct: float, reward: float):
    if not path:
      return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
      'ts': time.time(),
      'idx': task['idx'],
      'used_strong': bool(used_strong),
      'correct': float(correct),
      'reward': float(reward)
    }
    with open(path, 'a', encoding='utf-8') as f:
      f.write(json.dumps(payload, ensure_ascii=False) + '\n')

  algorithm = VERL(
    config={
      'algorithm': {
        'adv_estimator': 'grpo',
        'use_kl_in_reward': False
      },
      'data': {
        'train_batch_size': int(os.environ.get('VERL_TRAIN_BATCH_SIZE', '8')),
        'max_prompt_length': int(os.environ.get('VERL_MAX_PROMPT_LENGTH', '4096')),
        'max_response_length': int(os.environ.get('VERL_MAX_RESPONSE_LENGTH', '256'))
      },
      'actor_rollout_ref': {
        'rollout': {
          'name': 'vllm',
          'tensor_model_parallel_size': int(os.environ.get('VERL_TP', '1')),
          'gpu_memory_utilization': float(os.environ.get('VERL_GPU_UTIL', '0.6')),
          'n': int(os.environ.get('VERL_SAMPLES_PER_PROMPT', '2'))
        },
        'actor': {
          'ppo_mini_batch_size': int(os.environ.get('VERL_PPO_MINI_BATCH', '8')),
          'ppo_micro_batch_size_per_gpu': int(os.environ.get('VERL_PPO_MICRO_BATCH', '2')),
          'optim': {'lr': float(os.environ.get('VERL_LR', '1e-6'))},
          'use_kl_loss': False,
          'kl_loss_coef': 0.0,
          'entropy_coeff': 0
        },
        'ref': {
          'log_prob_micro_batch_size_per_gpu': int(os.environ.get('VERL_REF_MICRO_BATCH', '2'))
        },
        'model': {
          # Policy model to train (HF path).
          'path': os.environ.get('VERL_POLICY_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct'),
          'use_remove_padding': True,
          'enable_gradient_checkpointing': True
        }
      },
      'trainer': {
        'n_gpus_per_node': int(os.environ.get('VERL_N_GPUS', '1')),
        'nnodes': 1,
        'total_epochs': int(os.environ.get('VERL_EPOCHS', '1')),
        'logger': ['console']
      }
    }
  )

  agl.Trainer(algorithm=algorithm).fit(agent=policy_agent, train_dataset=train, val_dataset=val)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

