from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from agent_selector import answer_with_selection
from trace_utils import append_jsonl

_CACHE_ROOT = Path(__file__).resolve().parent / '.hf_cache'
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HOME', str(_CACHE_ROOT))
os.environ.setdefault('HF_DATASETS_CACHE', str(_CACHE_ROOT / 'datasets'))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(_CACHE_ROOT / 'hub'))


@dataclass(frozen=True)
class EvalResult:
  correct: int
  total: int


def _extract_boxed(text: str) -> Optional[str]:
  # common in MATH solutions: \boxed{...}
  if not text:
    return None
  matches = re.findall(r'\\boxed\{([^}]*)\}', text)
  if matches:
    return matches[-1].strip()
  return None


def _normalize_answer(s: str) -> str:
  if s is None:
    return ''
  s = str(s).strip()
  s = s.replace('$', '')
  s = re.sub(r'\s+', '', s)
  return s


def _compare(pred: str, gold: str) -> bool:
  pred_n = _normalize_answer(pred)
  gold_n = _normalize_answer(gold)
  if pred_n == gold_n:
    return True

  # try boxed extraction from the model output
  boxed = _extract_boxed(pred)
  if boxed and _normalize_answer(boxed) == gold_n:
    return True

  # last-resort: try float comparison when both look numeric
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


def _build_prompt(problem: str) -> str:
  return '\n'.join([
    'You are solving a MATH-500 competition problem.',
    'Think privately; do NOT show steps, reasoning, or explanations.',
    r'Output ONLY the final answer in the form: \boxed{...}',
    'Do not output anything else (no words, no punctuation outside the box).',
    r'Use simplest exact form. For rationals use \frac{a}{b}. Avoid decimals unless unavoidable.',
    '',
    f'Problem: {problem}'
  ])


def main() -> int:
  from datasets import load_dataset
  from tqdm import tqdm

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default=None, help='Force a specific Ollama model (disables selection tournament)')
  parser.add_argument('--limit', type=int, default=0, help='Evaluate only first N problems (0 = all)')
  parser.add_argument('--show-model', action='store_true', help='Print selected model per problem')
  parser.add_argument('--log-jsonl', default='', help='Optional JSONL log path (per-item results)')
  args = parser.parse_args()

  load_dotenv()
  agent_trace_path = os.environ.get('AGENT_TRACE_PATH', '')

  ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
  total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))

  correct = 0
  for idx, row in enumerate(tqdm(ds.select(range(total)), total=total)):
    problem = row['problem']
    gold = row['answer']
    prompt = _build_prompt(problem)

    start = time.time()
    pred, selected = answer_with_selection(
      prompt,
      [],
      force_model=args.model,
      trace_path=agent_trace_path,
      selection_k=3
    )
    duration_ms = int((time.time() - start) * 1000)
    is_correct = _compare(pred, gold)
    if is_correct:
      correct += 1

    if args.show_model:
      picked = selected.model or '(fast-path)'
      print(f'\n[agent] model={picked} reason={selected.reason} temp={selected.temperature}\n')

    if agent_trace_path:
      append_jsonl(agent_trace_path, {
        'event': 'math500.item',
        'idx': idx,
        'model': selected.model,
        'temperature': selected.temperature,
        'reason': selected.reason,
        'duration_ms': duration_ms,
        'correct': bool(is_correct),
        'level': row.get('level'),
        'type': row.get('type')
      })

    if args.log_jsonl:
      append_jsonl(args.log_jsonl, {
        'event': 'math500.item',
        'idx': idx,
        'model': selected.model,
        'temperature': selected.temperature,
        'reason': selected.reason,
        'duration_ms': duration_ms,
        'correct': bool(is_correct),
        'level': row.get('level'),
        'type': row.get('type')
      })

  result = EvalResult(correct=correct, total=total)
  acc = (result.correct / result.total) if result.total else 0.0
  print(f'Correct: {result.correct}/{result.total}')
  print(f'Accuracy: {acc:.4f}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

