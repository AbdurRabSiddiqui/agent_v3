from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List


_CACHE_ROOT = Path(__file__).resolve().parent.parent / '.hf_cache'
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('HF_HOME', str(_CACHE_ROOT))
os.environ.setdefault('HF_DATASETS_CACHE', str(_CACHE_ROOT / 'datasets'))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(_CACHE_ROOT / 'hub'))


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


def load_math500_tasks(*, limit: int = 400) -> List[Dict[str, str]]:
  from datasets import load_dataset

  ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
  n = len(ds) if limit <= 0 else min(int(limit), len(ds))
  out: List[Dict[str, str]] = []
  for idx, row in enumerate(ds.select(range(n))):
    out.append({
      'idx': str(idx),
      'prompt': build_math500_prompt(row['problem']),
      'gold': str(row['answer'] or '')
    })
  return out

