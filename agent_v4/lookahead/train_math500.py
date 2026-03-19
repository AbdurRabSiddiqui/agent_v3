from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def _iter_jsonl(path: str):
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
      line = (line or '').strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except Exception:
        continue


@dataclass(frozen=True)
class ModelArtifacts:
  prompt_vectorizer: TfidfVectorizer
  response_vectorizer: TfidfVectorizer
  models: list[str]
  response_forecasters: dict[str, Ridge]
  correct_heads: dict[str, SGDClassifier]
  latency_heads: dict[str, SGDRegressor]
  energy_heads: dict[str, SGDRegressor]


def train(
  *,
  dataset_jsonl: str,
  out_path: str,
  max_prompt_features: int = 60000,
  max_response_features: int = 20000
) -> str:
  rows = [r for r in _iter_jsonl(dataset_jsonl)]
  if not rows:
    raise RuntimeError(f'No rows found in dataset: {dataset_jsonl}')

  prompts = [str(r.get('prompt') or '') for r in rows]
  responses = [str(r.get('output') or '') for r in rows]
  models = sorted({str(r.get('model') or '').strip() for r in rows if str(r.get('model') or '').strip()})
  if not models:
    raise RuntimeError('No models found in dataset rows.')

  prompt_vec = TfidfVectorizer(
    max_features=int(max_prompt_features),
    ngram_range=(1, 2),
    min_df=1
  )
  response_vec = TfidfVectorizer(
    max_features=int(max_response_features),
    ngram_range=(1, 2),
    min_df=1
  )

  Xp = prompt_vec.fit_transform(prompts)
  Yr = response_vec.fit_transform(responses)

  response_forecasters: dict[str, Ridge] = {}
  correct_heads: dict[str, SGDClassifier] = {}
  latency_heads: dict[str, SGDRegressor] = {}
  energy_heads: dict[str, SGDRegressor] = {}

  for m in models:
    idxs = [i for i, r in enumerate(rows) if str(r.get('model') or '').strip() == m]
    if not idxs:
      continue

    X_m = Xp[idxs]
    Yresp_m = Yr[idxs]
    y_correct = np.array([int(rows[i].get('correct_int') or 0) for i in idxs], dtype=np.int32)
    y_lat = np.array([float(rows[i].get('duration_ms') or 0.0) for i in idxs], dtype=np.float32)
    y_energy = np.array([float(rows[i].get('energy_j') or 0.0) for i in idxs], dtype=np.float32)

    # Lookahead-style latent forecast: predict response TF-IDF from prompt TF-IDF.
    forecaster = Ridge(alpha=1.0, random_state=0)
    forecaster.fit(X_m, Yresp_m)
    response_forecasters[m] = forecaster

    # Use prompt + predicted-response-latent as features for quality/cost.
    Yhat = forecaster.predict(X_m)
    X_joint = np.hstack([X_m.toarray(), Yhat.toarray()]).astype(np.float32)

    clf = Pipeline([
      ('scaler', StandardScaler(with_mean=False)),
      ('clf', SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=2000, tol=1e-3, random_state=0))
    ])
    clf.fit(X_joint, y_correct)
    correct_heads[m] = clf  # type: ignore[assignment]

    lat_reg = Pipeline([
      ('scaler', StandardScaler(with_mean=False)),
      ('reg', SGDRegressor(alpha=1e-6, max_iter=2000, tol=1e-3, random_state=0))
    ])
    lat_reg.fit(X_joint, y_lat)
    latency_heads[m] = lat_reg  # type: ignore[assignment]

    e_reg = Pipeline([
      ('scaler', StandardScaler(with_mean=False)),
      ('reg', SGDRegressor(alpha=1e-6, max_iter=2000, tol=1e-3, random_state=0))
    ])
    e_reg.fit(X_joint, y_energy)
    energy_heads[m] = e_reg  # type: ignore[assignment]

  art = {
    'prompt_vectorizer': prompt_vec,
    'response_vectorizer': response_vec,
    'models': models,
    'response_forecasters': response_forecasters,
    'correct_heads': correct_heads,
    'latency_heads': latency_heads,
    'energy_heads': energy_heads
  }

  Path(out_path).parent.mkdir(parents=True, exist_ok=True)
  joblib.dump(art, out_path)
  return out_path


def main() -> int:
  ap = argparse.ArgumentParser(description='Train a Lookahead-style selector for MATH-500 from logged traces.')
  ap.add_argument('--dataset-jsonl', default='logs/lookahead_math500_dataset.jsonl', help='Dataset built by lookahead/dataset_math500.py')
  ap.add_argument('--out', default='models/lookahead_math500.joblib', help='Output model artifact (joblib)')
  ap.add_argument('--max-prompt-features', type=int, default=60000)
  ap.add_argument('--max-response-features', type=int, default=20000)
  args = ap.parse_args()

  out = train(
    dataset_jsonl=args.dataset_jsonl,
    out_path=args.out,
    max_prompt_features=int(args.max_prompt_features),
    max_response_features=int(args.max_response_features)
  )
  print(f'Wrote lookahead model to "{out}"')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

