from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np


@dataclass(frozen=True)
class LookaheadScore:
  model: str
  p_correct: float
  pred_latency_ms: float
  pred_energy_j: float
  utility: float


class LookaheadPolicy:
  def __init__(
    self,
    *,
    artifact_path: str,
    lambda_t: float = 0.0,
    lambda_e: float = 0.0,
    min_p_correct: float = 0.0,
    fallback_model: str = ''
  ):
    self.artifact_path = artifact_path
    self.lambda_t = float(lambda_t)
    self.lambda_e = float(lambda_e)
    self.min_p_correct = float(min_p_correct)
    self.fallback_model = str(fallback_model or '')

    art = joblib.load(artifact_path)
    self.prompt_vectorizer = art['prompt_vectorizer']
    self.models = list(art.get('models') or [])
    self.response_forecasters = art.get('response_forecasters') or {}
    self.correct_heads = art.get('correct_heads') or {}
    self.latency_heads = art.get('latency_heads') or {}
    self.energy_heads = art.get('energy_heads') or {}

  @classmethod
  def from_env(cls) -> Optional['LookaheadPolicy']:
    if (os.environ.get('MODEL_SELECT_POLICY') or '').strip().lower() != 'lookahead':
      return None
    path = (os.environ.get('LOOKAHEAD_MODEL_PATH') or 'models/lookahead_math500.joblib').strip()
    if not Path(path).exists():
      return None
    return cls(
      artifact_path=path,
      lambda_t=float(os.environ.get('LOOKAHEAD_LAMBDA_T') or 0.0),
      lambda_e=float(os.environ.get('LOOKAHEAD_LAMBDA_E') or 0.0),
      min_p_correct=float(os.environ.get('LOOKAHEAD_MIN_P_CORRECT') or 0.0),
      fallback_model=str(os.environ.get('LOOKAHEAD_FALLBACK_MODEL') or '').strip()
    )

  def score_models(self, *, prompt: str, candidates: list[str]) -> list[LookaheadScore]:
    prompt = str(prompt or '')
    Xp = self.prompt_vectorizer.transform([prompt])

    scores: list[LookaheadScore] = []
    for m in list(candidates or []):
      if m not in self.response_forecasters:
        continue

      forecaster = self.response_forecasters[m]
      Yhat = forecaster.predict(Xp)
      X_joint = np.hstack([Xp.toarray(), Yhat.toarray()]).astype(np.float32)

      p = 0.0
      if m in self.correct_heads:
        head = self.correct_heads[m]
        try:
          proba = head.predict_proba(X_joint)[0]
          p = float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception:
          try:
            p = float(head.decision_function(X_joint)[0])
          except Exception:
            p = 0.0

      lat = float(self.latency_heads[m].predict(X_joint)[0]) if m in self.latency_heads else 0.0
      en = float(self.energy_heads[m].predict(X_joint)[0]) if m in self.energy_heads else 0.0

      utility = float(p - (self.lambda_t * lat) - (self.lambda_e * en))
      scores.append(LookaheadScore(
        model=m,
        p_correct=p,
        pred_latency_ms=lat,
        pred_energy_j=en,
        utility=utility
      ))

    scores_sorted = sorted(scores, key=lambda s: s.utility, reverse=True)
    return scores_sorted

  def select_model(self, *, prompt: str, candidates: list[str]) -> tuple[str, list[LookaheadScore]]:
    scored = self.score_models(prompt=prompt, candidates=candidates)
    if not scored:
      return '', []

    best = scored[0]
    if best.p_correct >= self.min_p_correct:
      return best.model, scored

    if self.fallback_model and self.fallback_model in candidates:
      return self.fallback_model, scored

    return best.model, scored

  def to_json(self, scored: list[LookaheadScore], *, chosen: str) -> dict[str, Any]:
    return {
      'chosen_model': chosen,
      'scores': [
        {
          'model': s.model,
          'p_correct': round(float(s.p_correct), 6),
          'pred_latency_ms': round(float(s.pred_latency_ms), 2),
          'pred_energy_j': round(float(s.pred_energy_j), 3),
          'utility': round(float(s.utility), 6)
        } for s in scored
      ]
    }

  def to_json_text(self, scored: list[LookaheadScore], *, chosen: str) -> str:
    return json.dumps(self.to_json(scored, chosen=chosen), ensure_ascii=False)

