from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Optional

_CACHE_MODELS: Optional[list[str]] = None
_CACHE_TIME_S: float = 0.0


def is_truthy(value: str) -> bool:
  return (value or '').strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def env(name: str, default: str = '') -> str:
  return os.environ.get(name, default)


def get_installed_ollama_models(*, ttl_s: float = 5.0, allow_cloud: Optional[bool] = None) -> list[str]:
  """
  Return installed Ollama model names from `ollama list`.
  Uses a short TTL cache so newly pulled models appear quickly without restart.
  """
  global _CACHE_MODELS, _CACHE_TIME_S

  now = time.time()
  if _CACHE_MODELS is not None and (now - _CACHE_TIME_S) <= max(float(ttl_s), 0.0):
    return _CACHE_MODELS

  if allow_cloud is None:
    allow_cloud = is_truthy(env(
      'MODEL_ALLOW_OLLAMA_CLOUD',
      env('MODEL_ALLOW_CLOUD', env('ROUTER_ALLOW_OLLAMA_CLOUD', env('ROUTER_ALLOW_CLOUD', '0')))
    ))

  try:
    out = subprocess.check_output(['ollama', 'list'], text=True)
  except Exception:
    _CACHE_MODELS = []
    _CACHE_TIME_S = now
    return _CACHE_MODELS

  models: list[str] = []
  for line in out.splitlines()[1:]:
    line = (line or '').strip()
    if not line:
      continue
    name = line.split()[0]
    if not name or name == 'NAME':
      continue
    if not allow_cloud and name.endswith('-cloud'):
      continue
    models.append(name)

  _CACHE_MODELS = models
  _CACHE_TIME_S = now
  return models


def parse_model_size_b(model: str) -> float:
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


def best_by_size(models: list[str]) -> Optional[str]:
  if not models:
    return None
  return sorted(models, key=parse_model_size_b, reverse=True)[0]


def fallback_to_installed_model(requested_model: str, *, installed: Optional[list[str]] = None) -> str:
  models = installed if installed is not None else get_installed_ollama_models()
  if not models:
    return requested_model
  models_set = set(models)
  if requested_model in models_set:
    return requested_model

  # Prefer configured models first (including strong/ultra/tool), then fall back to the biggest installed.
  for candidate in [
    env('OLLAMA_TOOL_MODEL'),
    env('OLLAMA_ULTRA_MODEL'),
    env('OLLAMA_STRONG_MODEL'),
    env('OLLAMA_GENERAL_MODEL'),
    env('OLLAMA_FAST_MODEL'),
    env('OLLAMA_TINY_MODEL'),
    env('OLLAMA_MODEL')
  ]:
    if candidate and candidate in models_set:
      return candidate

  biggest = best_by_size(models)
  if biggest:
    return biggest

  return requested_model

