from __future__ import annotations

import importlib
import os

from ollama_utils import fallback_to_installed_model


def _load_chat_ollama():
  candidates = [
    ('langchain_ollama', 'ChatOllama'),
    ('langchain_community.chat_models', 'ChatOllama')
  ]

  for module_name, symbol_name in candidates:
    try:
      module = importlib.import_module(module_name)
      return getattr(module, symbol_name)
    except Exception:
      continue

  raise ImportError(
    'Could not import ChatOllama. Install langchain-ollama or langchain-community.'
  )


def build_llm(*, model: str, temperature: float = 0.2):
  ChatOllama = _load_chat_ollama()
  base_url = os.environ.get('OLLAMA_BASE_URL')
  model = fallback_to_installed_model(model)
  if base_url:
    return ChatOllama(model=model, temperature=temperature, base_url=base_url)
  return ChatOllama(model=model, temperature=temperature)


def get_default_model() -> str:
  return os.environ.get('OLLAMA_GENERAL_MODEL') or os.environ.get('OLLAMA_MODEL') or 'llama3.1:8b'
