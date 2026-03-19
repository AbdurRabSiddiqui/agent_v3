from __future__ import annotations

import os

from agent import get_default_model
from agent_selector import answer_with_selection
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv


HELP = """Commands:
  /help              show this help
  /model             show current model
  /model <name>      force Ollama model (example: /model llama3.1:8b)
  /new               start a fresh session
  /auto              enable automatic model selection
  /exit              quit
"""


def main() -> int:
  load_dotenv()
  model = get_default_model()
  tiny_model = os.environ.get('OLLAMA_TINY_MODEL') or ''
  fast_model = os.environ.get('OLLAMA_FAST_MODEL') or ''
  strong_model = os.environ.get('OLLAMA_STRONG_MODEL') or ''
  ultra_model = os.environ.get('OLLAMA_ULTRA_MODEL') or ''
  verbose = os.environ.get('AGENT_VERBOSE', '0') not in ('0', 'false', 'False')
  is_agent_debug = os.environ.get('AGENT_DEBUG', os.environ.get('ROUTER_DEBUG', '0')) not in ('0', 'false', 'False')
  session_id = 'default'
  histories = {'default': []}
  is_auto = True
  forced_model = None
  last_selected_model = None

  print('Local Agent (LangChain + Ollama)')
  if tiny_model:
    print(f'Model (tiny): {tiny_model}')
  if fast_model:
    print(f'Model (fast): {fast_model}')
  print(f'Model (general): {model}')
  if strong_model:
    print(f'Model (strong): {strong_model}')
  if ultra_model:
    print(f'Model (ultra): {ultra_model}')
  print('Type a task. Use /help for commands.\n')

  while True:
    try:
      user_input = input('> ').strip()
    except (EOFError, KeyboardInterrupt):
      print('\nbye')
      return 0

    if not user_input:
      continue

    if user_input in ('/exit', '/quit'):
      print('bye')
      return 0

    if user_input == '/help':
      print(HELP)
      continue

    if user_input == '/new':
      session_id = f'session-{os.urandom(6).hex()}'
      histories[session_id] = []
      print('Started a new session')
      continue

    if user_input == '/auto':
      is_auto = True
      forced_model = None
      print('Auto model selection: ON')
      continue

    if user_input.startswith('/model'):
      parts = user_input.split(maxsplit=1)
      if len(parts) == 1:
        if forced_model:
          print(f'Model (forced): {forced_model}')
        elif last_selected_model:
          print(f'Model (last selected): {last_selected_model}')
        else:
          print('Model: (none yet)')
        continue

      forced_model = parts[1].strip()
      if not forced_model:
        print('Error: model name required')
        continue

      is_auto = False
      print(f'Set forced model: {forced_model}')
      continue

    try:
      chat_history = histories.get(session_id, [])
      output, selected = answer_with_selection(
        user_input,
        chat_history,
        force_model=(forced_model if not is_auto else None),
        trace_path=os.environ.get('AGENT_TRACE_PATH', ''),
        selection_k=None
      )
      last_selected_model = selected.model or last_selected_model
      if is_agent_debug:
        picked = selected.model or '(fast-path)'
        print(f'[agent] model={picked} reason={selected.reason}')

      print(output or '(no output)')
      histories[session_id] = [
        *chat_history,
        HumanMessage(content=user_input),
        AIMessage(content=output or '')
      ]
    except Exception as err:
      msg = str(err)
      if 'model' in msg and 'not found' in msg:
        print(f'Error: {msg}')
        print('Hint: run `ollama list` and set OLLAMA_*_MODEL in .env to an installed model, or `ollama pull <model>`.')
      else:
        print(f'Error: {err}')

  return 0


if __name__ == '__main__':
  raise SystemExit(main())

