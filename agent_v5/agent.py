from __future__ import annotations

import importlib
import json
import os
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
try:
  from langchain_core.messages import ToolMessage
except Exception:
  ToolMessage = None  # type: ignore

from tools import TOOLS
from ollama_utils import fallback_to_installed_model
from trace_utils import append_jsonl


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


def _render_tools_text(tools) -> str:
  try:
    from langchain_core.tools.render import render_text_description_and_args
    return render_text_description_and_args(list(tools))
  except Exception:
    pass

  lines = []
  for t in tools:
    name = getattr(t, 'name', 'tool')
    desc = getattr(t, 'description', '') or ''
    lines.append(f'- {name}: {desc}')
  return '\n'.join(lines)


def _parse_action_blob(text: str) -> Optional[Dict[str, Any]]:
  if not text:
    return None

  codeblock = re.findall(r'```(?:json)?\s*({[\s\S]*?})\s*```', text, flags=re.I)
  candidates = []
  if codeblock:
    candidates.extend(codeblock)

  # last-resort: try to find a JSON object in plain text
  m = re.search(r'({[\s\S]*})', text)
  if m:
    candidates.append(m.group(1))

  for raw in reversed(candidates):
    try:
      obj = json.loads(raw)
    except Exception:
      continue
    if isinstance(obj, dict) and 'action' in obj:
      if 'action_input' not in obj:
        obj['action_input'] = ''
      return obj

  return None


def _tool_invoke(tool, action_input: Any) -> str:
  try:
    if isinstance(action_input, dict):
      return str(tool.invoke(action_input))

    # Try to infer the single parameter name
    key = 'input'
    schema = getattr(tool, 'args_schema', None)
    if schema is not None and hasattr(schema, 'model_fields'):
      fields = list(schema.model_fields.keys())
      if fields:
        key = fields[0]
    elif hasattr(tool, 'args') and isinstance(tool.args, dict) and tool.args:
      key = list(tool.args.keys())[0]

    return str(tool.invoke({key: action_input}))
  except Exception as err:
    return f'Error: {err}'


class ToolLoopExecutor:
  def __init__(self, *, llm, tools, max_iterations: int = 10, verbose: bool = False):
    self.llm = llm
    self.tools = list(tools)
    self.max_iterations = max_iterations
    self.verbose = verbose
    self.tool_map = {t.name: t for t in self.tools if hasattr(t, 'name')}
    self.llm_with_tools = self._try_bind_tools()

  def _try_bind_tools(self):
    binder = getattr(self.llm, 'bind_tools', None)
    if not callable(binder):
      return None
    try:
      return binder(self.tools)
    except Exception:
      return None

  def _truncate_history(self, chat_history):
    try:
      limit = int(os.environ.get('AGENT_MAX_HISTORY_MESSAGES', '12') or '12')
    except Exception:
      limit = 12
    if limit <= 0:
      return []
    return list(chat_history)[-limit:]

  def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
    user_input = (inputs or {}).get('input', '')
    chat_history = self._truncate_history((inputs or {}).get('chat_history') or [])
    trace_path = os.environ.get('AGENT_TRACE_PATH', '')
    start_time = __import__('time').time()

    tools_text = _render_tools_text(self.tools)
    tool_names = ', '.join(sorted(self.tool_map.keys()))

    system_prompt_native = '\n'.join([
      'You are a reliable local tool-using assistant running via Ollama.',
      '',
      'Operating principles:',
      '- Goal: solve the user request correctly with minimal latency/compute.',
      '- Minimize tokens and tool calls. Use a quick plan, then execute.',
      '- Prefer direct answers when you are confident and no external state is needed.',
      '- Use tools only when they are the fastest/safest path to truth (files/time/exact calculation).',
      '- Never fabricate tool outputs. Only use observations returned by tools.',
      '- If a tool errors, fix the input and retry once. If it still fails, explain the failure and next best step.',
      '- If a single missing detail blocks progress, ask ONE targeted question and stop.',
      '- If the user request is unsafe or outside the sandbox, explain why and offer a safe alternative.',
      '',
      'Security / sandbox:',
      '- File tools are restricted to this project folder; do not attempt to access outside paths.',
      '- Do not attempt network access unless the environment explicitly provides a tool for it.',
      '',
      'Output rules:',
      '- When you can answer directly, answer concisely (no filler).',
      '- When tools are needed, make the minimum necessary tool call(s), then produce the final answer.',
      '- Do not expose private scratch work. Provide only results and essential context.',
      '',
      'Coding / editing rules:',
      '- If asked to change code, make the smallest correct change.',
      '- Preserve existing style and structure.',
      '- Avoid breaking changes unless explicitly requested.',
      '',
      'Tools:',
      tools_text,
      '',
      f'Tool names: {tool_names}'
    ])

    system_prompt_legacy = system_prompt_native + '\n' + '\n'.join([
      '',
      'Legacy tool protocol (STRICT):',
      '- If you need to use a tool, respond with EXACTLY ONE JSON action blob and NOTHING ELSE.',
      '- Do NOT wrap it in extra text.',
      '- Use action "Final Answer" only when you are done.',
      '- For tool calls, action MUST be one of the Tool names listed above.',
      '- For the final response, action_input MUST be the final user-facing answer text.',
      '',
      'JSON action blob format:',
      '```json',
      '{',
      '  "action": "tool_name_or_Final Answer",',
      '  "action_input": "input string or object"',
      '}',
      '```'
    ])

    if self.llm_with_tools and ToolMessage is not None:
      messages = [
        SystemMessage(content=system_prompt_native),
        *chat_history,
        HumanMessage(content=user_input)
      ]

      repeats = {}
      for _ in range(self.max_iterations):
        res = self.llm_with_tools.invoke(messages)
        tool_calls = getattr(res, 'tool_calls', None) or getattr(res, 'additional_kwargs', {}).get('tool_calls')

        if tool_calls:
          if trace_path:
            append_jsonl(trace_path, {
              'event': 'agent.tool_calls',
              'mode': 'native',
              'tool_calls': tool_calls
            })
          for call in tool_calls:
            name = (call.get('name') if isinstance(call, dict) else None) or ''
            args = (call.get('args') if isinstance(call, dict) else None) or {}
            call_id = (call.get('id') if isinstance(call, dict) else None) or name

            signature = json.dumps({'name': name, 'args': args}, sort_keys=True, ensure_ascii=False)
            repeats[signature] = repeats.get(signature, 0) + 1
            if repeats[signature] >= 4:
              return {'output': 'Agent stopped: repeated the same tool call too many times.'}

            tool = self.tool_map.get(name)
            observation = _tool_invoke(tool, args) if tool else f'Error: unknown tool "{name}"'
            messages.append(ToolMessage(content=str(observation), tool_call_id=str(call_id)))
          continue

        text = (getattr(res, 'content', None) or str(res)).strip()
        if text:
          if trace_path:
            append_jsonl(trace_path, {
              'event': 'agent.final',
              'mode': 'native',
              'duration_ms': int((__import__('time').time() - start_time) * 1000),
              'input_chars': len(user_input or ''),
              'output_chars': len(text or '')
            })
          return {'output': text}

      return {'output': 'Agent stopped due to iteration limit or time limit.'}

    scratchpad = ''
    repeats = {}
    for _ in range(self.max_iterations):
      messages = [
        SystemMessage(content=system_prompt_legacy),
        *chat_history,
        HumanMessage(content=f'{user_input}\n\n{scratchpad}'.strip())
      ]

      res = self.llm.invoke(messages)
      text = (getattr(res, 'content', None) or str(res)).strip()

      action = _parse_action_blob(text)
      if not action:
        return {'output': text}

      name = str(action.get('action', '')).strip()
      action_input = action.get('action_input', '')

      if name.lower() in ('final answer', 'final'):
        return {'output': str(action_input).strip()}

      signature = json.dumps({'action': name, 'action_input': action_input}, sort_keys=True, ensure_ascii=False)
      repeats[signature] = repeats.get(signature, 0) + 1
      if repeats[signature] >= 4:
        return {'output': 'Agent stopped: repeated the same tool call too many times.'}

      tool = self.tool_map.get(name)
      observation = _tool_invoke(tool, action_input) if tool else f'Error: unknown tool "{name}"'
      if trace_path:
        append_jsonl(trace_path, {
          'event': 'agent.tool_call',
          'mode': 'legacy',
          'tool': name,
          'action_input': action_input,
          'observation': observation[:5000]
        })

      scratchpad = '\n'.join([
        scratchpad,
        'Action:',
        '```json',
        json.dumps({'action': name, 'action_input': action_input}, ensure_ascii=False),
        '```',
        f'Observation: {observation}',
        'If the observation is an error, correct the action_input and try again.'
      ]).strip()

    return {'output': 'Agent stopped due to iteration limit or time limit.'}


def build_agent_executor(*, model: str, temperature: float = 0.2, verbose: bool = True):
  llm = build_llm(model=model, temperature=temperature)
  try:
    max_iters = int(os.environ.get('AGENT_MAX_ITERATIONS', '10') or '10')
  except Exception:
    max_iters = 10
  return ToolLoopExecutor(llm=llm, tools=TOOLS, max_iterations=max_iters, verbose=verbose)


def get_default_model() -> str:
  return os.environ.get('OLLAMA_GENERAL_MODEL') or os.environ.get('OLLAMA_MODEL') or 'llama3.1:8b'


def run_task(task: str, *, model: Optional[str] = None) -> str:
  executor = build_agent_executor(model=model or get_default_model(), verbose=False)
  result = executor.invoke({'input': task, 'chat_history': []})
  return result.get('output', '')


def llm_fallback_answer(user_input: str, chat_history, *, model: str, temperature: float = 0.2) -> str:
  llm = build_llm(model=model, temperature=temperature)
  system_prompt = '\n'.join([
    'Answer concisely and directly.',
    'If it is a math problem, return only the final answer (no steps).',
    'If critical information is missing, ask a single clarifying question.'
  ])
  messages = [
    SystemMessage(content=system_prompt),
    *list(chat_history),
    HumanMessage(content=user_input)
  ]
  res = llm.invoke(messages)
  return (getattr(res, 'content', None) or str(res)).strip()


