# LangChain + Ollama Basic Agent (Python)

A minimal “agentic” AI that runs locally via **Ollama** and can carry out simple tasks using tools.

## What it can do

- Math via a calculator tool
- Tell you the current time
- List files in a directory (restricted to this project folder)
- Read small text files (restricted to this project folder)
- Write/append small text files (restricted to this project folder)
- Estimate electric bills and recommend a rate plan (simple simulator)

## Prerequisites

- Python 3.10+ recommended
- Ollama installed and running
- Pull an Ollama model (example):

```bash
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:32b
```

List your installed local models (these names are what you put in `.env`):

```bash
ollama list
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your env file:

```bash
cp env.sample .env
```

## Dependencies

Python packages (from `requirements.txt`):

- `langchain`
- `langchain-core`
- `langchain-community`
- `langchain-ollama`
- `pydantic`
- `datasets`
- `tqdm`
- `sympy`
- `python-dotenv`
- `ipykernel`
- `matplotlib`
- `nvidia-ml-py` (optional, only for `gpu_energy_logger.py`)

## Run

```bash
python agent_cli.py
```

Then type a task, e.g.:

- "What files are in this folder?"
- "Create a note file called notes/todo.txt with a 5-item todo list."
- "Read notes/todo.txt and summarize it."
- "What is (17\*23) + 91?"

## CLI commands (in-app)

Once `python agent_cli.py` is running, you can use:

- `/help`: show help
- `/new`: start a fresh session (clears chat history)
- `/auto`: enable automatic agent model selection
- `/model`: show the currently forced/last-selected model
- `/model <name>`: force a specific Ollama model (disables auto selection)
- `/exit`: quit

## Feature examples (copy/paste)

All examples below are meant to be pasted into the running CLI prompt (`>`).

### Math (deterministic fast-path for simple expressions)

```text
> what is (17*23) + 91?
```

```text
> calculate 2^20
```

### Time (tool)

```text
> what time is it in UTC?
```

### List files (tool; restricted to this repo folder)

```text
> list files in this folder
```

```text
> list files in notes
```

### Read a text file (tool; restricted to this repo folder)

```text
> read notes/todo.txt
```

### Write a text file (tool; restricted to this repo folder)

```text
> create a note file called notes/todo.txt with a 5-item todo list
```

### Append to a text file (tool; restricted to this repo folder)

```text
> append a new item "buy coffee filters" to notes/todo.txt
```

### Rate plan selection (tools)

List available plans:

```text
> list available rate plans
```

Recommend cheapest plan for a 24-hour usage profile (24 numbers):

```text
> recommend the cheapest plan for this hourly usage profile: [0.8,0.7,0.6,0.6,0.5,0.5,0.6,0.9,1.2,1.4,1.6,1.7,1.8,1.7,1.6,1.5,1.4,1.6,1.8,1.7,1.4,1.2,1.0,0.9]
```

Same, but include solar generation (24 numbers):

```text
> recommend the cheapest plan for this usage profile: [0.8,0.7,0.6,0.6,0.5,0.5,0.6,0.9,1.2,1.4,1.6,1.7,1.8,1.7,1.6,1.5,1.4,1.6,1.8,1.7,1.4,1.2,1.0,0.9]. my solar hourly generation is: [0,0,0,0,0,0,0.1,0.3,0.6,0.9,1.1,1.2,1.3,1.2,1.0,0.7,0.4,0.2,0,0,0,0,0,0]
```

Estimate a daily bill for a specific plan:

```text
> estimate my daily bill for plan "tou-saver" with hourly usage: [0.8,0.7,0.6,0.6,0.5,0.5,0.6,0.9,1.2,1.4,1.6,1.7,1.8,1.7,1.6,1.5,1.4,1.6,1.8,1.7,1.4,1.2,1.0,0.9]
```

## Notes

- All filesystem tools are sandboxed to this repo directory (they can’t access outside paths).
- This setup is **Ollama-only** (no API keys needed). The agent picks between your local Ollama models (and can compare multiple) for speed vs quality.

## How this folder works (high level)

- `agent_cli.py`: interactive CLI entrypoint. Uses the agent’s model selector per user message, then either answers directly or uses the tool-agent loop.
- `agent_selector.py`: agent-owned model selection. Compares 2–3 installed Ollama models (draft+judge) for non-tool prompts, and escalates tool-agent models on failures/stops. Writes selection traces to `AGENT_TRACE_PATH`.
- `agent.py`: tool-agent loop. Prefers native tool-calling when the installed LangChain/Ollama stack supports it, and falls back to a strict JSON-blob protocol otherwise.
- `tools.py`: general sandboxed tools (calculator, file read/write, etc.) + imports power tools.
- `rate_tools.py`: deterministic “power plan” tools used by both the agent and the CLI fast-path.
- `specialists.py`: deterministic fast-paths to avoid LLM calls when the request is safely solvable without a model.
- `eval_math500.py`: evaluates the agent’s per-question model selection on MATH-500 (so you can inspect what it chose each item).
- `ollama_utils.py`: shared `ollama list` discovery + TTL cache + model fallback.
- `trace_utils.py`: JSONL trace helper used by agent/eval.
- `gpu_energy_logger.py`: optional NVIDIA NVML power/energy logger (CSV + plots).

## Model configuration (Ollama)

The agent considers **all installed Ollama models** (from `ollama list`). The env vars below are only **preferences/defaults**:

- `OLLAMA_TINY_MODEL`: fastest for short general Q&A (example: `phi3:mini`)
- `OLLAMA_FAST_MODEL`: small, better quality than tiny (example: `gemma3:4b`)
- `OLLAMA_GENERAL_MODEL`: default for most questions (example: `deepseek-r1:8b`)
- `OLLAMA_TOOL_MODEL`: preferred when the tool-agent needs to plan/call tools (example: `deepseek-r1:8b`)
- `OLLAMA_STRONG_MODEL`: used for hard math / complex requests (example: `deepseek-r1:32b`)
- `OLLAMA_ULTRA_MODEL`: used for very long prompts (example: `deepseek-r1:32b`)

Cloud examples (commented out / not used for local-only):

- `# OLLAMA_STRONG_MODEL=deepseek-v3.1:671b-cloud`
- `# OLLAMA_ULTRA_MODEL=gpt-oss:120b-cloud`

Useful switches:

- `AGENT_DEBUG=1`: prints which model was selected for each prompt (also accepts legacy `ROUTER_DEBUG=1`)
- `MODEL_ALLOW_OLLAMA_CLOUD=1`: allows selecting `*-cloud` models from `ollama list` (some setups require `ollama login`)
- `MODEL_LIST_TTL_S=5`: how long to cache `ollama list` results (seconds)
- `MODEL_SELECT_CANDIDATES=3`: how many installed models to compare per prompt
- `OLLAMA_JUDGE_MODEL`: optional judge model for draft+judge selection (defaults to largest installed)
- `MODEL_SELECT_DRAFT_TIMEOUT_S`: per-draft timeout (seconds) so a hung model doesn’t block progress
- `MODEL_SELECT_JUDGE_TIMEOUT_S`: per-judge timeout (seconds) with automatic judge failover
- `MODEL_SELECT_TOOL_TIMEOUT_S`: per tool-agent attempt timeout (seconds) with escalation
- `AGENT_TRACE_PATH=logs/agent_trace.jsonl`: write selection + tool-agent + eval traces (JSONL)
- `POWER_LABEL_PATH=logs/power_label.json`: live JSON label file for GPU power logging
- `AGENT_MAX_ITERATIONS=10`: tool-agent loop limit
- `AGENT_MAX_HISTORY_MESSAGES=12`: truncate long chat histories

## Tracing / observability (JSONL)

This repo uses lightweight **JSONL** tracing (one JSON object per line) so you can record **selection decisions**, **tool calls**, and **final outputs/metadata** for debugging and evaluation.

- **Enable tracing**: set `AGENT_TRACE_PATH` (example: `logs/agent_trace.jsonl`)
- **Where it writes**: the file paths you set; parent directories are created automatically

Example:

```bash
export AGENT_TRACE_PATH=logs/agent_trace.jsonl
python agent_cli.py
```

To also see the selected model in the CLI:

```bash
export AGENT_DEBUG=1
python agent_cli.py
```

## Power grid rate-plan selection (example)

List plans:

```bash
python agent_cli.py
```

Then ask:

- "List available rate plans."
- "Recommend the cheapest plan for this hourly usage profile: [24 numbers...]."
- "Same, but I have solar. My solar hourly generation is: [24 numbers...]."

Plans live in `rate_plans.sample.json`. Replace it with your real utility tariff data.

## Hugging Face MATH-500 evaluation

This runs the agent’s per-question model selection (multi-model compare) against `HuggingFaceH4/MATH-500` and reports accuracy.

```bash
python eval_math500.py --limit 20
python eval_math500.py
```

Show the chosen model per question:

```bash
python eval_math500.py --limit 20 --show-model
```

Optional per-item JSONL log:

```bash
python eval_math500.py --limit 50 --log-jsonl logs/math500.jsonl
```

## GPU power/energy logging (optional, NVIDIA)

Record live GPU power/energy to a CSV and save plots at the end:

```bash
python gpu_energy_logger.py --duration 60 --efficient --save-plots --out-dir logs
```

Run until Ctrl+C (Jupyter/terminal) and still save plots:

```bash
python gpu_energy_logger.py --duration 0 --interval 0.2 --efficient --save-plots --out-dir logs
```

### What to run (live-labeled power trace per prompt)

Start GPU logger first (live-labeled CSV + plots on stop):

```bash
python gpu_energy_logger.py --duration 0 --interval 0.2 --efficient \
  --csv logs/gpu_energy.csv \
  --label-path logs/power_label.json \
  --save-plots --out-dir logs
```

In another terminal, run your agent workload (this writes `logs/power_label.json` live):

```bash
export AGENT_TRACE_PATH=logs/agent_trace.jsonl
python eval_math500.py --limit 20 --show-model --log-jsonl logs/math500.jsonl
```

Stop the GPU logger:

- If it’s running in the foreground, press **Ctrl+C** in that terminal. It will stop recording and still write the plots (because `--save-plots` is enabled).
- If it’s running in the background, stop it with `kill -INT <pid>` (SIGINT), which is equivalent to Ctrl+C:

```bash
pgrep -af "python gpu_energy_logger.py"
kill -INT <pid>
```

### Results (CSV + JSONL) with correctness + power per prompt

After a benchmark run (e.g. MATH-500) and GPU logging, generate `logs/results.csv` and `logs/results.jsonl`:

```bash
python eval_report.py --benchmark math500 \
  --eval-jsonl logs/math500.jsonl \
  --agent-trace logs/agent_trace.jsonl \
  --gpu-csv logs/gpu_energy.csv \
  --out-csv logs/results.csv \
  --out-jsonl logs/results.jsonl
```

Notes:

- `eval_report.py` uses `logs/agent_trace.jsonl` to fill `selected_model`, `judge_model_used`, `timeouts_count`, and `retries_count`.
- For long runs on A100, start from `env.sample` (it sets higher default timeouts for drafts/judge/tools).

Power plots written by the GPU logger:

- `logs/gpu_power.png` (total + effective)
- `logs/gpu_agent_power.png` (orange only: effective power)
- `logs/gpu_energy.png` (total + effective energy)

### SWE-bench adapter (to JSONL)

If your SWE-bench harness outputs a `.json` or `.jsonl` results file, convert it to the standard format this repo can aggregate:

```bash
python swebench_adapter.py --input /path/to/swebench_results.json --output logs/swebench.jsonl
```

Then generate results:

```bash
python eval_report.py --benchmark swebench \
  --eval-jsonl logs/swebench.jsonl \
  --agent-trace logs/agent_trace.jsonl \
  --gpu-csv logs/gpu_energy.csv \
  --out-csv logs/results.csv \
  --out-jsonl logs/results.jsonl
```

## Change log (what was updated)

- Added agent-owned model selection (`agent_selector.py`) using draft+judge for non-tool prompts and escalation for tool prompts.
- Updated CLI (`agent_cli.py`) to use agent-driven selection (no router dependency) and added `AGENT_DEBUG`.
- Updated MATH-500 eval (`eval_math500.py`) to run per-question agent selection and log decisions to `AGENT_TRACE_PATH`.
- Improved MATH-500 prompt to enforce `\\boxed{...}` output.
