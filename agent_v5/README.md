# Tool-capable Ollama Agent (Python)

A local, tool-capable agent that runs on **Ollama** and uses a **HELIOS-inspired early-stop** selection loop to reduce cost (latency/energy) by escalating to stronger models only when needed. Everything is instrumented with JSONL traces and optional live GPU power logging.

## What it can do

- Math via a calculator tool
- Tell you the current time
- List files in a directory (restricted to this project folder)
- Read small text files (restricted to this project folder)
- Write/append small text files (restricted to this project folder)
- Estimate electric bills and recommend a rate plan (simple simulator)
- Run MATH-500 evaluation with per-question traces and optional GPU power logging

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

## Testing playbook (recommended)

This section is the quickest way to validate the system end-to-end: **baseline vs early-stop**, traces, GPU CSV, and aggregated results.

### Step 0: sanity checks

- Ensure Ollama is running:

```bash
ollama list
```

- Ensure `AGENT_TRACE_PATH` is set (in `.env` or exported):

```bash
export AGENT_TRACE_PATH=logs/agent_trace.jsonl
```

### Step 1: run MATH-500 (baseline tournament)

Force the legacy multi-draft + judge tournament:

```bash
export MODEL_SELECT_FORCE_TOURNAMENT=1
export MODEL_SELECT_EARLY_STOP=0
python eval_math500.py --limit 20 --show-model --log-jsonl logs/math500_baseline.jsonl
```

Outputs:

- `logs/agent_trace.jsonl` (selection traces, including `agent.select.draft_*` + `agent.select.judge_*`)
- `logs/math500_baseline.jsonl` (per-item correctness/duration)

### Step 2: run MATH-500 (HELIOS-inspired early-stop)

Enable early-stop and verifier:

```bash
export MODEL_SELECT_EARLY_STOP=1
export MODEL_SELECT_FORCE_TOURNAMENT=0
export MODEL_SELECT_VERIFY=1
export OLLAMA_VERIFY_MODEL="${OLLAMA_VERIFY_MODEL:-${OLLAMA_TINY_MODEL:-deepseek-r1:8b}}"
python eval_math500.py --limit 20 --show-model --log-jsonl logs/math500_early_stop.jsonl
```

What to look for in `logs/agent_trace.jsonl`:

- `agent.select.early_stop.start`
- `agent.select.verify`
- `agent.select.early_stop.accept` (fast path)
- `agent.select.early_stop.fallback_to_tournament` (safety net)

### Step 3: aggregate results (with or without GPU CSV)

If you have GPU CSV, include it:

```bash
python eval_report.py --benchmark math500 \
  --eval-jsonl logs/math500_early_stop.jsonl \
  --agent-trace logs/agent_trace.jsonl \
  --gpu-csv logs/gpu_energy.csv \
  --out-csv logs/results_early_stop.csv \
  --out-jsonl logs/results_early_stop.jsonl
```

If you do not have GPU CSV yet, you can still aggregate by passing any existing CSV path (or record a short run first). The early-stop fields appear as:

- `early_stop_used`, `early_stop_fallback`, `early_stop_accept_attempt`, `early_stop_verifier_model`
- `verify_count`, `verify_accept_count`, `verify_max_confidence`

### Step 4 (optional): GPU power logging for MATH-500

Start the GPU logger (terminal A):

```bash
python gpu_energy_logger.py --duration 0 --interval 0.2 --efficient \
  --csv logs/gpu_energy.csv \
  --label-path logs/power_label.json \
  --save-plots --out-dir logs
```

Run a workload (terminal B):

```bash
export AGENT_TRACE_PATH=logs/agent_trace.jsonl
python eval_math500.py --limit 20 --show-model --log-jsonl logs/math500.jsonl
```

Stop the GPU logger with Ctrl+C.

Plots written:

- `logs/gpu_power.png` (orange only: **effective power**; paginated)
- `logs/gpu_agent_power.png` (orange only: **effective power**; paginated)

Then aggregate:

```bash
python eval_report.py --benchmark math500 \
  --eval-jsonl logs/math500.jsonl \
  --agent-trace logs/agent_trace.jsonl \
  --gpu-csv logs/gpu_energy.csv \
  --out-csv logs/results.csv \
  --out-jsonl logs/results.jsonl
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
- `agentlightning` (optional, only for `training/train_math500_agent_lightning.py`)

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
- `agent_selector.py`: agent-owned model selection. Uses **early-stop** (draft → verify → accept/escalate) for non-tool prompts and falls back to **draft+judge tournament** when uncertain. Escalates tool-agent models on failures/stops. Writes selection traces to `AGENT_TRACE_PATH`.
- `agent.py`: tool-agent loop. Prefers native tool-calling when the installed LangChain/Ollama stack supports it, and falls back to a strict JSON-blob protocol otherwise.
- `tools.py`: general sandboxed tools (calculator, file read/write, etc.) + imports power tools.
- `rate_tools.py`: deterministic “power plan” tools used by both the agent and the CLI fast-path.
- `specialists.py`: deterministic fast-paths to avoid LLM calls when the request is safely solvable without a model.
- `verifier.py`: verifier prompt + strict JSON parser used by early-stop.
- `eval_math500.py`: evaluates the agent’s per-question model selection on MATH-500 (so you can inspect what it chose each item).
- `ollama_utils.py`: shared `ollama list` discovery + TTL cache + model fallback.
- `trace_utils.py`: JSONL trace helper used by agent/eval.
- `gpu_energy_logger.py`: optional NVIDIA NVML power logger (CSV + plots; CSV also tracks cumulative energy fields).
- `training/`: Agent Lightning-compatible training utilities for policy experimentation.

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
- `MODEL_SELECT_EARLY_STOP=1`: enable HELIOS-inspired early-stop for non-tool prompts (default: 1)
- `MODEL_SELECT_FORCE_TOURNAMENT=0`: force draft+judge tournament even when early-stop is enabled (default: 0)
- `MODEL_SELECT_VERIFY=1`: enable verifier calls (default: 1)
- `OLLAMA_VERIFY_MODEL`: preferred verifier model (default: `OLLAMA_TINY_MODEL`/`OLLAMA_FAST_MODEL`/smallest candidate)
- `MODEL_SELECT_VERIFY_TIMEOUT_S=25`: verifier timeout (seconds)
- `MODEL_SELECT_ACCEPT_THRESHOLD=0.78`: verifier accept threshold (confidence)
- `MODEL_SELECT_REJECT_THRESHOLD=0.78`: verifier reject/escalate threshold (confidence)
- `AGENT_TRACE_PATH=logs/agent_trace.jsonl`: write selection + tool-agent + eval traces (JSONL)
- `POWER_LABEL_PATH=logs/power_label.json`: live JSON label file for GPU power logging
- `AGENT_MAX_ITERATIONS=10`: tool-agent loop limit
- `AGENT_MAX_HISTORY_MESSAGES=12`: truncate long chat histories

## HELIOS-inspired early-stop selection (how it works)

For **non-tool** prompts, the agent tries models from smallest → largest:

- **draft**: generate an answer from a cheaper model
- **verify**: a small verifier model returns strict JSON: `{"accept": bool, "confidence": 0..1, "reason": "..."}`
- **accept** if the verifier is confident enough; otherwise **escalate** to a stronger model
- if still uncertain (or everything fails), fall back to the existing **draft+judge tournament**

This reduces average cost while keeping a conservative reliability backstop.

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

## GPU power logging (optional, NVIDIA)

Record live GPU power (and cumulative energy fields in CSV) and save plots at the end:

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

- `logs/gpu_power.png` (orange only: **effective power**)
- `logs/gpu_power_02.png`, ... (pagination: max 8 prompts per image)
- `logs/gpu_agent_power.png` (orange only: **effective power**, paginated similarly)
- If `--trace-jsonl` is provided to the GPU logger: `logs/gpu_power_annotated.png`, `logs/gpu_power_annotated_02.png`, ...

Notes:

- The CSV still includes energy fields for later analysis (`total_energy_j`, `effective_energy_j`), but the logger does **not** generate energy PNGs.

## Agent Lightning policy experiments (MATH-500)

This repo includes minimal training utilities compatible with Agent Lightning to experiment with accept/escalate policies on the **first 400** MATH-500 items.

Build simple per-model effective power priors from an existing labeled GPU run:

```bash
python training/build_cost_priors.py --gpu-csv logs/gpu_energy.csv --out-json logs/model_power_priors.json
```

Run the Agent Lightning-compatible harness (baseline algorithm for smoke-testing + reward shaping):

```bash
python training/train_math500_agent_lightning.py --limit 400 --train-n 320 --power-priors logs/model_power_priors.json
```

Reward shaping env vars (optional):

- `RL_LAMBDA_STRONG`: penalty when the run escalates to the strong model (default `0.05`)
- `RL_LAMBDA_TIME`: penalty per second of wall time (default `0.0`)
- `RL_LAMBDA_ENERGY`: penalty per Joule of estimated effective energy (default `0.0`)

Full RL training with Agent Lightning typically uses the VERL algorithm (`pip install agentlightning[verl]`) and is best run on Linux with a compatible PyTorch/vLLM stack.

### Training a policy model (weights) with Agent Lightning + VERL

If you want to train an actual **policy model** (update weights), use Agent Lightning’s VERL algorithm.

Requirements (practical):

- Linux + NVIDIA GPU
- A working PyTorch + vLLM stack (Agent Lightning VERL docs recommend following their installation guide)
- Install: `pip install "agentlightning[verl]"`

This repo includes `training/train_policy_model_verl.py`, which trains a **verifier/policy model** that outputs strict JSON:

`{"accept": <bool>, "confidence": <0..1>, "reason": "<short>"}`

That policy decides whether to accept the **small-model** draft or escalate to the **strong** model; reward is computed from MATH-500 correctness minus an escalation penalty.

Example:

```bash
export VERL_POLICY_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export VERL_EPOCHS=1
export VERL_N_GPUS=1
export RL_LAMBDA_STRONG=0.05
python training/train_policy_model_verl.py --limit 400 --train-n 320
```

Tune:

- `VERL_POLICY_MODEL`: base model to fine-tune (HF path)
- `VERL_TRAIN_BATCH_SIZE`, `VERL_MAX_PROMPT_LENGTH`, `VERL_MAX_RESPONSE_LENGTH`
- `VERL_GPU_UTIL`, `VERL_LR`, `VERL_SAMPLES_PER_PROMPT`

### Using your already collected 400-question dataset

If you already have logs from a prior run (first 400 MATH-500 items), you can turn them into an **offline dataset** without re-running inference:

Required files (typically in `logs/`):

- `logs/math500.jsonl` (per-item `math500.item` results)
- `logs/agent_trace.jsonl` (draft/judge trace events)
- `logs/gpu_energy.csv` (optional; for energy fields)

Build an offline dataset JSONL:

```bash
python training/make_offline_dataset_from_logs.py \
  --limit 400 \
  --eval-jsonl logs/math500.jsonl \
  --agent-trace logs/agent_trace.jsonl \
  --gpu-csv logs/gpu_energy.csv \
  --out-jsonl logs/math500_offline_dataset.jsonl
```

This outputs fields like `draft0_correct_int`, which is useful to estimate “how often we could have accepted the first (smallest) draft” and to tune early-stop policies before collecting new on-policy rollouts.

### Recommended workflow with your existing 400-run dataset

1) **Offline analysis dataset** (no re-run):

```bash
python training/make_offline_dataset_from_logs.py \
  --limit 400 \
  --eval-jsonl logs/math500.jsonl \
  --agent-trace logs/agent_trace.jsonl \
  --gpu-csv logs/gpu_energy.csv \
  --out-jsonl logs/math500_offline_dataset.jsonl
```

2) **Build cost priors** (optional, from the same GPU CSV):

```bash
python training/build_cost_priors.py --gpu-csv logs/gpu_energy.csv --out-json logs/model_power_priors.json
```

3) **Run policy experiments** (Agent Lightning-compatible harness):

```bash
python training/train_math500_agent_lightning.py \
  --limit 400 \
  --train-n 320 \
  --power-priors logs/model_power_priors.json
```

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
