# Local Multi-Model Agent (Ollama) with Lookahead + RL

This project runs a tool-capable agent on top of **local Ollama models**, with full JSONL tracing and optional GPU power logging.

It supports two model-selection modes:

- **Tournament selector (legacy)**: run multiple drafts across models + (optional) judge to pick the best. Accurate but expensive.
- **Lookahead selector (new)**: train a *response-aware* predictor from your collected tournament data, then select a **single model** per prompt (optionally 1-step fallback). Much cheaper.

Then you can train **agentic policies** (controller LLM) with **Agent Lightning + VERL** to explore different energy/accuracy tradeoffs.

## Prerequisites

- Python 3.10+
- Ollama running locally (`OLLAMA_BASE_URL`, default `http://127.0.0.1:11434`)
- Local models pulled via `ollama pull ...`

Optional:

- NVIDIA GPU + driver + NVML + `nvidia-ml-py` for GPU power logging (`gpu_energy_logger.py`)
- Linux + NVIDIA GPU(s) for Agent Lightning + VERL training (recommended)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp env.sample .env
```

## Repo structure (key files)

- `agent_cli.py`: interactive CLI entrypoint
- `agent_selector.py`: model selection + execution (tournament + Lookahead policy mode)
- `agent.py`: tool-using agent loop (LangChain-based)
- `eval_math500.py`: MATH-500 evaluation harness
- `eval_report.py`: aggregates eval JSONL + agent trace + GPU CSV into `results.csv/jsonl`
- `gpu_energy_logger.py`: GPU power logger (CSV + paginated plots; energy is kept in CSV)
- `lookahead/`: Lookahead dataset builder + training + policy
  - `lookahead/dataset_math500.py`: build training data from your trace + GPU logs
  - `lookahead/train_math500.py`: train Lookahead artifacts (joblib)
  - `lookahead/policy.py`: inference-time model selection
- `training/agent_lightning_math500/`: Agent Lightning RL harness (controller LLM)
  - `lit_math_router_agent.py`: LitAgent wrapper
  - `train_rl_verl.py`: VERL training entrypoint (policy RL)

## Model configuration (Ollama)

The agent sees **all installed Ollama models** (`ollama list`). These env vars are preferences:

- `OLLAMA_TINY_MODEL`, `OLLAMA_FAST_MODEL`, `OLLAMA_GENERAL_MODEL`, `OLLAMA_STRONG_MODEL`, `OLLAMA_ULTRA_MODEL`
- `OLLAMA_JUDGE_MODEL`: judge preference for tournament mode
- `MODEL_SELECT_CANDIDATES`: number of candidates considered

## Tracing (JSONL)

Enable tracing by setting:

- `AGENT_TRACE_PATH=logs/agent_trace.jsonl`

You’ll see events like:

- `agent.select.*` (draft/judge attempts, failures, final selection)
- `agent.run.*` (tools/chat runs)
- `math500.item` (per-item benchmark result)
- `agent.policy.lookahead.*` (Lookahead selection decisions)

## GPU power logging (optional, NVIDIA)

### Live-labeled logging

Start the GPU logger first:

```bash
python gpu_energy_logger.py --duration 0 --interval 0.2 --efficient \
  --csv logs/gpu_energy.csv \
  --label-path logs/power_label.json \
  --save-plots --out-dir logs
```

Run a workload in another terminal (this writes `logs/power_label.json` live):

```bash
export AGENT_TRACE_PATH=logs/agent_trace.jsonl
python eval_math500.py --limit 50 --show-model --log-jsonl logs/math500.jsonl
```

Stop the GPU logger with Ctrl+C. It will write plots.

### Plots written

- `logs/gpu_power.png`, `logs/gpu_power_02.png`, ... (orange-only effective power, paginated to 8 prompts/page)
- `logs/gpu_agent_power.png`, `logs/gpu_agent_power_02.png`, ... (same idea, labeled by prompt_idx segments)
- If you pass `--trace-jsonl logs/agent_trace.jsonl`, you also get:
  - `logs/gpu_power_annotated.png`, `logs/gpu_power_annotated_02.png`, ...

Energy is tracked in the CSV, not in PNGs.

## Benchmarking: MATH-500

Run with the default selector:

```bash
python eval_math500.py --limit 20 --show-model --log-jsonl logs/math500.jsonl
```

## Aggregation: correctness + energy + latency

After running a benchmark and GPU logging:

```bash
python eval_report.py --benchmark math500 \
  --eval-jsonl logs/math500.jsonl \
  --agent-trace logs/agent_trace.jsonl \
  --gpu-csv logs/gpu_energy.csv \
  --out-csv logs/results.csv \
  --out-jsonl logs/results.jsonl
```

## Lookahead: train and run single-model selection

### 1) Collect training data (using tournament mode)

Run MATH-500 with the current selector (draft+judge) so you have multi-model drafts in `logs/agent_trace.jsonl`:

```bash
export AGENT_TRACE_PATH=logs/agent_trace.jsonl
python eval_math500.py --limit 400 --log-jsonl logs/math500.jsonl
```

If you also want energy labels per model, run the GPU logger in parallel (above) to produce `logs/gpu_energy.csv`.

### 2) Build Lookahead dataset

This extracts `agent.select.draft` outputs and re-grades them against MATH-500 gold answers. If GPU CSV is present, it also attributes effective energy to each draft by integrating power samples by `(prompt_idx, phase_model)` labels.

```bash
python lookahead/dataset_math500.py \
  --agent-trace logs/agent_trace.jsonl \
  --gpu-csv logs/gpu_energy.csv \
  --out-jsonl logs/lookahead_math500_dataset.jsonl
```

### 3) Train Lookahead artifacts

```bash
python lookahead/train_math500.py \
  --dataset-jsonl logs/lookahead_math500_dataset.jsonl \
  --out models/lookahead_math500.joblib
```

### 4) Run MATH-500 using Lookahead (single model per prompt)

Enable Lookahead policy:

```bash
export MODEL_SELECT_POLICY=lookahead
export LOOKAHEAD_MODEL_PATH=models/lookahead_math500.joblib
export LOOKAHEAD_LAMBDA_T=0.0
export LOOKAHEAD_LAMBDA_E=0.0
export LOOKAHEAD_MIN_P_CORRECT=0.0
export LOOKAHEAD_FALLBACK_MODEL=deepseek-r1:32b
```

Now run eval:

```bash
python eval_math500.py --limit 200 --show-model --log-jsonl logs/math500_lookahead.jsonl
```

Notes:

- Lookahead is currently applied to **MATH-500 prompts only** (detected by the prompt template).
- If Lookahead fails (timeout / invalid boxed output), the system falls back to the tournament path and logs `agent.policy.lookahead.fallback`.

## Policy changes (where to edit)

- **Lookahead inference rule**: `lookahead/policy.py` (`utility = p_correct - lambda_t * latency - lambda_e * energy`)
- **Lookahead training**:
  - dataset: `lookahead/dataset_math500.py`
  - model training: `lookahead/train_math500.py`
- **Runtime integration**: `agent_selector.py` (Lookahead is triggered when `MODEL_SELECT_POLICY=lookahead`)

## Agent Lightning RL (controller policy training)

This trains a **controller LLM** that outputs structured actions like:

```json
{"action":"draft","model":"deepseek-r1:14b"}
```

It uses Agent Lightning’s VERL integration to optimize the controller from rollout rewards, while your **actual answer model remains local Ollama**.

### 1) Install Agent Lightning (training environment)

On Linux + NVIDIA GPUs:

```bash
pip install agentlightning
# Follow Agent Lightning’s VERL installation guide if needed:
# https://microsoft.github.io/agent-lightning/stable/tutorials/installation/
```

### 2) Run VERL training

You must pass the candidate list explicitly (these are your local Ollama answer models).

```bash
python training/agent_lightning_math500/train_rl_verl.py \
  --limit 64 \
  --candidates "deepseek-r1:8b,deepseek-r1:14b,deepseek-r1:32b" \
  --lambda-t 0.0 \
  --lambda-e 0.01 \
  --controller-model "Qwen/Qwen2.5-1.5B-Instruct" \
  --n-runners 1
```

Notes:

- The RL harness currently uses an **energy proxy** based on model size (e.g. `:32b`). True energy measurement under parallel rollouts is non-trivial; start with proxy rewards and validate with real GPU logs offline.
- The controller also receives Lookahead scores (if `MODEL_SELECT_POLICY=lookahead` + artifacts exist) as an optional hint in the observation.

## Note on logs location (your question)

You do **not** need to move/paste your saved `logs/` folder into `lookahead/`.

- Keep your saved logs anywhere.
- Point the dataset builder at them:

```bash
python lookahead/dataset_math500.py \
  --agent-trace /path/to/logs/agent_trace.jsonl \
  --gpu-csv /path/to/logs/gpu_energy.csv \
  --out-jsonl logs/lookahead_math500_dataset.jsonl
```
