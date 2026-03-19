from __future__ import annotations

import argparse
import os

import agentlightning as agl

from training.agent_lightning_math500.lit_math_router_agent import LitMathRouterAgent, load_math500_tasks


def main() -> int:
  ap = argparse.ArgumentParser(description='Train controller policy for MATH-500 using Agent Lightning + VERL.')
  ap.add_argument('--limit', type=int, default=64, help='Number of tasks for training (default: 64)')
  ap.add_argument('--lambda-t', type=float, default=0.0, help='Latency penalty weight')
  ap.add_argument('--lambda-e', type=float, default=0.0, help='Energy proxy penalty weight')
  ap.add_argument('--candidates', default='', help='Comma-separated candidate models; defaults to env MODEL_SELECT_CANDIDATES set used by agent_selector')
  ap.add_argument('--controller-model', default='Qwen/Qwen2.5-1.5B-Instruct', help='HF model path for controller LLM')
  ap.add_argument('--n-runners', type=int, default=1, help='Number of agent runners (recommend 1 for stable measurements)')
  args = ap.parse_args()

  candidates = [c.strip() for c in (args.candidates or '').split(',') if c.strip()]
  if not candidates:
    # Simple default: expect caller to set candidates explicitly.
    raise SystemExit('Provide --candidates "model1,model2,model3" for RL training.')

  agent = LitMathRouterAgent(
    candidates=candidates,
    lambda_t=float(args.lambda_t),
    lambda_e=float(args.lambda_e),
    max_steps=2
  )

  train_dataset = load_math500_tasks(limit=int(args.limit))

  verl_config = {
    'algorithm': {'adv_estimator': 'grpo', 'use_kl_in_reward': False},
    'data': {
      'train_batch_size': 16,
      'max_prompt_length': 4096,
      'max_response_length': 512
    },
    'actor_rollout_ref': {
      'rollout': {
        'name': 'vllm',
        'n': 4,
        'multi_turn': {'format': 'hermes'},
        'tensor_model_parallel_size': 1,
        'gpu_memory_utilization': 0.6
      },
      'actor': {
        'ppo_mini_batch_size': 16,
        'ppo_micro_batch_size_per_gpu': 4,
        'optim': {'lr': 1e-6},
        'use_kl_loss': False,
        'kl_loss_coef': 0.0,
        'entropy_coeff': 0
      },
      'ref': {'log_prob_micro_batch_size_per_gpu': 8},
      'model': {
        'path': str(args.controller_model),
        'use_remove_padding': True,
        'enable_gradient_checkpointing': True
      }
    },
    'trainer': {
      'n_gpus_per_node': int(os.environ.get('AGL_N_GPUS', '1')),
      'val_before_train': False,
      'test_freq': 0,
      'save_freq': 64,
      'total_epochs': 1
    }
  }

  algorithm = agl.VERL(config=verl_config)
  trainer = agl.Trainer(n_runners=int(args.n_runners), algorithm=algorithm)
  trainer.fit(agent, train_dataset=train_dataset)  # type: ignore[arg-type]
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

