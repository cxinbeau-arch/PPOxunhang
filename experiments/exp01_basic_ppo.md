# Exp01: Basic PPO

## 命令

```bash
python train/train_ppo.py --config configs/env_basic.yaml --total_timesteps 200000 --run_name exp01_basic_ppo
```

## 验收

- `success_rate >= 0.80`
- `task_completion_rate >= 0.90`
- monitor reward 曲线整体上升


