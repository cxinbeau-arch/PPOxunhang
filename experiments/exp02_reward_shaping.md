# Exp02: Reward Shaping

## 目的

比较 sparse reward 与 dense reward 对 PPO 的影响。

## 命令

```bash
python train/train_ppo.py --config configs/env_basic.yaml --total_timesteps 200000 --run_name exp02a_sparse
python train/train_ppo.py --config configs/env_obstacle.yaml --total_timesteps 300000 --run_name exp02b_dense
```
