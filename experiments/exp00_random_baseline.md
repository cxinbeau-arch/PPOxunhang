# Exp00: Random / Greedy / A* Baseline

## 命令

```bash
python train/evaluate.py --config configs/env_basic.yaml --policy random --episodes 100 --run_name exp00_random --out logs/eval/exp00_random/evaluation.json --replay_dir replays/exp00_random
python train/evaluate.py --config configs/env_basic.yaml --policy greedy --episodes 100 --run_name exp00_greedy --out logs/eval/exp00_greedy/evaluation.json --save_replay replays/exp00_greedy
python train/evaluate.py --config configs/env_basic.yaml --policy astar --episodes 100 --run_name exp00_astar --out logs/eval/exp00_astar/evaluation.json --save_replay replays/exp00_astar
```

