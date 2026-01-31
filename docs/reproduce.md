## Reproducing results

This document describes how to reproduce results for each supported dataset.

For the meaning of script arguments and configuration fields, see `docs/configuration_reference.md`.

### Common configuration (applies to all datasets)

1. In `.env`, set:
   - `MAX_TASK_EXECUTION_CNT=8`
   - `MAX_WORKER_RECURSION_LIMIT=10`
   - `WORKER_TOOL_ENHANCE_INTERVAL=5`

2. In `conf.yaml`, set:
   - `BASIC_MODEL.model` and `CLUSTER_MODEL.model` to `gemini-3-pro`
   - `VISION_MODEL.model` and `SUMMARIZE_MODEL.model` to `gemini-3-flash`
   - `TOOL_ANALYZE_MODEL.model` to `gpt-5-mini`

3. Run the evolution loop:

```bash
./scripts/evolve.sh --dataset <DATASET> --run_name <RUN_NAME> --batch_size 16
```

4. After the run completes, evaluate:

```bash
uv run scripts/evaluate.py --benchmark <BENCHMARK> --predictions output/<RUN_NAME>/predictions.jsonl --dataset <DATASET_FILE_OR_NAME> --max-workers 30
```

### HLE

1. `.env` settings: see “Common configuration”.
2. `conf.yaml` settings: see “Common configuration”.
3. Run:

```bash
./scripts/evolve.sh --dataset HLE --run_name hle --batch_size 16
```

4. Evaluate:

```bash
uv run scripts/evaluate.py --benchmark hle --predictions output/hle/predictions.jsonl --dataset dataset/HLE/hle.json --max-workers 30
```

### DeepSearchQA (DEEPSEARCHQA)

1. `.env` settings: see “Common configuration”.
2. `conf.yaml` settings: see “Common configuration”.
3. Run:

```bash
./scripts/evolve.sh --dataset DEEPSEARCHQA --run_name deepsearchqa --batch_size 16
```

4. Evaluate:

```bash
uv run scripts/evaluate.py --benchmark deepsearchqa --predictions output/deepsearchqa/predictions.jsonl --dataset dataset/DEEPSEARCHQA/DSQA-full.json --max-workers 30
```

### FinSearchComp (FINSEARCHCOMP)

1. `.env` settings: see “Common configuration”.
2. `conf.yaml` settings: see “Common configuration”.
3. Run:

```bash
./scripts/evolve.sh --dataset FINSEARCHCOMP --run_name finsearchcomp --batch_size 16
```

4. Evaluate:

```bash
uv run scripts/evaluate.py --benchmark finsearchcomp --predictions output/finsearchcomp/predictions.jsonl --dataset dataset/FinSearchComp/t2_t3_questions.json --max-workers 30
```

### XBENCH-deepsearch

1. `.env` settings: see “Common configuration”.
2. `conf.yaml` settings: see “Common configuration”.
3. Run:

```bash
./scripts/evolve.sh --dataset XBENCH-deepsearch --run_name xbench-deepsearch --batch_size 16
```

4. Evaluate:

```bash
uv run scripts/evaluate.py --benchmark xbench --predictions output/xbench-deepsearch/predictions.jsonl --dataset dataset/XBENCH/DeepSearch-2510.json --max-workers 30
```

### XBENCH-scienceqa

1. `.env` settings: see “Common configuration”.
2. `conf.yaml` settings: see “Common configuration”.
3. Run:

```bash
./scripts/evolve.sh --dataset XBENCH-scienceqa --run_name xbench-scienceqa --batch_size 16
```

4. Evaluate:

```bash
uv run scripts/evaluate.py --benchmark xbench --predictions output/xbench-scienceqa/predictions.jsonl --dataset dataset/XBENCH/ScienceQA.json --max-workers 30
```

### About the “args reference” document

The placeholder file referenced above (`docs/cli_reference.md`) is intended to explain:
- The meaning of `./scripts/evolve.sh` arguments (e.g. `--dataset`, `--run_name`, `--batch_size`, `--start`, `--train_steps`, `--merge_policy`, `--timeout`)
- The meaning of `scripts/evaluate.py` arguments (e.g. `--benchmark`, `--predictions`, `--dataset`, `--max-workers`, `--eval-model`)
- The purpose of key `.env` variables and `conf.yaml` model blocks

Alternative good names for that future document:
- `docs/cli_reference.md`
- `docs/script_args.md`
- `docs/runner_reference.md`
