# Plan: Make max_length and test/dev split (not ejecuted yet)configurable from Base.yaml

## Context

Currently `max_length` is hardcoded as `300` in 3 `.py` files, requiring manual edits across multiple files whenever switching between 100-frame and 300-frame configurations. Similarly, `CVT_test()` always uses `test_data` — switching to `dev_data` requires editing code. Both should be driven from `Configs/Base.yaml`.

## Changes

### 1. `Configs/Base.yaml` — Add new config keys

Add under `data:`:
```yaml
max_length: 300        # Max skeleton sequence length for padding (was hardcoded)
```

Add under `training:`:
```yaml
eval_split: "test"     # Which split to use for CVT_test inference ("dev" or "test")
```

### 2. `data_operate/dataset.py` — Pass max_length to collate_fn

- `collate_fn` is a module-level function that can't access config
- Fix: convert it to a closure via `make_collate_fn(max_length)` that returns the collate function
- Line 18: Wrap current `collate_fn` inside `make_collate_fn(max_length=300)`
- Callers of `collate_fn` need updating:
  - `data_operate/dataset.py` lines ~54, ~94 (in `make_data_iter` and `__init__` of DataLoader usage)

### 3. `data_operate/dataset.py` — Update DataLoader callers to pass max_length

- `make_data_iter` class (line ~80) needs to accept `max_length` param and pass it to `make_collate_fn`
- All callers of `make_data_iter` need to pass `max_length` from config

### 4. `helpers.py` line 244 — Add max_length parameter to `calculate_dtw`

- Change signature: `def calculate_dtw(references, hypotheses, trg_length, max_length=300)`
- Replace hardcoded `max_length = 300` with the parameter

### 5. `CVT/CVT_prediction.py` — Propagate max_length to calculate_dtw

- `pre_validate_on_data()` calls `calculate_dtw` (line ~112)
- Add `max_length` parameter to `pre_validate_on_data()` and pass it through

### 6. `CVT/CVT_training.py` line 545 — Use self.config for max_length

- In `produce_validation_video()`: replace `max_length = 300` with `max_length = self.config["data"].get("max_length", 300)`

### 7. `CVT/CVT_training.py` — Propagate max_length through training/validation calls

- In `CVTTrainManager.__init__`: store `self.max_length = cfg["data"].get("max_length", 300)`
- In `train_and_validate`: pass max_length to `make_data_iter` and `pre_validate_on_data`
- In `CVT_test()`: read max_length from config and pass to `make_data_iter` and `pre_validate_on_data`

### 8. `CVT/CVT_training.py` line 760 — Make eval split configurable

- In `CVT_test()`: read `eval_split = cfg["training"].get("eval_split", "test")`
- Use `dev_data if eval_split == "dev" else test_data` at line 760

## Files modified (5 files)

1. `Configs/Base.yaml` — Add `max_length` and `eval_split`
2. `data_operate/dataset.py` — `make_collate_fn()` closure + update `make_data_iter`
3. `helpers.py` — Add `max_length` param to `calculate_dtw()`
4. `CVT/CVT_prediction.py` — Add `max_length` param to `pre_validate_on_data()`
5. `CVT/CVT_training.py` — Read from config in all 3 locations, configurable eval split

## Verification

1. Set `max_length: 300` and `eval_split: "test"` in Base.yaml → behavior identical to current
2. Set `max_length: 100` and `eval_split: "dev"` → should work for 100-frame inference on dev set
3. Grep for `max_length = 100` and `max_length = 300` in .py files → should find zero hardcoded values
