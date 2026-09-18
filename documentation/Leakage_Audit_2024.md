# Leakage Audit — SPY 2024 Real-Data Integration Test

Written before any model was trained on the real 2024 dataset, per project policy: audit first, train second.
Each item below is a concrete finding grounded in the actual code, not a generic assertion.

## 1. Feature/target timestamp ordering — SAFE
`data/dataset.py::create_sequences(data, lookback, horizon)` builds `X[i] = data[i:i+lookback]` and
`y[i] = data[i+lookback:i+lookback+horizon]` by pure index arithmetic on a chronologically-sorted array.
Every input index is strictly less than every target index for the same sample, by construction.
`scripts/run_real_data_2024_integration_test.py::verify_no_lookahead` additionally checks this concretely
against the actual calendar **dates** (not just array indices) for every constructible sequence in every
split, and hard-fails (`AssertionError`) on any violation — this is a gate the integration test run must
pass, not a soft warning. Result: see "Verification run" below.

## 2. Future option information — SAFE
`data/processor.py::process_raw_snapshot` and `apply_arbitrage_filters` operate on one day's option chain
at a time (grouped by `timestamp` in `data/dataset.py::import_historical_dataframe`'s per-day loop). No
function in the pipeline has access to any other day's raw contract data while processing a given day's
surface — a given day's IV solve, filtering, and RBF interpolation cannot see tomorrow's quotes.

## 3. Risk-free-rate alignment — SAFE by construction
`data/adapters/spy_options_dataset.py::fetch_historical_risk_free_rate` joins each trading date to the
historical `^IRX` series via `pd.merge_asof(dates_df, irx_df, on="date", direction="backward")` — this
guarantees each date gets the most recent **prior-or-same-day** T-Bill close, never a future one. Verified
by construction of the `merge_asof` call itself (`direction="backward"` is the only direction that cannot
look forward).

## 4. Dividend information — SAFE by construction
`data/adapters/spy_options_dataset.py::compute_trailing_dividend_yield` sums `dividend_amount` over the
window `(date - 365 days, date]` using a strictly backward-scanning two-pointer loop — for date `d`, only
dividends with `dividend_date <= d` are ever included. No dividend that would be paid after `d` can enter
the yield estimate for `d`.

## 5. Normalization/statistics computed using future data — SAFE
`training/train.py::train_model` computes `train_mean = np.mean(train_data)` and
`train_std = np.std(train_data)` from `train_data = dataset[:train_end]` **only** — validation and test
splits are never included in these statistics (confirmed by reading the exact lines: normalization
happens immediately after the three-way chronological split, using only the first slice).

## 6. Interpolation using future observations — SAFE
`data/processor.py::interpolate_to_grid` fits one RBF interpolator per day, using only that day's own
scattered `(kappa, tau, iv)` observations (passed in from that single day's filtered contracts). No
cross-day interpolation or smoothing occurs anywhere in the pipeline.

## 7. Sequence construction (train/validation/test boundary) — SAFE, more conservative than strictly necessary
`train_model` calls `create_sequences` **separately** on each already-sliced split array
(`train_data`, `val_data`, `test_data`), not once on the full series before splitting. Consequence: the
first `lookback` surfaces of the validation and test splits cannot be used as prediction targets within
their own split (there's no history *within that split* to build their lookback window from), even though
that history technically exists in the prior split and would be legitimately available at prediction time
in a live setting. This is a real cost (fewer usable samples near each split's start) but it is the safest
possible design — no sequence's input window can ever span two different splits, so there is no leakage
vector here to fix.

## 8. Train/validation/test contamination — SAFE
The three-way split is a single chronological index slice: `train = dataset[:train_end]`,
`val = dataset[train_end:val_end]`, `test = dataset[val_end:]`. No shuffling of the raw time series happens
before this slice, and the slice indices are computed once from `train_ratio`/`val_ratio` and never
adjusted based on anything computed from the data itself.

## 9. Overlapping sequences across splits — NONE
Direct consequence of #7: since sequences are built independently per already-sliced split, no sequence's
`(input, target)` window can span a split boundary.

## 10. Cached predictions — mitigated
`training/train.py::train_model` now accepts an `output_dir` parameter (added for this task); the real-data
run passes `output_dir="models/real_data"`, so `models/predictions/{model}_pred.npy` (the production
synthetic-trained cache) is never read or written by this experiment — it writes to
`models/real_data/predictions/` instead.

## 11. Checkpoint reuse — mitigated
Two independent guards, so a config or code mistake in one doesn't silently promote a real-data checkpoint
to production: (a) `output_dir="models/real_data"` means checkpoints are saved to
`models/real_data/checkpoint_{name}.pt`, never `models/checkpoint_{name}.pt`; (b)
`configs/real_data_config.yaml` sets `model.name: "real_data_experiment_2024"`, a sentinel that can never
equal `"lstm"`/`"transformer"`/`"hybrid"` — the only condition under which `train_model` copies a checkpoint
to the live-served `models/checkpoint.pt` (`if model_name == active_model_name`) can never be true for this
experiment, even if `output_dir` were accidentally omitted.

## 12. Random shuffling — SAFE
`train_loader = DataLoader(..., shuffle=True)` shuffles the **order samples are drawn into SGD batches**,
not the contents of any sample's input/target window (those are fixed by `create_sequences` before the
`DataLoader` ever sees them). `val_loader` and the final test-set evaluation do not shuffle at all. Shuffled
batch order affects gradient noise, not information availability — no leakage vector here.

---

## Split Methodology

Chronological `TRAIN → VALIDATION → TEST`, computed as a single index slice on the 2024 surface array
(sorted ascending by date), using `configs/real_data_config.yaml`'s `train_ratio=0.6`, `val_ratio=0.2`,
`test_ratio=0.2` — identical convention to the existing synthetic pipeline, applied to real dates instead
of a synthetic index count.

*Filled in from the actual run's `data/historical/spy_2024_split_report.json` (252 valid surfaces out of
253 trading dates; one date — 2024's lowest-liquidity session, only 2 raw contracts — was correctly
skipped by the existing `<5 valid contracts` rule rather than forced through):*

| Split | Date range | Surfaces | Sequences built (lookback=20, horizon=10) | Look-ahead violations found |
|---|---|---|---|---|
| Train | 2024-01-02 to 2024-08-07 | 151 | 122 | 0 / 122 checked |
| Validation | 2024-08-08 to 2024-10-17 | 50 | 21 | 0 / 21 checked |
| Test | 2024-10-18 to 2024-12-31 | 51 | 22 | 0 / 22 checked |

Per-horizon usable sample counts (informational — `create_sequences` itself always builds at
`horizon=max(horizons)=10`; shorter horizons are extracted by indexing downstream, not built as separate
sequence sets, per existing project convention):

| Split | 1-day usable | 5-day usable | 10-day usable |
|---|---|---|---|
| Train | 131 | 127 | 122 |
| Validation | 30 | 26 | 21 |
| Test | 31 | 27 | 22 |

## Verification run

`scripts/run_real_data_2024_integration_test.py::verify_no_lookahead` result: **0 violations found across
all 165 sequences checked (122 train + 21 validation + 22 test)** — every input date is strictly before
every target date, and every horizon (1, 5, 10 days ahead) lands at the correct offset, confirmed against
actual calendar dates, not just array indices. This is a hard assertion gate in the integration test script
(`assert violations == 0`), not a soft warning — the script would have failed loudly had this not held.
