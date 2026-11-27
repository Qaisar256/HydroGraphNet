# HydroGraphNet

Adaptive Hydrology-Aware GRU with Hierarchical Upstream Aggregation for River Turbidity Forecasting and Early Warning.

This is a reference implementation reconstructed from HydroGraphV7, not the original experimental code. It does not claim to reproduce the manuscript's numerical results until real data and topology are audited and experiments are run. No published metrics are hard-coded.

## Installation

Python 3.10+; execute commands from repository root. CPU or CUDA supported.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests
```

## Dataset 1: Ballycanew, Ireland

Negri et al. (2024), https://doi.org/10.1016/j.envsoft.2024.106073, https://github.com/CamillaNegri/Ballycanew_Ptool. Downloads three original CSV files, preserving the source discharge scale (`Qcumecs`) because its physical interpretation has not been independently established. Turbidity is NTU. Commit and hashes are recorded in the raw manifest.

```bash
python -m data.download_ballycanew
python -m data.preprocess --config config/ballycanew_config.yaml
python train.py --config config/ballycanew_config.yaml --mode full
python evaluate.py --checkpoint outputs/ballycanew/full/best.pt
python predict.py --checkpoint outputs/ballycanew/full/best.pt
```

Single-station full mode intentionally reduces to GRU-only; it cannot demonstrate a spatial graph improvement. Target counts are calculated, not forced to 1,843/365/366. The first calendar day is excluded as a target; early histories are padded with missingness masks. Splits are by target date (through 2014, 2015, 2016).

## Dataset 2: Little Blue River, Missouri

Manuscript reference: https://doi.org/10.5066/F77W6B4Z. The release covers five monitored sites and includes estimated values. The script instead requests daily mean discharge (00060, converted from ft3/s to m3/s) and turbidity (63680, FNU) from https://api.waterdata.usgs.gov/. This is a distinct acquisition route, not a byte-identical reproduction of the DOI release. The API requests all seven manuscript stations. It retains raw responses and coverage reports; it does not fabricate missing series. FNU is not interchangeable with NTU. Do not label resulting errors NTU or compare them directly with the published NTU numbers without a verified calibration or a compatible source series. An optional USGS_API_KEY environment variable is supported. API access has not been live-tested in this package.

```bash
python -m data.download_little_blue_river
python -m data.preprocess --config config/little_blue_river_config.yaml
python train.py --config config/little_blue_river_config.yaml --mode gru
```

Before graph training, audit the station overlap and independently verify the proposed directed reaches using authoritative hydrography (e.g. NHDPlus). The paper's edges are included as an unverified candidate configuration, not verified geographic facts. Record the evidence for each edge, and only then set `graph_verified: true`. If coverage fails, do not bypass the check just to obtain a result. Revise the station list and edges and document the deviation; the default minimum training overlap of 30 days is an implementation choice, not a paper-derived eligibility threshold. No distance, elevation or physical travel time is invented. Zero-lag representations are used. The Little Blue River chronological cutoff is an explicit configurable choice.

```bash
python train.py --config config/little_blue_river_config.yaml --mode full
python evaluate.py --checkpoint outputs/little_blue_river/full/best.pt
```

## Model and ablations

Shared GRU, two graph attention layers, discharge and availability features, sigmoid fusion, Softplus turbidity head, and binary alert head. GRU-only and edge-removal modes are equivalent by design. `shuffle` changes edge endpoints while preserving edge count; `no_discharge` removes edge discharge only; `no_fusion` replaces the learned gate with an equal mixture. All modes use common target dates and normalization. AdamW decoupled weight decay only; no extra explicit L2 term. Alert loss weight 0.1, gradient clipping 5, fixed fusion 0.5, and seed 20261109 are reconstruction choices. No spatial holdout or automated hyperparameter sweep is implemented in this version. The two-head attention implementation averages head weights. Missing inputs are neutral-filled after training-only scaling, never using future data. Alert labels are strictly above the training 75th percentile; classification metrics use probability >=0.5, and regression-threshold alerts are separately exported.

```bash
python train.py --config config/little_blue_river_config.yaml --mode no_discharge
python evaluate.py --checkpoint outputs/little_blue_river/no_discharge/best.pt
```

Other modes: `gru`, `no_edges`, `shuffle`, `no_fusion`. Validation MAE determines best checkpoint; test targets are excluded from optimization. Test predictions, per-station and pooled metrics, forecast plot and training history are stored under `outputs/`. Run separate outputs for different configurations. Do not overwrite processed data between training and evaluation. All metrics are computed on observed finite targets only. Reproducibility across hardware/software is not guaranteed.

## License and data rights

MIT is proposed for the new code, subject to author approval before publishing. Third-party observations retain their respective source terms and attribution. Raw data and checkpoints are excluded from Git. Cite the original data providers and the HydroGraphNet manuscript when using this code.
