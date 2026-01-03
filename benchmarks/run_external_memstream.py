from __future__ import annotations

import copy
import json
import logging
import multiprocessing

import pandas as pd
from config_external_memstream import N_CHECKPOINTS, QUANTILE_Q_VALUES, TRACKS
from tqdm import tqdm

from river import metrics

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def run_dataset_with_q(no_dataset, no_track, q_value):
    track = TRACKS[no_track]
    dataset = track.datasets[no_dataset]
    dataset_name = dataset.__class__.__name__
    
    # Get dataset-specific model with q value
    model = track.get_model_for_dataset(dataset, q=q_value)
    
    # Get base MemStream params for model string
    base_model = model.anomaly_detector
    model_str = f"QF(q={q_value}) | MemStreamAutoencoder(beta={base_model.max_threshold},memlen={base_model.memory_size})"
    
    print(f"Processing {model_str} on {dataset_name}")

    results = []
    track = copy.deepcopy(track)
    time = 0.0
    for i in tqdm(
        track.run(model, dataset, n_checkpoints=N_CHECKPOINTS),
        total=N_CHECKPOINTS,
        desc=f"{dataset_name} q={q_value}",
    ):
        time += i["Time"].total_seconds()
        res = {
            "step": i["Step"],
            "track": track.name,
            "model": model_str,
            "dataset": dataset_name,
            "q": q_value,
        }
        for k, v in i.items():
            if isinstance(v, metrics.base.Metric):
                res[k] = v.get()
        res["Memory in Mb"] = i["Memory"] / 1024**2
        res["Time in s"] = time
        results.append(res)
    return results


def run_track(no_track: int, n_workers: int = 50):
    pool = multiprocessing.Pool(processes=n_workers)
    track = TRACKS[no_track]
    
    # Create runs for all combinations of datasets and q values
    runs = []
    for dataset_idx in range(len(track.datasets)):
        for q_value in QUANTILE_Q_VALUES:
            runs.append((dataset_idx, no_track, q_value))
    
    results = []
    for val in pool.starmap(run_dataset_with_q, runs):
        results.extend(val)
    
    csv_name = f"{track.name.replace(' ', '_').lower()}_memstream"
    pd.DataFrame(results).to_csv(f"./{csv_name}.csv", index=False)


if __name__ == "__main__":
    details = {}
    # Create details for each track
    for i, track in enumerate(TRACKS):
        details[track.name] = {"Dataset": {}, "Model": {}}
        for dataset in track.datasets:
            dataset_name = dataset.__class__.__name__
            details[track.name]["Dataset"][dataset_name] = repr(dataset)
            
            # Add models for each q value
            for q_value in QUANTILE_Q_VALUES:
                model = track.get_model_for_dataset(dataset, q=q_value)
                base_model = model.anomaly_detector
                model_str = f"{dataset_name}_QF(q={q_value})_MemStream(beta={base_model.max_threshold},memlen={base_model.memory_size})"
                details[track.name]["Model"][model_str] = repr(model)
        
        with open("details_external_memstream.json", "w") as f:
            json.dump(details, f, indent=2)
        
        run_track(no_track=i, n_workers=50)
