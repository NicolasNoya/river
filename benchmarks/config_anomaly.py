from __future__ import annotations

import itertools
from river import (
    anomaly,
    evaluate,
    feature_extraction,
    preprocessing,
)

N_CHECKPOINTS = 50

LEARNING_RATE = 0.005

TRACKS = [
    evaluate.AnomalyDetectionTrack(),
]

# Define hyperparameter grids for each model
HYPERPARAMETER_GRIDS = {
    "HalfSpaceTrees": {
        "n_trees": [10, 25, 50],
        "height": [8, 10, 12],
        "window_size": [250, 500],
        "seed": [42],
    },
    "LocalOutlierFactor": {
        "n_neighbors": [10, 20, 30, 50],
    },
    "OneClassSVM": {
        "nu": [0.01, 0.05, 0.1, 0.2],
    },
}

# Preprocessing options
PREPROCESSING_OPTIONS = {
    "HalfSpaceTrees": [
        ("MinMaxScaler", preprocessing.MinMaxScaler()),
    ],
    "LocalOutlierFactor": [
        ("None", None),
        ("StandardScaler", preprocessing.StandardScaler()),
        ("MinMaxScaler", preprocessing.MinMaxScaler()),
    ],
    "OneClassSVM": [
        ("None", None),
        ("StandardScaler", preprocessing.StandardScaler()),
        ("MinMaxScaler", preprocessing.MinMaxScaler()),
        ("StandardScaler+RBFSampler", preprocessing.StandardScaler() | feature_extraction.RBFSampler()),
    ],
}

# QuantileFilter options (q values to test, None means no filter)
QUANTILE_FILTER_Q = [None, 0.95, 0.99, 0.995]


def generate_model_configs():
    """Generate all model configurations from hyperparameter grids."""
    models = {}

    for model_name, param_grid in HYPERPARAMETER_GRIDS.items():
        # Get all combinations of hyperparameters
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for param_combo in itertools.product(*param_values):
            params = dict(zip(param_names, param_combo))

            # Get preprocessing options for this model
            preprocessing_opts = PREPROCESSING_OPTIONS[model_name]

            for prep_name, prep_pipeline in preprocessing_opts:
                for q_value in QUANTILE_FILTER_Q:
                    # Create base model
                    if model_name == "HalfSpaceTrees":
                        base_model = anomaly.HalfSpaceTrees(**params)
                    elif model_name == "LocalOutlierFactor":
                        base_model = anomaly.LocalOutlierFactor(**params)
                    elif model_name == "OneClassSVM":
                        base_model = anomaly.OneClassSVM(**params)

                    # Build pipeline with preprocessing
                    if prep_pipeline is not None:
                        pipeline = prep_pipeline | base_model
                    else:
                        pipeline = base_model

                    # Wrap with QuantileFilter if specified
                    if q_value is not None:
                        pipeline = anomaly.QuantileFilter(pipeline, q=q_value)

                    # Generate descriptive name
                    config_name_parts = []

                    # Add QuantileFilter info
                    if q_value is not None:
                        config_name_parts.append(f"QF(q={q_value})")

                    # Add preprocessing info
                    if prep_name != "None":
                        config_name_parts.append(prep_name)

                    # Add model name and params
                    param_str = ",".join([f"{k}={v}" for k, v in params.items() if k != "seed"])
                    config_name_parts.append(f"{model_name}({param_str})")

                    config_name = " | ".join(config_name_parts)

                    models[config_name] = pipeline

    return models


# Generate all model configurations
MODELS = {
    "Anomaly detection": generate_model_configs()
}

# Print summary of configurations
if __name__ == "__main__":
    print(f"Total configurations generated: {len(MODELS['Anomaly detection'])}")
    print("\nBreakdown by base model:")
    for base_model in HYPERPARAMETER_GRIDS.keys():
        count = sum(1 for name in MODELS['Anomaly detection'].keys() if base_model in name)
        print(f"  {base_model}: {count} configurations")

    print("\nFirst 10 configurations:")
    for i, name in enumerate(list(MODELS['Anomaly detection'].keys())[:10]):
        print(f"  {i+1}. {name}")
