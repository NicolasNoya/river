from __future__ import annotations

from river import anomaly, evaluate, metrics
from external_datasets import (
    Cardio,
    Cover,
    # DOS,
    Ionosphere,
    # KDD,
    Mammography,
    # NSL,
    Pima,
    Satimage2,
    Statlog,
    SYN,
    # UNSW,
)

N_CHECKPOINTS = 50

# QuantileFilter q values to test
QUANTILE_Q_VALUES = [0.90, 0.95, 0.99]


class ExternalAnomalyDetectionTrack(evaluate.tracks.Track):
    """Custom track for external anomaly detection datasets with dataset-specific models."""

    def __init__(self):
        # Define all datasets (only fast ones for testing the fix)
        self.all_datasets = [
            # KDD(),
            # NSL(),
            # UNSW(),
            # DOS(),
            # SYN(),
            # Ionosphere(),
            Cardio(),
            # Statlog(),
            Satimage2(),
            Mammography(),
            # Pima(),
            # Cover(),  # Also large, commenting out
        ]

        # Dataset-specific MemStream configurations
        # Format: beta, memlen
        self.dataset_configs = {
            # "KDD": {"beta": 1, "memory_size": 256},
            # "NSL": {"beta": 0.1, "memory_size": 2048},
            # "UNSW": {"beta": 0.1, "memory_size": 2048},
            # "DOS": {"beta": 0.1, "memory_size": 2048},
            # "SYN": {"beta": 1, "memory_size": 16},
            # "Ionosphere": {"beta": 0.001, "memory_size": 4},
            "Cardio": {"beta": 1, "memory_size": 64},
            # "Statlog": {"beta": 0.01, "memory_size": 32},
            "Satimage2": {"beta": 10, "memory_size": 256},
            "Mammography": {"beta": 0.1, "memory_size": 128},
            # "Pima": {"beta": 0.001, "memory_size": 64},
            # "Cover": {"beta": 0.0001, "memory_size": 2048},
        }

        super().__init__(
            name="External Anomaly Detection",
            datasets=self.all_datasets,
            metric=metrics.ROCAUC(),
        )

    def get_model_for_dataset(self, dataset, q=0.95):
        """Get the properly configured model for a specific dataset.
        
        Parameters
        ----------
        dataset
            The dataset to get the model for.
        q
            The quantile threshold for the QuantileFilter wrapper.
        """
        dataset_name = dataset.__class__.__name__
        
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"No configuration found for dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        
        # Create MemStreamAutoencoder with dataset-specific params
        # beta -> max_threshold, memlen -> memory_size
        # grace_period = memory_size (as per best practices)
        # Paper reproduction settings: batch_size=1, epochs=5000, learning_rate=0.01
        base_model = anomaly.MemStreamAutoencoder(
            memory_size=config["memory_size"],
            max_threshold=config["beta"],
            grace_period=config["memory_size"],
            epochs=5000,
            learning_rate=0.01,
            batch_size=1,
        )
        
        # Wrap with QuantileFilter for classification
        model = anomaly.QuantileFilter(base_model, q=q)
        
        return model


TRACKS = [
    ExternalAnomalyDetectionTrack(),
]

# Models will be created per-dataset in the track
MODELS = {
    "External Anomaly Detection": {}
}
