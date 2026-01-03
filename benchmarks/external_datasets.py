from __future__ import annotations

import pathlib
from river import stream
from river.datasets import base

DATA_DIR = "/sc/home/nemesio.navarro/river_data"


class ExternalCSVDataset(base.FileDataset):
    """Base class for external CSV datasets with last column as label."""

    def __init__(self, name: str, n_samples: int, n_features: int, filename: str):
        super().__init__(
            n_samples=n_samples,
            n_features=n_features,
            task=base.BINARY_CLF,
            filename=filename,
            directory=f"{DATA_DIR}/{name}",
        )
        self.dataset_name = name

    def __iter__(self):
        # Read CSV with all columns converted to appropriate types
        # Features as float, label as int
        converters = {f"f{i}": float for i in range(self.n_features)}
        converters["y"] = int
        
        return stream.iter_csv(
            self.path,
            target="y",
            converters=converters,
        )


class Cardio(ExternalCSVDataset):
    """Cardiotocography dataset."""

    def __init__(self):
        super().__init__(
            name="cardio",
            n_samples=1831,
            n_features=21,
            filename="cardio.csv",
        )


class Cover(ExternalCSVDataset):
    """Covertype dataset."""

    def __init__(self):
        super().__init__(
            name="cover",
            n_samples=286048,
            n_features=10,
            filename="cover.csv",
        )


class DOS(ExternalCSVDataset):
    """CICIDS-DoS dataset."""

    def __init__(self):
        super().__init__(
            name="dos",
            n_samples=1048575,
            n_features=95,
            filename="dos.csv",
        )


class Ionosphere(ExternalCSVDataset):
    """Ionosphere dataset."""

    def __init__(self):
        super().__init__(
            name="ionosphere",
            n_samples=351,
            n_features=33,
            filename="ionosphere.csv",
        )


class KDD(ExternalCSVDataset):
    """KDDCUP99 dataset."""

    def __init__(self):
        super().__init__(
            name="kdd",
            n_samples=494021,
            n_features=121,
            filename="kdd.csv",
        )


class Mammography(ExternalCSVDataset):
    """Mammography dataset."""

    def __init__(self):
        super().__init__(
            name="mammography",
            n_samples=11183,
            n_features=6,
            filename="mammography.csv",
        )


class NSL(ExternalCSVDataset):
    """NSL-KDD dataset."""

    def __init__(self):
        super().__init__(
            name="nsl",
            n_samples=125973,
            n_features=126,
            filename="nsl.csv",
        )


class Pima(ExternalCSVDataset):
    """Pima Indians Diabetes dataset."""

    def __init__(self):
        super().__init__(
            name="pima",
            n_samples=768,
            n_features=8,
            filename="pima.csv",
        )


class Satimage2(ExternalCSVDataset):
    """Satimage-2 dataset."""

    def __init__(self):
        super().__init__(
            name="satimage-2",
            n_samples=5803,
            n_features=36,
            filename="satimage-2.csv",
        )


class Statlog(ExternalCSVDataset):
    """Statlog Landsat Satellite dataset."""

    def __init__(self):
        super().__init__(
            name="statlog",
            n_samples=6435,
            n_features=36,
            filename="statlog.csv",
        )


class SYN(ExternalCSVDataset):
    """SYN dataset."""

    def __init__(self):
        super().__init__(
            name="syn",
            n_samples=10000,
            n_features=1,
            filename="syn.csv",
        )


class UNSW(ExternalCSVDataset):
    """UNSW-NB 15 dataset."""

    def __init__(self):
        super().__init__(
            name="unsw",
            n_samples=2540044,
            n_features=122,
            filename="unsw.csv",
        )
