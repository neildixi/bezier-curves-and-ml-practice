def load_pendigits(data_dir):
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import os
    data_dir = Path(data_dir)
    pendigits_features = pd.read_csv(data_dir / "pendigits_features.csv")
    pendigits_labels = pd.read_csv(data_dir / "pendigits_labels.csv")
    return pendigits_features, pendigits_labels

def row_to_trajectory(row):
    coordinates =row.to_numpy()
    trajectory= coordinates.reshape(8,2)
    return trajectory
