if __name__ == "__main__":
    from pathlib import Path
    ROOT = Path(__file__).parent.parent 
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.load import load_pendigits, row_to_trajectory 
    from src.features.geometric import extract_geometric_features
    from src.features.normalize import normalize_trajectory
    import numpy as np
    import matplotlib.pyplot as plt
    features, labels = load_pendigits(ROOT /"data" / "raw")
    row_number = np.random.randint(0, len(features))
    row = features.iloc[row_number]
    trajectory = row_to_trajectory(row)
    geometric_features = extract_geometric_features(trajectory)
    print(f"Geometric features for row {row_number}: {geometric_features}")
    print(f"Shape of geometric features: {len(geometric_features)}")
    print(geometric_features)
    normalized_trajectory = normalize_trajectory(trajectory)
    normalized_geometric_features = extract_geometric_features(normalized_trajectory)
    print(f"Shape of normalized geometric features: {len(normalized_geometric_features)}")
    print(normalized_geometric_features)
