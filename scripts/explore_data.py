if __name__ == "__main__":
    
    from pathlib import Path
    ROOT = Path(__file__).parent.parent
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.load import load_pendigits, row_to_trajectory
    import matplotlib.pyplot as plt 
    import numpy as np
    features, labels = load_pendigits(ROOT /"data" / "raw")

    fig, ax = plt.subplots(2,5)
    for i in range(10):
        random_index = np.random.randint(0,len(features))
        trajectory = row_to_trajectory(features.iloc[random_index])
        ax[i//5, i%5].set_title(f"Digit: {labels.iloc[random_index]["Class"]}")
        ax[i//5, i%5].plot(trajectory[:,0], trajectory[:,1], "o-")
        ax[i//5, i%5].set_xlabel("X Axis")
        ax[i//5, i%5].set_ylabel("Y Axis")
        ax[i//5, i%5].grid(True)
        ax[i//5, i%5].set_aspect("equal")

    plt.show()


