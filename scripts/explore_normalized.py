if __name__ == "__main__":
    
    from pathlib import Path
    ROOT = Path(__file__).parent.parent
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data.load import load_pendigits, row_to_trajectory
    from src.features.normalize import normalize_trajectory
    import matplotlib.pyplot as plt 
    import numpy as np
    features, labels = load_pendigits(ROOT /"data" / "raw")

    fig, ax = plt.subplots(5,4)
    for i in range(10):
        random_index = np.random.randint(0,len(features))
        trajectory = row_to_trajectory(features.iloc[random_index])
        normalized_trajectory = normalize_trajectory(trajectory)
        ax[2*i//4, 2*i%4].set_title(f"Digit: {labels.iloc[random_index]["Class"]}")
        ax[2*i//4, 2*i%4].plot(trajectory[:,0], trajectory[:,1], "o-")
        ax[2*i//4, 2*i%4].set_xlabel("X Axis")
        ax[2*i//4, 2*i%4].set_ylabel("Y Axis")
        ax[2*i//4, 2*i%4].grid(True)
        ax[2*i//4, 2*i%4].set_aspect("equal")

        ax[(2*i+1)//4, (2*i+1)%4].set_title(f"Digit: {labels.iloc[random_index]["Class"]}")
        ax[(2*i+1)//4, (2*i+1)%4].plot(normalized_trajectory[:,0], normalized_trajectory[:,1], "o-")
        ax[(2*i+1)//4, (2*i+1)%4].set_xlabel("X Axis")
        ax[(2*i+1)//4, (2*i+1)%4].set_ylabel("Y Axis")
        ax[(2*i+1)//4, (2*i+1)%4].grid(True)
        ax[(2*i+1)//4, (2*i+1)%4].set_aspect("equal")

    plt.show()


