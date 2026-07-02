import numpy as np
def normalize_trajectory(trajectory):
    centroid_x = np.mean(trajectory[:,0])
    centroid_y = np.mean(trajectory[:,1])
    centered = trajectory - np.array([centroid_x,centroid_y])
    max_distance = np.max(np.abs(centered))
    normalized = centered / max_distance
    
    return normalized