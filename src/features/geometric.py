import numpy as np 
def arc_length(trajectory):
    total_distance = 0
    for i in range(len(trajectory)-1):
        x_distance = trajectory[i+1,0] - trajectory[i,0]
        y_distance = trajectory[i+1,1] - trajectory[i,1]
        distance = np.sqrt(x_distance**2 + y_distance**2)
        total_distance += distance
    return total_distance 

def turning_angles(trajectory):
    turning_angles = []
    for i in range(len(trajectory)-2):
        v_in = trajectory[i+1] - trajectory[i] 
        v_out = trajectory[i+2] - trajectory[i+1]
        if np.linalg.norm(v_in) == 0 or np.linalg.norm(v_out) == 0:
            turning_angles.append(0)
            continue
        cos_angle = np.dot(v_in, v_out) / (np.linalg.norm(v_in) * np.linalg.norm(v_out))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        turning_angles.append(angle)
    return np.array(turning_angles)

def bounding_box(trajectory):
    x_coords = trajectory[:,0]
    y_coords = trajectory[:,1]
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)
    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = width / height if height != 0 else 0
    area = width * height 
    return width, height, aspect_ratio, area 

def extract_geometric_features(trajectory):
    geometric_features = []
    geometric_features.append(arc_length(trajectory))
    w, h, ar, area = bounding_box(trajectory)
    geometric_features.append(w)
    geometric_features.append(h)
    geometric_features.append(ar)
    geometric_features.append(area)
    for turning_angle in turning_angles(trajectory):
        geometric_features.append(turning_angle)
    return np.array(geometric_features,dtype=float)  