# import autograd.numpy as np
# from autograd import grad
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

class SplinePath:
    def __init__(self, x_coefs, y_coefs, breaks):
        self.x_coefs = x_coefs
        self.y_coefs = y_coefs
        self.breaks = breaks


def load_splines():
    x_coefs = np.loadtxt("../splines/nuscenes_scene_23_x_spline_coefs.csv", delimiter=',')
    y_coefs = np.loadtxt("../splines/nuscenes_scene_23_y_spline_coefs.csv", delimiter=',')
    breaks = np.loadtxt("../splines/nuscenes_scene_23_breaks.csv", delimiter=',')

    return x_coefs, y_coefs, breaks


def find_spline_interval(s, path):
    breaks = path.breaks

    if s < breaks[0] or s > breaks[-1]:
        if s < 0 and abs(s) < 1E-2:
            print(f"The value s = {s} is slightly below 0.")
            return 0
        print(f"The value s = {s} does not lie within the valid spline domain.")
        print("Path length has been overrun.")
        return None

    upper_idx_lim = len(breaks) - 1
    lower_idx_lim = 0
    interval_idx = (upper_idx_lim + lower_idx_lim) // 2

    while True:
        if breaks[interval_idx] <= s <= breaks[interval_idx + 1]:
            return interval_idx
        elif s > breaks[interval_idx + 1]:
            lower_idx_lim = max(0, interval_idx + 1)
            interval_idx = (upper_idx_lim + lower_idx_lim) // 2
        elif s < breaks[interval_idx]:
            upper_idx_lim = min(len(breaks) - 1, interval_idx)
            interval_idx = (upper_idx_lim + lower_idx_lim) // 2
        else:
            print("Unexpected case with s = ", s)
            raise Exception("Unexpected value of s")


def spline_x(s, path, spline_idx):
    x_coefs = path.x_coefs
    breaks = path.breaks

    delta_s = s - breaks[spline_idx]
    x_value = (x_coefs[spline_idx][0] * delta_s ** 3 +
               x_coefs[spline_idx][1] * delta_s ** 2 +
               x_coefs[spline_idx][2] * delta_s +
               x_coefs[spline_idx][3])
    return x_value



def spline_y(s, path, spline_idx):
    y_coefs = path.y_coefs
    breaks = path.breaks

    delta_s = s - breaks[spline_idx]
    y_value = (y_coefs[spline_idx][0] * delta_s ** 3 +
               y_coefs[spline_idx][1] * delta_s ** 2 +
               y_coefs[spline_idx][2] * delta_s +
               y_coefs[spline_idx][3])
    return y_value


def find_best_s(q, path, ds=0.05, enable_global_search=False, sq_dist_tol=100):
    x, y, _, _, s = qenumerate

    if s < 0:
        return 0

    spline_idx = find_spline_interval(s, path)
    x_proj = spline_x(s, path, spline_idx)
    y_proj = spline_y(s, path, spline_idx)
    current_dist = (x - x_proj) ** 2 + (y - y_proj) ** 2

    ss = np.concatenate([np.arange(max(0, s - 5), s, ds),
                         np.arange(s + ds, s + 5, ds)])
    spline_indices = [find_spline_interval(val, path) for val in ss]
    xs = [spline_x(val, path, idx) for val, idx in zip(ss, spline_indices)]
    ys = [spline_y(val, path, idx) for val, idx in zip(ss, spline_indices)]
    sq_distances = (np.array(xs) - x) ** 2 + (np.array(ys) - y) ** 2
    best_dist = np.min(sq_distances)
    best_idx = np.argmin(sq_distances)

    if best_dist > sq_dist_tol and enable_global_search:
        # print("Local search too narrow. Performing global search.")
        ss = np.arange(path.breaks[0], path.breaks[-1], ds)
        spline_indices = [find_spline_interval(val, path) for val in ss]
        xs = [spline_x(val, path, idx) for val, idx in zip(ss, spline_indices)]
        ys = [spline_y(val, path, idx) for val, idx in zip(ss, spline_indices)]
        sq_distances = (np.array(xs) - x) ** 2 + (np.array(ys) - y) ** 2
        best_dist = np.min(sq_distances)
        best_idx = np.argmin(sq_distances)

    if best_dist >= current_dist:
        return s
    else:
        return ss[best_idx]


def dpath_ds(s, path, spline_idx):
    dx_ds = grad(lambda s: spline_x(s, path, spline_idx))
    dy_ds = grad(lambda s: spline_y(s, path, spline_idx))
    return dx_ds(s), dy_ds(s)


def heading(s, path, spline_idx):
    dx_ds, dy_ds = dpath_ds(s, path, spline_idx)
    # print('dx_ds:', dx_ds)dpath_ds
    # print('dy_ds:', dy_ds)
    return np.arctan2(dy_ds, dx_ds)


def get_path_obj(x_list, y_list):

    # trajectory points
    # x_points = np.array(remove_repeated_items_at_end(x_list))
    # y_points = np.array(remove_repeated_items_at_end(y_list))
    x_points = np.array(x_list)
    y_points = np.array(y_list)
    # Calculate distances between consecutive points
    distances = np.sqrt(np.diff(x_points) ** 2 + np.diff(y_points) ** 2)
    cumulative_arc_length = np.insert(np.cumsum(distances), 0, 0)

    # Use cumulative arc length for parameterization
    param = cumulative_arc_length

    segment_length = 4  # Or another number > 3
    segment_indices = np.arange(0, len(x_points), segment_length)
    # if segment_indices[-1] != len(x_points) - 1:
    #     segment_indices = np.append(segment_indices, len(x_points) - 1)

    num_segments = len(segment_indices) - 1
    x_coefs = np.zeros((num_segments, 4))
    y_coefs = np.zeros((num_segments, 4))

    # Compute coefficients for each segment
    for i in range(num_segments):
        start_idx, end_idx = segment_indices[i], segment_indices[i + 1]
        x_spline = scipy.interpolate.CubicSpline(param[start_idx:end_idx], x_points[start_idx:end_idx])
        y_spline = scipy.interpolate.CubicSpline(param[start_idx:end_idx], y_points[start_idx:end_idx])
        x_coefs[i, :] = x_spline.c[:, 0]
        y_coefs[i, :] = y_spline.c[:, 0]

    # Breaks are the parameter values at the start of each segment
    breaks = param[segment_indices[:-1]]
    return SplinePath(x_coefs, y_coefs, breaks)

def remove_repeated_items_at_end(arr):
    # Convert to NumPy array
    if arr.is_cuda:
        arr = arr.cpu().numpy()
    else:
        arr = arr.numpy()
    last_item = arr[-1]  # Get the last item of the array
    # Find the first index from the end where the value is different from the last item
    # Reverse the array for this purpose and use np.where to find the indices
    reversed_indices = np.where(arr[::-1] != last_item)[0]
    
    if reversed_indices.size == 0:  # If all values are the same
        return arr[:1]  # Return an array with only one element (the last item)
    
    first_diff_index_from_end = reversed_indices[0]
    # Calculate the index to slice from by subtracting from the length of the array
    index_to_keep = arr.size - first_diff_index_from_end - 1

    return arr[:index_to_keep + 1]