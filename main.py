# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import seaborn as sns
from IPython import display
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance_matrix
from skimage import io
from tqdm import tqdm as progress_bar

M = 256
N = 3
def propeller(theta, m):

    return np.sin(N*theta + m * np.pi / 10)
thetas = np.linspace(-np.pi, np.pi, 1000)
r = propeller(thetas, m=0)

_ = plt.figure(figsize=[8, 8])
_ = plt.polar(thetas, r)

thetas = np.linspace(-np.pi, np.pi, 1000)
rs = propeller(thetas, m=0)


def polar_to_cartesian(theta, r):

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def capture(func: np.array, resolution: int, threshold: float = 0.1, low: float = -1, high: float = 1) -> np.array:

    grid_x, grid_y = np.meshgrid(np.linspace(low, high, resolution), np.linspace(low, high, resolution))
    grid = np.column_stack([grid_x.flatten(), grid_y.flatten()])

    distances = distance_matrix(grid, func)
    capture = (np.min(distances, axis=1) <= threshold).astype(int).reshape(resolution, resolution)

    return capture

_ = plt.imshow(capture(polar_to_cartesian(thetas, rs), resolution=256, threshold=0.05, low=-np.pi / 2, high=np.pi / 2), cmap="Grays")

thetas = np.linspace(-np.pi, np.pi, 1000)
ms = np.arange(-M // 2, M // 2)

funcs = []

for m in ms.tolist():
    r = propeller(thetas, m=m)
    func = polar_to_cartesian(thetas, r)
    funcs.append(func)

funcs = np.asarray(funcs)

def record(funcs: list, capture_kwargs) -> np.array:

    return np.asarray([capture(func, **capture_kwargs) for func in progress_bar(funcs)])

recording = record(funcs, capture_kwargs=dict(resolution=256, threshold=0.05, low=-np.pi / 2, high=np.pi / 2))
recording.shape

offset = 0
length = 4
capture = np.zeros([256, 256])

for frame in recording:
    capture[offset: offset + length, :] = frame[offset: offset + length, :]
    offset += length
plt.clf()
plt.imshow(capture, cmap="Grays")
plt.savefig('images.png')