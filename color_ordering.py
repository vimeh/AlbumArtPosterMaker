import os
from os.path import isfile, join
import sys
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

_PRIOR = 1

def debug(prior, str):
    if(prior <= _PRIOR):
        print(str)

def magnitude(arr):
    squared_sum = math.sqrt(arr[0]**2 + arr[1]**2 + arr[2]**2)
    return squared_sum

def main():
    if (len(sys.argv) != 2):
        print("Usage: python3 art.py <dir>")
        return

    # Get all images in a folder
    mypath = sys.argv[1]
    os.chdir(mypath)
    cwd = os.getcwd()
    debug(3, cwd)
    onlyfiles = [f for f in os.listdir(cwd) if isfile(join(cwd, f))]
    onlyfiles = [x for x in onlyfiles if '.jpg' in x]
    debug(2, onlyfiles)

    # Run k-means on all images to find dominant colors
    dominant_colors = []
    for jpg in onlyfiles:
        debug(1, "Assessing {0}".format(jpg))
        image = cv2.imread(jpg)
        debug(2, image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = KMeans(n_clusters = 1)
        clt.fit(image)

        color = clt.cluster_centers_[0]
        dominant_colors.append(color)

    dominant_colors = np.array(dominant_colors)
    magnitudes      = np.apply_along_axis(magnitude, 1, dominant_colors)
    magnitudes      = magnitudes.reshape(magnitudes.shape[0],1)

    data            = np.hstack((dominant_colors, magnitudes))
    data            = np.hstack((data, np.array(onlyfiles).reshape(len(onlyfiles),1)))
    data            = data[data[:,-2].argsort()]
    debug(2, data)

    for i in range(len(onlyfiles)):
        os.rename(data[i,-1], "{:02d}.jpg".format(i))


if __name__ == '__main__':
    main()
