# Another implementation to take a look at:\
# http://charlesleifer.com/blog/using-python-and-k-means-to-find-the-dominant-colors-in-images/

# Could eventually make an online tool

import os
from os.path import isfile, join
import sys
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

_PRIOR  = 1
cwd     = ''
mypath  = ''
data    = []

def debug(prior, str):
    if(prior <= _PRIOR):
        print(str)

def magnitude(arr):
    squared_sum     = math.sqrt(arr[0]**2 + arr[1]**2 + arr[2]**2)
    return squared_sum

def dom_color():
    cwd             = os.getcwd()
    onlyfiles       = [f for f in os.listdir(cwd) if isfile(join(cwd, f))]
    onlyfiles       = [x for x in onlyfiles if '.jpg' in x]
    onlyfiles.sort()
    # TODO name fields in numpy array?
    data            = np.asarray(onlyfiles).reshape(len(onlyfiles), 1)

    # Run k-means on all images to find dominant colors
    dominant_colors = []
    for jpg in onlyfiles:
        debug(2, "Assessing {0}".format(jpg))
        image       = cv2.imread(jpg)
        image       = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        image       = cv2.resize(image, (50,50))

        image       = image.reshape((image.shape[0] * image.shape[1], 3))
        clt         = KMeans(n_clusters = 1)
        clt.fit(image)

        color       = clt.cluster_centers_[0]
        dominant_colors.append(color)


    dominant_colors = np.array(dominant_colors).reshape(len(dominant_colors),3)
    data = np.hstack((data,dominant_colors))
    # TODO Output a debugging swatch of the dominant color
    # for img in range(0, data.shape[0]):
    #     out_nm      = str(data[img,:0]) + '_dom_color.png'
    #     out         = cv2.cvtColor(np.asarray(data[img, 1:4]), cv2.COLOR_Lab2BGR)
    #     debug(1, out_nm)
    #     cv2.imwrite(out_nm, out)

    magnitudes      = np.apply_along_axis(magnitude, 1, dominant_colors)
    magnitudes      = magnitudes.reshape(magnitudes.shape[0],1)

    data            = np.array(onlyfiles).reshape(len(onlyfiles),1)
    data            = np.hstack((data, dominant_colors))
    data            = np.hstack((data, magnitudes))
    data            = data[data[:,-1].argsort()]

    return data

def collage(data, x, y):
    assert (x*y <= data.shape[0]) # TODO < ?
    final           = []
    for i in range(0,y):
        for j in range(0,x):
            img_nm  = data[x*i+j,0]
            debug(2, 'Adding to collage ' + img_nm)
            img     = cv2.imread(img_nm)
            img     = cv2.resize(img, (700, 700))
            img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if j > 0:
                row = np.hstack((row, img))
            else:
                row = img

        if i > 0:
            final   = np.vstack((final, row))
        else:
            final   = row

    return cv2.cvtColor(final, cv2.COLOR_RGB2BGR)


# Note: Order based on alphabetical ordering
def ref_collage(data, x, y):
    assert (x*y <= data.shape[0]) # TODO < ?
    final           = []
    order           = [ 'pinkfloyd_thedarksideofthemoon',
                        'foofighters_wastinglight',
                        'blink182_takeoffyourpantsandj',
                        'theblackkeys_brothers',
                        'arcticmonkeys_am',
                        'coldwarkids_robberscoward',
                        'thewho_tommy',
                        'jcole_2014foresthillsdrive',
                        'thebeatles_abbeyroad',
                        'altj_anawesomewave',
                        'nirvana_nevermind20thanniver',
                        'thebeatles_sgtpepperslonelyhear',
                        'milkychance_sadnecessary',
                        'thestrokes_isthisit',
                        'theblackkeys_turnblue',
                        'tameimpala_currents',
                        'chancetherapper_acidrap',
                        'kanyewest_mybeautifuldarktwist',
                        'sum41_allkillernofiller',
                        'yellowcard_oceanavenue',
                        'riseagainst_thesuffererthewitnes',
                        'foofighters_inyourhonor',
                        'kendricklamar_goodkidmaadcity',
                        'thenational_troublewillfindme',
                        ]

    num              = 0
    row              = []
    for img_nm in order:
        jpg          = img_nm + '.jpg'
        debug(2, 'Adding to reference collage ' + jpg)
        img          = cv2.imread(jpg)
        img          = cv2.resize(img, (900, 900))
        img          = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if num % x == 0:
            row      = img
        else:
            row = np.hstack((row, img))

        if (num+1) % x == 0:
            if (num+1 == x):
                final = row
            elif num > 0:
                final = np.vstack((final, row))

        num           += 1


    return cv2.cvtColor(final, cv2.COLOR_RGB2BGR)

def main():
    if (len(sys.argv) != 2):
        print("Usage: python3 art.py <dir>")
        return

    # Get all images in a folder
    cwd             = os.getcwd()
    mypath          = sys.argv[1]
    os.chdir(mypath)

    data            = dom_color()
    coll            = collage(data, 6, 4)
    cv2.imwrite("collage.png", coll)

    ref_coll        = ref_collage(data, 6, 4)
    cv2.imwrite("ref_collage.png", ref_coll)


if __name__ == '__main__':
    main()
