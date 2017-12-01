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

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX

	# return the bar chart
	return bar

def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()

	# return the histogram
	return hist


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
        debug(1, "Assessing {0}".format(jpg))
        image       = cv2.imread(jpg)
        # TODO test to see if different output if done in RGB space
        # would allow easier output of primary color for visual debugging
        image       = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image       = cv2.resize(image, (400,400))

        image       = image.reshape((image.shape[0] * image.shape[1], 3))
        clt         = KMeans(n_clusters = 1)
        clt.fit(image)

        # # build a histogram of clusters and then create a figure
        # # representing the number of pixels labeled to each color
        # hist = centroid_histogram(clt)
        # bar = plot_colors(hist, clt.cluster_centers_)
        #
        # # show our color bart
        # plt.figure()
        # plt.axis("off")
        # plt.imshow(bar)
        # plt.savefig("3_dom_col/3_dom_col_" + jpg, bbox_inches='tight', pad_inches=0)
        # plt.close()
        # # plt.show()

        color       = clt.cluster_centers_[0]
        dominant_colors.append(color)


    dominant_colors = np.array(dominant_colors).reshape(len(dominant_colors),3)
    data = np.hstack((data,dominant_colors))
    # TODO Output a debugging swatch of the dominant color
    # see charlesleifer code above for plotting histogram of dominant colors
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
            # img_nm  = '1_dom_col/1_dom_col_' + data[x*i+j,0]
            img_nm  = data[x*i+j,0]
            debug(1, 'Adding to collage ' + img_nm)
            img     = cv2.imread(img_nm)
            img     = cv2.resize(img, (200, 200))
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


def ref_collage(x, y):
    order           = [ 'pinkfloyd_thedarksideofthemoon',
                        'foofighters_wastinglight',
                        'blink182_takeoffyourpantsandj',
                        'theblackkeys_brothers',
                        'arcticmonkeys_am',
                        'coldwarkids_robberscoward',
                        'thewho_tommy',
                        'jcole_2014foresthillsdrive',
                        'thebeatles_abbeyroad',
                        'relientk_forgetandnotslowdown',
                        'altj_anawesomewave',
                        'nirvana_nevermind20thanniver',
                        'thebeatles_sgtpepperslonelyhear',
                        'kanyewest_mybeautifuldarktwist',
                        'thestrokes_isthisit',
                        'chancetherapper_acidrap',
                        'theblackkeys_turnblue',
                        'tameimpala_currents',
                        'sum41_allkillernofiller',
                        'yellowcard_oceanavenue',
                        'riseagainst_thesuffererthewitnes',
                        'thenational_troublewillfindme',
                        'kendricklamar_goodkidmaadcity',
                        'foofighters_inyourhonor'
                        ]
    assert (x*y <= len(order))

    final            = []
    num              = 0
    row              = []
    for img_nm in order:
        # jpg          = '1_dom_col/1_dom_col_' + img_nm + '.jpg'
        jpg          = img_nm + '.jpg'
        debug(1, 'Adding to reference collage ' + jpg)
        img          = cv2.imread(jpg)
        # img     = cv2.resize(img, (200, 200))
        img          = cv2.resize(img, (1000, 1000))
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

    # Set path of images folder
    cwd             = os.getcwd()
    mypath          = sys.argv[1]
    os.chdir(mypath)

    # # Assess dominant color
    # data            = dom_color()
    #
    # # Build and export collage
    # coll            = collage(data, 6, 4)
    # cv2.imwrite("../collage.png", coll)

    # Reference collage based on manual ordering of images
    ref_coll        = ref_collage(6, 4)
    cv2.imwrite("../ref_collage.png", ref_coll)


if __name__ == '__main__':
    main()
