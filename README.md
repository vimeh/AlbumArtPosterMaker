# What is?

Started as Python OpenCV experiment to make a collage of album art. Began to explore methods to automatically determine order. Implemented simple linear classifiers with promising results, training order based upon input from k-means segmentation of the images in LAB color space.

## Getting Started

### Prequisites

* python3
* opencv2
* numpy + matplotlib
* sklearn

### Installing
Clone into a directory

```
git clone https://github.com/vinard/AlbumArtPosterMaker.git
```

Enter directory and make directory for source images

```
cd AlbumArtPosterMaker
mkdir src_images"
```

Gather album art from somewhere (eg., [AlbumArtExchange](https://www.albumartexchange.com)), and save images into directory for source images. Works best with square images of at least 300x300px.

From top level directory, run:


```
python3 color_ordering.py ./src_images
```

Sit back and wait! Collage should be written in top level directory after images' dominant color is assessed

## Future Work (TODO)

* Gather human ordered data (Mechanical Turk, 3Blue1Brown sponsor, etc.)
* More robust classifiers
* Read suggested materials from Prof Nayar OH
* Continue updating README to follow [good practices template](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)

## Notes

### 11/01 Office Hours with Prof Nayar

* Color Indexing by Swain and Ballard
  * Will get far
  * Easy way to do search
  * Constrain dimensions (255->10) and normalize
    * Histogram is reduced
    * Easier to train on
* Histogram intersection
* Not just sorting, also matching
* 3 images, center is closer to which?
  * Mechanical Turk (Amazon)
  * Or build on website and send to people
    1. Scrape all album covers from website
    2. Set up interface for comparison

## Acknowledgements

* Another implementation to take a look at:
 [Charles Leifer](http://charlesleifer.com/blog/using-python-and-k-means-to-find-the-dominant-colors-in-images/)
* plot_colors and centroid_histogram modified from: [Adrian Rosebrock](https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/)
