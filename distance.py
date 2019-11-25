"""
This allows distance to be calculated from detected objects in video feed.

It is assumed that the first and only positional argument to the constructor
is of I/O operation (video stream)

#! Currently the class only allows images

In order to determine the distance from our camera 
to a known object or marker (in our case we are just using face detection), 
we are going to utilize triangle similarity.

The triangle similarity goes something like this: 
Let’s say we have a marker or object with a known width W. 
We then place this marker some distance D from our camera. 
We take a picture of our object using our camera and 
then measure the apparent width in pixels P. 
This allows us to derive the perceived focal length F of our camera:

F = (P x  D) / W

As I continue to move my camera both closer and farther away 
from the object/marker, I can apply the triangle similarity 
to determine the distance of the object to the camera:

D’ = (W x F) / P

Reference: https://stackoverflow.com/questions/14038002/opencv-how-to-calculate-distance-between-camera-and-object-using-image

Formula = distance to object (mm) = focal length (mm) * real height of the object (mm) * image height (pixels)
                          ----------------------------------------------------------------
                                object height (pixels) * sensor height (mm)

If imported into another file the module contains the following
functions:

    * find_marker() - returns computational bounding box of the of the object region 
    * distance_to_camera() - returns the distance from the maker to the camera
"""

__author__ = 'Gary Frederick'  # slightflex
__version__ = 0.1   # file version

# external modules
import numpy as np              # for array computing
from imutils import paths       # paths module
# functions such as translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges
import imutils
import cv2                      # computer vision library


class distanceDetection(object):
    """This allows distance to be calculated from detected objects in video feed."""
    # todo: change self.image to accept video feed

    def __init__(self, image):
        self.image = image
        self.knownWidth = None
        self.focalLength = None
        self.perWidth = None

    def find_marker(self):
        """Marker instantiation"""
        # convert the image to grayscale, blur it, and detect edges
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)
        # find the contours in the edged image and keep the largest one;
        # we'll assume that this is our piece of paper in the image
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # compute the bounding box of the of the paper region and return it
        return cv2.minAreaRect(c)

    def distance_to_camera(self):
        """"Distance from camera"""
        # compute and return the distance from the maker to the camera
        return (self.knownWidth * self.focalLength) / self.perWidth
    ########################Getters and Setters ###############################

    def _set_width(self, width):
        self.knownWidth = width

    def _get_width(self):
        return self.knownWidth

    def _set_focal_length(self, length):
        self.focalLength = length

    def _get_focal_length(self):
        return self.focalLength

    def _set_per_width(self, width):
        self.perWidth = width

    def _get_per_wifth(self):
        return self.perWidth
    ########################Getters and Setters ###############################


if __name__ == "__main__":
    # initialize the known distance from the camera to the object, which
    # in this case is 24 inches
    KNOWN_DISTANCE = 24.0
    # initialize the known object width, which in this case, the piece of
    # paper is 12 inches wide
    KNOWN_WIDTH = 11.0
    # load the furst image that contains an object that is KNOWN TO BE 2 feet
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length
    image = cv2.imread("images/2ft.png")
    marker = find_marker(image)
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    # loop over the images
    for imagePath in sorted(paths.list_images("images")):
        # load the image, find the marker in the image, then compute the
        # distance to the marker from the camera
        image = cv2.imread(imagePath)
        marker = find_marker(image)
        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

        # draw a bounding box around the image and display it
        box = cv2.cv.BoxPoints(
            marker) if imutils.is_cv2() else cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        cv2.putText(image, "%.2fft" % (inches / 12),
                    (image.shape[1] - 200, image.shape[0] -
                     20), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 3)
        cv2.imshow("image", image)
        cv2.waitKey(0)
