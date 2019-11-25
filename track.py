
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
from video import create_capture
from common import clock, draw_str


def detect(img, cascade):
    # create grid each object will be put in a list
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    # check to see if the rectangles are empty
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def draw_rects(img, rects, color):
    # draw grid
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


def main():
    import sys
    import getopt

    # start video reading
    args, video_src = getopt.getopt(
        sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    # try all video outputs on device
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    # get xml formatted sqaure
    cascade_fn = args.get(
        '--cascade', "./haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn = args.get('--nested-cascade',
                         "./haarcascades/haarcascade_eye.xml")
    # find each cascade file
    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))
    # set our model
    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(
        cv.samples.findFile('./data/lena.jpg')))

    while True:
        # read the video output
        # return each image frame
        _ret, img = cam.read()
        # covert the image to one color space
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # this equalize function does some crazy shi
        # it basically creates a histogram of the grayscale image
        # from the video feed it computes the sum of the histogram so each
        # bin is 255 and takes the integral just to noramalize the images
        # contrast and brightness for effective reading
        gray = cv.equalizeHist(gray)
        # timer start for how long it took the algorithm to detect the object
        t = clock()
        # feed the detect function the gray image frames from video feed
        #  and the cascade for the object model
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
        # stop timer
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        # display to window our output
        cv.imshow('facedetect', vis)
        # exit key
        if cv.waitKey(5) == 27:
            break

    print('Done')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
