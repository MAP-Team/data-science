# import cv2
# import numpy as np

# def track_ping_pong_ball():

# 	# Initialize camera
# 	camera = cv2.VideoCapture(0)

# 	# Run until ESC key has been pressed:
# 	while cv2.waitKey(1) is not 27:

# 		# Get video feed
# 		_, frame = camera.read()

# 		# Create color HSV color range for orange
# 		lower_range = np.array([5, 130, 110])
# 		upper_range = np.array([30, 255, 255])

# 		# Get HSV values from frame
# 		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 		# Find intersections among hvs and our color range
# 		mask = cv2.inRange(hsv, lower_range, upper_range)

# 		# Draw circles around detected circles in mask
# 		# with_circles = add_border_to_circles(frame, mask)
# 		# cv2.imshow("Detected Circles", with_circles)

# 		cv2.imshow("Detected Orange", mask)

# 	# Shut camera and close out all created windows
# 	camera.release()
# 	cv2.destroyAllWindows()

# def calibrate_color_mask():

# 	# Initialize camera
# 	camera = cv2.VideoCapture(0)

# 	# Create trackbars to alter the HSV values:
# 	cv2.namedWindow('Trackbars')
# 	cv2.createTrackbar("L - H", "Trackbars", 0, 255, lambda x: None)
# 	cv2.createTrackbar("L - S", "Trackbars", 0, 255, lambda x: None)
# 	cv2.createTrackbar("L - V", "Trackbars", 0, 225, lambda x: None)
# 	cv2.createTrackbar("U - H", "Trackbars", 255, 255, lambda x: None)
# 	cv2.createTrackbar("U - S", "Trackbars", 255, 255, lambda x: None)
# 	cv2.createTrackbar("U - V", "Trackbars", 255, 255, lambda x: None)

# 	# Run until ESC key has been pressed
# 	while cv2.waitKey(1) is not 27:

# 		# Get video feed
# 		_, frame = camera.read()

# 		# Read HSV values from trackbar positions:
# 		l_h = cv2.getTrackbarPos("L - H", "Trackbars")
# 		l_s = cv2.getTrackbarPos("L - S", "Trackbars")
# 		l_v = cv2.getTrackbarPos("L - V", "Trackbars")
# 		u_h = cv2.getTrackbarPos("U - H", "Trackbars")
# 		u_s = cv2.getTrackbarPos("U - S", "Trackbars")
# 		u_v = cv2.getTrackbarPos("U - V", "Trackbars")

# 		# Use trackbar HSV values to create color range for image mask:
# 		lower_range = np.array([l_h, l_s, l_v])
# 		upper_range = np.array([u_h, u_s, u_v])

# 		# Get HSV values from frame:
# 		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 		# Create and display masked frame:
# 		mask = cv2.inRange(hsv, lower_range, upper_range)
# 		cv2.imshow("Masked", mask)

# 	# Shut camera and close out created windows
# 	camera.release()
# 	cv2.destroyAllWindows()

# def add_border_to_circles(og_frame, output_frame):

# 	# Make copy of original frame
# 	with_circles = output_frame.copy()

# 	# Convert to black and white
# 	bw_frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2GRAY)

# 	# Detect circles in b&w frame
# 	circles = cv2.HoughCircles(bw_frame, cv2.HOUGH_GRADIENT, 1.2, 100)

# 	# If circles have been detected
# 	if circles is not None:
# 		circles = np.round(circles[0, :]).astype('int')

# 		# Add circle bounds to frame
# 		for (x, y, r) in circles:
# 			cv2.circle(
# 				with_circles,  # The image to draw the circle on
# 				(x, y),  # The coordinates of the center of the circle
# 				r,       # The radius of the circle
# 				(17, 202, 190),  # The HSV color values of drawn circle, I went with orange
# 				4        # The thickness of drawn circle
# 			)

# 	return np.hstack([output_frame, with_circles])


# if __name__ == '__main__':
# 	track_ping_pong_ball()

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
from video import create_capture
from common import clock, draw_str


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


def main():
    import sys
    import getopt

    args, video_src = getopt.getopt(
        sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get(
        '--cascade', "./haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn = args.get('--nested-cascade',
                         "./haarcascades/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(
        cv.samples.findFile('./data/lena.jpg')))

    while True:
        _ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv.imshow('facedetect', vis)

        if cv.waitKey(5) == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
