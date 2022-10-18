import cv2
import numpy as np
from util import showImage, drawPerspective, drawBoundaries, drawContour, drawPoint, writedoc
from line import Line, partitionLines, filterCloseLines


def getPerspective(image, points, houghThreshold=160, hough_threshold_step=20):
    yy, xx, _ = image.shape
    tmp = np.zeros(image.shape[0:2], np.uint8);
    drawContour(tmp, points, (255,), 1)

    grid = None
    for i in range(houghThreshold//hough_threshold_step):

        hough_lines = cv2.HoughLines(tmp, 1, np.pi / 180,
                                     houghThreshold - i * hough_threshold_step)  # numpy array
        if hough_lines is None:
            continue

        lines = [Line(l[0], l[1]) for l in hough_lines.squeeze(axis=1)]  # list of Line objects

        (horizontal, vertical) = partitionLines(lines)
        vertical = filterCloseLines(vertical, horizontal=False)
        horizontal = filterCloseLines(horizontal, horizontal=True)

        if len(vertical) == 2 and len(horizontal) == 2:
            grid = (vertical, horizontal)
            break

    if grid is None:
        return None

    if vertical[0].getCenter()[0] > vertical[1].getCenter()[0]:
        v2, v1 = vertical
    else:
        v1, v2 = vertical

    if horizontal[0].getCenter()[1] > horizontal[1].getCenter()[1]:
        h2, h1 = horizontal
    else:
        h1, h2 = horizontal

    return h1.intersect(v1), h1.intersect(v2), h2.intersect(v2), h2.intersect(v1)

