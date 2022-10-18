#!/bin/env python3

import cv2
import numpy as np
import argparse
from line import Line

def draw_contour(image, contour, color, thickness=4):
   rnd = lambda x : (round(x[0]), round(x[1]))
   for i in range(len(contour)):
      p1 = tuple(contour[i])
      p2 = tuple(contour[int((i+1) % len(contour))])
      cv2.line(image, rnd(p1), rnd(p2), color, thickness)


def longest_contour(contours):
   """Finds the contour with the largest area in the list.
   :param contours: list of contours
   :returns: the largest countour
   """

   largest = (0, [])
   for c in contours:
      contour_area = cv2.contourArea(c)
      if contour_area > largest[0]:
         largest = (contour_area, c)

   return largest[1]


def ignore_contours(img,
                   contours,
                   hierarchy=None,
                   min_ratio_bounding=0.6,
                   min_area_percentage=0.01,
                   max_area_percentage=0.40):
   """Filters a contour list based on some rules. If hierarchy != None,
   only top-level contours are considered.
   param img: source image
   :param contours: list of contours
   :param hierarchy: contour hierarchy
   :param min_ratio_bounding: minimum contour area vs. bounding box area ratio
   :param min_area_percentage: minimum contour vs. image area percentage
   :param max_area_percentage: maximum contour vs. image area percentage
   :returns: a list with the unfiltered countour ids
   """

   ret = []
   i = -1

   if hierarchy is not None:
      while len(hierarchy.shape) > 2:
         hierarchy = np.squeeze(hierarchy, 0)
   img_area = img.shape[0] * img.shape[1]

   ratio = lambda a, b : min(a,b)/float(max(a,b)) if a != 0 and b != 0 else -1

   for c in contours:
      i += 1

      if hierarchy is not None and \
         not hierarchy[i][2] == -1:
         continue

      _,_,w,h = tmp = cv2.boundingRect(c)
      if ratio(h,w) < min_ratio_bounding:
         continue

      contour_area = cv2.contourArea(c)
      img_contour_ratio = ratio(img_area, contour_area)
      if img_contour_ratio < min_area_percentage:
         continue
      if img_contour_ratio > max_area_percentage:
         continue

      ret.append(i)

   return ret


def extract_boards(img, w, h):
   """Extracts all boards from an image. This function applies perspective correction.
   :param img: source image
   :param w: output width
   :param h: output height
   :returns: a list the extracted board images
   """
   im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   _, im_bw = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

   contours,hierarchy = cv2.findContours(im_bw,  cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

   contour_ids = ignore_contours(im_bw, contours, hierarchy)
   boards = []
   for i in contour_ids:
      c = contours[i]
      c = np.squeeze(c,1)
      perspective = get_perspective(img, c)
      if perspective is not None:
         b = extract_perspective(img, perspective, w, h)
         boards.append(b)

   return boards


def extract_grid(img,
                nvertical,
                nhorizontal,
                threshold1 = 50,
                threshold2 = 150,
                apertureSize = 3,
                hough_threshold_step=20,
                hough_threshold_min=50,
                hough_threshold_max=150):
   """Finds the grid lines in a board image.
   :param img: board image
   :param nvertical: number of vertical lines
   :param nhorizontal: number of horizontal lines
   :returns: a pair (horizontal, vertical). Both elements are lists with the lines' positions.
   """


   w, h, _ = img.shape
   close_threshold_v = (w / nvertical) / 4
   close_threshold_h = (h / nhorizontal) / 4

   im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   _, im_bw = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   im_canny = cv2.Canny(im_bw, threshold1, threshold2, apertureSize=apertureSize)

   for i in range((hough_threshold_max - hough_threshold_min + 1) / hough_threshold_step):
      lines = cv2.HoughLines(im_canny, 1, np.pi / 180, hough_threshold_max - (hough_threshold_step * i))
      if lines is None:
         continue

      lines = [Line(l[0], l[1]) for l in lines.squeeze(axis=1)]
      horizontal, vertical = Line.partition_lines(lines)
      vertical = Line.filter_close_lines(vertical, horizontal=False, threshold=close_threshold_v)
      horizontal = Line.filter_close_lines(horizontal, horizontal=True, threshold=close_threshold_h)

      if len(vertical) >= nvertical and len(horizontal) >= nhorizontal:
         return (horizontal, vertical)


def extract_tiles(img, grid, w, h):
   ret = []

   for x in range(8):
      v1 = grid[1][x]
      v2 = grid[1][x+1]

      for y in range(8):
         h1 = grid[0][y]
         h2 = grid[0][y+1]

         perspective = (h1.intersect(v1), h1.intersect(v2), h2.intersect(v2), h2.intersect(v1))
         tile = extract_perspective(img, perspective, w, h)
         ret.append(((x,y), tile))

   return ret


def get_perspective(image, points, houghThreshold=160, hough_threshold_step=20):
   tmp = np.zeros(image.shape[0:2], np.uint8);
   draw_contour(tmp, points, (255,), 1)

   grid = None
   for i in range(houghThreshold//hough_threshold_step):

      hough_lines = cv2.HoughLines(tmp, 1, np.pi / 180,
                                     houghThreshold - i * hough_threshold_step)  # numpy array
      if hough_lines is None:
         continue

      lines = [Line(l[0], l[1]) for l in hough_lines.squeeze(axis=1)]  # list of Line objects

      horizontal, vertical = Line.partition_lines(lines)
      vertical = Line.filter_close_lines(vertical, horizontal=False)
      horizontal = Line.filter_close_lines(horizontal, horizontal=True)

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


def extract_perspective(image, perspective, w, h, dest=None):
   if dest is None:
      dest = ((0,0), (w, 0), (w,h), (0, h))

   if perspective is None:
      im_w, im_h,_ = image.shape
      perspective = ((0,0), (im_w, 0), (im_w,im_h), (0, im_h))

   perspective = np.array(perspective ,np.float32)
   dest = np.array(dest ,np.float32)

   coeffs = cv2.getPerspectiveTransform(perspective, dest)
   return cv2.warpPerspective(image, coeffs, (w, h))


if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='')
   parser.add_argument('filenames', metavar='filename', type=str, nargs='+',
                       help='The files to process.')

   parser.add_argument('-e', dest='extract_boards', action='store_const',
                       const=True, default=False,
                       help='extract boards from images (default: use image as-is)')

   args = parser.parse_args()

   for filename in args.filenames:
      image = cv2.imread(filename)
      print("---- %s ----" % filename)

      if args.extract_boards:
         print("Extracting Boards")

         # TODO: why these numbers?
         extract_width = 400
         extract_height = 400

         boards = extract_boards(image, extract_width, extract_height)

      else:
         boards = [image]

      for b in boards:
         print("Extracting Grid")
         grid = extract_grid(b, 9, 9)

         print(grid)
         if grid is None:
            print("Could not find Grid")
            continue

         print("Extracting Tiles")
         tiles = extract_tiles(b, grid, 100, 100)
