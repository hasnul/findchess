#!/bin/env python3

import cv2
import numpy as np
import argparse
from line import Line, Contours, Perspective


def extract_boards(img, w, h):
   """Extracts all boards from an image. This function applies perspective correction.
   :param img: source image
   :param w: output width
   :param h: output height
   :returns: a list the extracted board images
   """

   contours = Contours(img)
   contour_ids = contours.filter()
   boards = []
   for i in contour_ids:
      c = contours[i]
      c = np.squeeze(c,1)
      perspective = Perspective.get_perspective(img, c)
      if perspective is not None:
         b = perspective.get_image(img, w, h)
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

         perspective = Perspective(h1.intersect(v1), h1.intersect(v2), h2.intersect(v2), h2.intersect(v1))
         tile = perspective.get_image(img, w, h)
         ret.append(((x,y), tile))

   return ret


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
