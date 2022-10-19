#!/bin/env python3

import cv2
import numpy as np
import argparse


class Line:
   def __init__(self, rho, theta):
      self._rho = rho
      self._theta = theta
      self._cos_factor = np.cos(theta)
      self._sin_factor = np.sin(theta)
      self._center = (self._cos_factor * rho, self._sin_factor * rho)

   def get_center(self):
      return self._center

   def get_rho(self):
      return self._rho

   def get_theta(self):
      return self._theta

   def get_segment(self, lenLeft, lenRight):
      a = self._cos_factor
      b = self._sin_factor
      (x0, y0) = self._center

      x1 = int(x0 + lenRight * (-b))
      y1 = int(y0 + lenRight * a)
      x2 = int(x0 - lenLeft * (-b))
      y2 = int(y0 - lenLeft  * a)
      return ((x1, y1), (x2, y2))

   def is_horizontal(self, thresholdAngle=np.pi / 4):
      return abs(np.sin(self._theta)) > np.cos(thresholdAngle)

   def is_vertical(self, thresholdAngle=np.pi / 4):
      return abs(np.cos(self._theta)) > np.cos(thresholdAngle)

   def intersect(self, line):
      ct1 = np.cos(self._theta)
      st1 = np.sin(self._theta)
      ct2 = np.cos(line._theta)
      st2 = np.sin(line._theta)
      d = ct1 * st2 - st1 * ct2
      if d == 0.0: raise ValueError('parallel lines: %s, %s)' % (str(self), str(line)))
      x = (st2 * self._rho - st1 * line._rho) / d
      y = (-ct2 * self._rho + ct1 * line._rho) / d
      return (x, y)

   def draw(self, image, color=(0,0,255), thickness=2):
      p1, p2 = self.getSegment(1000,1000)
      cv2.line(image, p1, p2, color, thickness)


   def __repr__(self):
      return "(t: %.2fdeg, r: %.0f)" % (self._theta *360/np.pi, self._rho)


def partition_lines(lines):
   h = filter(lambda x: x.is_horizontal(), lines)
   v = filter(lambda x: x.is_vertical(), lines)

   h = [(l._center[1], l) for l in h]
   v = [(l._center[0], l) for l in v]

   h.sort()
   v.sort()

   h = [l[1] for l in h]
   v = [l[1] for l in v]

   return (h, v)


def filter_close_lines(lines, horizontal=True, threshold = 40):
   if horizontal:
      item = 1
   else:
      item = 0

   i = 0
   ret = []

   while i < len(lines):
      itmp = i
      while i < len(lines) and (lines[i]._center[item] - lines[itmp]._center[item] < threshold):
         i += 1
      ret.append(lines[itmp + int((i - itmp) / 2)])

   return ret


def draw_contour(image, contour, color, thickness=4):
   rnd = lambda x : (round(x[0]), round(x[1]))
   for i in range(len(contour)):
      p1 = tuple(contour[i])
      p2 = tuple(contour[int((i+1) % len(contour))])
      cv2.line(image, rnd(p1), rnd(p2), color, thickness)


class Contours:
   
   def __init__(self, img):
      im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      _, self.im_bw = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
      self.contours, self.hierarchy = cv2.findContours(self.im_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)


   def __getitem__(self, index):
      return self.contours[index]


   def longest(self):
      """Finds the contour enclosing the largest area.
      :param contours: list of contours
      :returns: the largest countour
      """

      longest = (0, [])
      for c in self.contours:
         contour_area = cv2.contourArea(c)
         if contour_area > longest[0]:
            longest = (contour_area, c)

      return longest[1]


   def filter(self, min_ratio_bounding=0.6, min_area_percentage=0.01, max_area_percentage=0.40):
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

      hierarchy = self.hierarchy
      if hierarchy is not None:
         while len(hierarchy.shape) > 2:
            hierarchy = np.squeeze(hierarchy, 0)

      img_area = self.im_bw.shape[0] * self.im_bw.shape[1]
 
      ratio = lambda a, b : min(a,b)/float(max(a,b)) if a != 0 and b != 0 else -1
 
      for c in self.contours:
         i += 1
 
         if hierarchy is not None and not hierarchy[i][2] == -1:
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


class Perspective:
   def __init__(self, a, b, c, d):
      self.perspective = a, b, c, d 


   @classmethod
   def get_perspective(cls, image, points, houghThreshold=160, hough_threshold_step=20):
      tmp = np.zeros(image.shape[0:2], np.uint8);
      draw_contour(tmp, points, (255,), 1)

      grid = None
      for i in range(houghThreshold//hough_threshold_step):

         hough_lines = cv2.HoughLines(tmp, 1, np.pi / 180,
                                      houghThreshold - i * hough_threshold_step)  # numpy array
         if hough_lines is None:
            continue

         lines = [Line(l[0], l[1]) for l in hough_lines.squeeze(axis=1)]  # list of Line objects

         horizontal, vertical = partition_lines(lines)
         vertical = filter_close_lines(vertical, horizontal=False)
         horizontal = filter_close_lines(horizontal, horizontal=True)

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

      return cls(h1.intersect(v1), h1.intersect(v2), h2.intersect(v2), h2.intersect(v1))


   def get_image(self, image, w, h, dest=None):
      if dest is None:
         dest = ((0,0), (w, 0), (w,h), (0, h))

      perspective = self.perspective
      if perspective is None:
         im_w, im_h,_ = image.shape
         perspective = ((0,0), (im_w, 0), (im_w,im_h), (0, im_h))

      perspective = np.array(perspective ,np.float32)
      dest = np.array(dest ,np.float32)

      coeffs = cv2.getPerspectiveTransform(perspective, dest)
      return cv2.warpPerspective(image, coeffs, (w, h))


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
      horizontal, vertical = partition_lines(lines)
      vertical = filter_close_lines(vertical, horizontal=False, threshold=close_threshold_v)
      horizontal = filter_close_lines(horizontal, horizontal=True, threshold=close_threshold_h)

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
