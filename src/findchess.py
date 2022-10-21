#!/bin/env python3

import cv2
import numpy as np
import argparse
import sys


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
      p1, p2 = self.get_segment(1000,1000)
      cv2.line(image, p1, p2, color, thickness)


   def __repr__(self):
      return "(t: %.2fdeg, r: %.0f)" % (self._theta *360/np.pi, self._rho)


def partition_lines(lines: list[Line]):
   h = filter(lambda x: x.is_horizontal(), lines)
   v = filter(lambda x: x.is_vertical(), lines)

   h = [(l._center[1], l) for l in h]
   v = [(l._center[0], l) for l in v]

   h.sort(key=lambda x: x[0])
   v.sort(key=lambda x: x[0])

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
      :param min_ratio_bounding: minimum contour area vs. bounding box area ratio
      :param min_area_percentage: minimum contour vs. image area percentage
      :param max_area_percentage: maximum contour vs. image area percentage
      :returns: list of contours
      """

      hierarchy = self.hierarchy
      if hierarchy is None:
         return []

      while len(hierarchy.shape) > 2:
         hierarchy = np.squeeze(hierarchy, 0)

      img_area = self.im_bw.shape[0] * self.im_bw.shape[1]
 
      ratio = lambda a, b : min(a,b)/float(max(a,b)) if a != 0 and b != 0 else -1

      NEXT_SIBLING, PREV_SIBLING, FIRST_CHILD, PARENT = 0, 1, 2, 3
      NO_NODE = -1
      parents = []
      filtered = []
      for i, c in enumerate(self.contours):
 
         if hierarchy[i][FIRST_CHILD] != NO_NODE:
            continue

         # TODO: Only perform the following if num boards known and num boards > 1
         if hierarchy[i][NEXT_SIBLING] == NO_NODE and hierarchy[i][PREV_SIBLING] == NO_NODE:
            continue
 
         _, _, w, h = cv2.boundingRect(c)
         if ratio(h,w) < min_ratio_bounding:
            continue
 
         contour_area = cv2.contourArea(c)
         img_contour_ratio = ratio(img_area, contour_area)
         if img_contour_ratio < min_area_percentage:
            continue
         if img_contour_ratio > max_area_percentage:
            continue

         parents.append(hierarchy[i][PARENT])
         filtered.append(c)
 
      if not np.all(parents == parents[0]):
         print("WARNING: Contour hierarchies with different parents.")

      return filtered


class Quadrangle:
   def __init__(self, a, b, c, d):
      self.corners = a, b, c, d


   # To keep old notebooks working
   @classmethod
   def get_perspective(cls, image, points, houghThreshold=160, hough_threshold_step=20):
      return cls.get_quad(image, points, houghThreshold, hough_threshold_step)


   @classmethod
   def get_quad(cls, image, points, houghThreshold=160, hough_threshold_step=20):
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

      if vertical[0].get_center()[0] > vertical[1].get_center()[0]:
         v2, v1 = vertical
      else:
         v1, v2 = vertical

      if horizontal[0].get_center()[1] > horizontal[1].get_center()[1]:
         h2, h1 = horizontal
      else:
         h1, h2 = horizontal
      return cls(h1.intersect(v1), h1.intersect(v2), h2.intersect(v2), h2.intersect(v1))


   def perspective_corr(self, image, w, h, dest=None):
      if dest is None:
         dest = ((0,0), (w, 0), (w,h), (0, h))

      corners = self.corners
      if corners is None:
         im_w, im_h,_ = image.shape
         corners = ((0,0), (im_w, 0), (im_w,im_h), (0, im_h))

      quadrangle = np.array(corners ,np.float32)
      dest = np.array(dest ,np.float32)

      coeffs = cv2.getPerspectiveTransform(quadrangle, dest)
      return cv2.warpPerspective(image, coeffs, (w, h))
   

   def correction(self, image, w, h, dest=None):  # support older code
      return self.perspective_corr(image, w, h, dest)


def extract_boards(img, grid=None, labels="row", correction=False, brdsize=None):
   """Extracts all boards from an image.

   Arguments:
      img (numpy array): image containing chess diagrams allegedly

   Keyword arguments:
      grid (2-tuple): arrangement of boards in rectangular grid as (numrows, numcols)
      labels (str): "row" or "col" -- count by row first or by column when labelling boards in a grid.
                    Ignored if grid is None.
      correction (bool): True => attempt perspective correction
      brdsize (int): required if correction asked; size of perspective corrected image

   Returns:
      A 2-tuple: (list of extracted board images, list of board labels)
   """

   contours = Contours(img)
   filtered = contours.filter()
   boards = []
   centroids = []
   centroid = lambda m : (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
   for contour in filtered:
      contour = np.squeeze(contour, 1)
      if correction:
         assert brdsize is not None
         assert isinstance(brdsize, int)
         quad = Quadrangle.get_quad(img, contour)
         b = quad.perspective_corr(img, brdsize, brdsize)
      else:
         x, y, w, h = cv2.boundingRect(contour)
         b = img[y:y+h, x:x+w]
      boards.append(b)
      centroids.append(np.array(centroid(cv2.moments(contour))))

   if grid:

      centroids = np.array(centroids)
      xcoords, ycoords = centroids[:, 0], centroids[:, 1]
      sorted_x, sorted_y = list(np.argsort(xcoords)), list(np.argsort(ycoords))

      assert len(grid) == 2
      numrows, numcols = grid
      findc = np.array(sorted_x).reshape(numcols, numrows)
      findr = np.array(sorted_y).reshape(numrows, numcols)
      col = lambda c: np.where(findc == c)[0][0]
      row = lambda r: np.where(findr == r)[0][0]
      if labels == "col":
         labels = [row(i) + col(i)*numrows + 1 for i in range(numrows*numcols)]
      else:
         labels = [col(i) + row(i)*numcols + 1 for i in range(numrows*numcols)]

   else:
      labels = list(range(1, len(boards) + 1))

   return boards, labels


# So that older notebooks still work without changes
Perspective = Quadrangle


if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Extract chessboard images from image containing chessboard diagrams')
   parser.add_argument('filenames', nargs='+', help='Image files to process.')
   parser.add_argument('-r', '--rows', type=int,
                       help='specifiy that chess boards in a rectangular grid with "r" rows')
   parser.add_argument('-c', '--cols', type=int,
                       help='number of columns in the grid; if rows are given, cols are required')
   parser.add_argument('-l', '--label',  default="row", choices=['row', 'col'],
                       help='row-wise or column-wise labelling for boards in a grid')
   parser.add_argument('--crop', metavar='PIXELS', type=int, help='number of pixels to crop along each edge')
   parser.add_argument('-p', '--print', action='store_true', help="just print the number of chess boards found; don't save")

   args = parser.parse_args()

   if args.rows and not args.cols:
      print('Missing column spec')
      sys.exit()

   if args.cols and not args.rows:
      print('Missing rows spec')
      sys.exit()

   import time
   start = time.time()

   if args.rows and args.cols:
      offset = args.rows * args.cols
   else:
      offset = 0

   for filenum, filename in enumerate(args.filenames):
      image = cv2.imread(filename)

      if args.rows and args.cols:
         boards, labels = extract_boards(image, grid=(args.rows, args.cols), labels=args.label)
      else:
         boards, labels = extract_boards(image)

      if args.print:
         print(f"{filename}: {len(boards)}")
      else:
         print(f"{filename}")
         for i, b in enumerate(boards):
            savefile = f"{filenum*offset + labels[i]:04d}.jpg"
            if args.crop:
               cv2.imwrite(savefile, b[args.crop:-args.crop, args.crop:-args.crop])
            else:
               cv2.imwrite(savefile, b)

   end = time.time()
   print(f"Time taken = {end - start:.3f} seconds")

