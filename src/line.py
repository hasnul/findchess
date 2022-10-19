import cv2
import numpy as np


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


   # Don't need classmethods here. These methods can stand as their own functions.
   # However, the module and the class are very closely named.

   @classmethod
   def partition_lines(cls, lines):
      h = filter(lambda x: x.is_horizontal(), lines)
      v = filter(lambda x: x.is_vertical(), lines)

      h = [(l._center[1], l) for l in h]
      v = [(l._center[0], l) for l in v]

      h.sort()
      v.sort()

      h = [l[1] for l in h]
      v = [l[1] for l in v]

      return (h, v)


   @classmethod
   def filter_close_lines(cls, lines, horizontal=True, threshold = 40):
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


class Contours:
   
   def __init__(self, img):
      im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      _, self.im_bw = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
      self.contours, self.hierarchy = cv2.findContours(self.im_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)


   def __getitem__(self, index):
      return self.contours[index]


   # For convenience
   @classmethod
   def draw_contour(cls, image, contour, color, thickness=4):
      rnd = lambda x : (round(x[0]), round(x[1]))
      for i in range(len(contour)):
         p1 = tuple(contour[i])
         p2 = tuple(contour[int((i+1) % len(contour))])
         cv2.line(image, rnd(p1), rnd(p2), color, thickness)


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
   pass
