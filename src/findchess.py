#!/bin/env python3

from __future__ import annotations
import cv2
import numpy as np
import argparse
import sys
# from scalene import scalene_profiler  # SIGSEGV; possible issue with cv2
from numpy.typing import NDArray
from typing import Type, List, Dict, Tuple, Optional, Union

BGRColor = Tuple[int, int, int]
Coord = Tuple[float, float]
GridCoord = Tuple[int, int]
CV2Contour = NDArray

MISSING_BOARD = -1, -1
UNKNOWN = -1  # num boards a priori
RED = (0, 0, 255)  # BGR
MAX_SEGMENT = 1000  # pixels


class LineError(Exception):
    pass


class Line:
    """
    Representation of a line in Hough space.
    """

    def __init__(self, rho: float, theta: float):

        if rho < 0:
            raise LineError(f"Expected parameter rho >= 0 but got rho = {rho:.3f}")

        if theta < 0 or theta > np.pi:
            raise LineError(f"Expected parameter theta to be >= 0 and < np.pi but got theta = {theta:.3f}")

        self._rho = rho
        self._theta = theta
        self._cos_factor: float = np.cos(theta)
        self._sin_factor: float = np.sin(theta)
        self._center: Coord = self._cos_factor * rho, self._sin_factor * rho

    def get_center(self) -> Coord:
        return self._center

    def get_rho(self) -> float:
        return self._rho

    def get_theta(self) -> float:
        return self._theta

    def get_segment(self, left_len: float, right_len: float) -> tuple[Coord, Coord]:
        """ Calculate the x-y coordinates of the ends of a segment derived from the line.

        Arguments:
            left_len (float): distance from the 'center' to the 'left' end of the segment
            right_len (float): distance from the 'center' to the 'right' end of the segment

        Returns:
            A tuple of 2-tuples containing the x-y coordinates of the line segment endpoints in image space.
        """
        if left_len < 0 or right_len < 0:
            raise LineError("Distance from center must be positive")

        a = self._cos_factor
        b = self._sin_factor
        (x0, y0) = self._center

        x1 = round(x0 + right_len * (-b))
        y1 = round(y0 + right_len * a)
        x2 = round(x0 - left_len * (-b))
        y2 = round(y0 - left_len * a)
        return ((x1, y1), (x2, y2))

    def inside_threshold_range(self, threshold: float = np.pi/4) -> bool:
        return self._theta > threshold and self._theta < (np.pi - threshold)

    def intersect(self, line: Line) -> Coord:
        ct1, st1 = self._cos_factor, self._sin_factor
        ct2, st2 = line._cos_factor, line._sin_factor
        d = ct1 * st2 - st1 * ct2

        if np.isclose(d, 0.0):  # default atol = 1e-8
            raise LineError('Parallel lines: %s, %s)' % (str(self), str(line)))

        x = (st2 * self._rho - st1 * line._rho) / d
        y = (-ct2 * self._rho + ct1 * line._rho) / d
        return (x, y)

    def draw(self, image, length: int = MAX_SEGMENT, color: BGRColor = RED, thickness: int = 2):
        p1, p2 = self.get_segment(length, length)
        cv2.line(image, p1, p2, color, thickness)

    def __repr__(self) -> str:
        return "(t: %.2fdeg, r: %.0f)" % (self._theta * 180/np.pi, self._rho)

    def __eq__(self, line: Line) -> bool:
        return np.isclose(self._rho, line._rho) and np.isclose(self._theta, line._theta)  # atol = 1e-8

    @staticmethod
    def partition_lines(lines: List[Line]) -> tuple[List[Line], List[Line]]:
        inside = list(filter(lambda line: line.inside_threshold_range(), lines))
        outside = list(filter(lambda line: not line.inside_threshold_range(), lines))
        return inside, outside

    @staticmethod
    def discard_close_lines(sorted_lines: List[Line], horizontal: bool = True, threshold: int = 40) -> List[Line]:

        index = 1 if horizontal else 0

        def find_cluster_starting_at(i):
            start = i
            start_center = sorted_lines[start]._center[index]
            while i < len(sorted_lines) and (sorted_lines[i]._center[index] - start_center < threshold):
                i += 1
            middle = (i - start)//2
            return sorted_lines[start + middle], i

        next = 0
        filtered = []
        while next < len(sorted_lines):
            cluster_middle, next = find_cluster_starting_at(next)
            filtered.append(cluster_middle)

        return filtered


def draw_contour(image: NDArray, contour: CV2Contour, color: BGRColor, thickness: int = 4):
    def rnd(x): return (round(x[0]), round(x[1]))
    for i in range(len(contour)):
        p1 = tuple(contour[i])
        p2 = tuple(contour[int((i+1) % len(contour))])
        cv2.line(image, rnd(p1), rnd(p2), color, thickness)


class Contours:

    def __init__(self, img: NDArray):
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, self.im_bw = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        self.contours, self.hierarchy = cv2.findContours(self.im_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    def __getitem__(self, index: int):
        return self.contours[index]

    def longest(self) -> Optional[CV2Contour]:
        """Finds the contour enclosing the largest area.
        :returns: the largest countour
        """
        longest: Tuple[float, Optional[CV2Contour]] = (0, None)
        for c in self.contours:
            contour_area = cv2.contourArea(c)
            if contour_area > longest[0]:
                longest = (contour_area, c)

        return longest[1]

    def filter(self, min_ratio_bounding=0.6, min_area_percentage=0.01, max_area_percentage=0.40) -> List[CV2Contour]:
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

        def ratio(a, b): return min(a, b)/float(max(a, b)) if a != 0 and b != 0 else -1

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
            if ratio(h, w) < min_ratio_bounding:
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
    def __init__(self, a: Coord, b: Coord, c: Coord, d: Coord):
        self.corners = a, b, c, d

    # To keep old notebooks working
    @classmethod
    def get_perspective(cls: Type[Quadrangle], image: NDArray, points: CV2Contour,
                        houghThreshold: int = 160, hough_threshold_step: int = 20) -> Optional[Quadrangle]:
        return cls.get_quad(image, points, houghThreshold, hough_threshold_step)

    @classmethod
    def get_quad(cls: Type[Quadrangle], image: NDArray, points: CV2Contour,
                 houghThreshold: int = 160, hough_threshold_step: int = 20) -> Optional[Quadrangle]:
        tmp = np.zeros(image.shape[0:2], np.uint8)
        draw_contour(tmp, points, (255, 255, 255), 1)

        grid = None
        for i in range(houghThreshold//hough_threshold_step):

            hough_lines = cv2.HoughLines(tmp, 1, np.pi / 180,
                                         houghThreshold - i * hough_threshold_step)  # numpy array
            if hough_lines is None:
                continue

            lines = [Line(line[0], line[1]) for line in hough_lines.squeeze(axis=1)]  # list of Line objects

            horizontal, vertical = Line.partition_lines(lines)
            vertical.sort(key=lambda line: line._center[0])
            vertical = Line.discard_close_lines(vertical, horizontal=False)
            horizontal.sort(key=lambda line: line._center[1])
            horizontal = Line.discard_close_lines(horizontal, horizontal=True)

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

    def perspective_corr(self, image: NDArray, w: int, h: int,
                         dest: Optional[Tuple[Coord, Coord, Coord, Coord]] = None) -> NDArray:
        if dest is None:
            dest = ((0, 0), (w, 0), (w, h), (0, h))

        corners = self.corners
        if corners is None:
            im_w, im_h, _ = image.shape
            corners = ((0, 0), (im_w, 0), (im_w, im_h), (0, im_h))

        quadrangle = np.array(corners, np.float32)
        dest_arr = np.array(dest, np.float32)

        coeffs = cv2.getPerspectiveTransform(quadrangle, dest_arr)
        return cv2.warpPerspective(image, coeffs, (w, h))

    # support older coder
    def correction(self, image: NDArray, w: int, h: int,
                   dest: Optional[Tuple[Coord, Coord, Coord, Coord]] = None) -> NDArray:
        return self.perspective_corr(image, w, h, dest)


def extract_boards(img: NDArray, grid: Optional[Tuple[int, int]] = None, priority: str = "row",
                   correction: bool = False, brdsize: Optional[int] = None
                   ) -> Tuple[List[Union[NDArray, None]], Union[List[GridCoord], List[int]]]:
    """Extracts all boards from an image.

    Arguments:
        img (numpy array): image containing chess diagrams allegedly

    Keyword arguments:
        grid (2-tuple): arrangement of boards in rectangular grid as (numrows, numcols)
        priority (str): "row" or "col" -- count by row first or by column when labelling boards in a grid.
                          Ignored if grid is None.
        correction (bool): True => attempt perspective correction
        brdsize (int): required if correction asked; size of perspective corrected image

    Returns:
        A 2-tuple: (list of extracted board images, list of board labels)
    """

    contours = Contours(img)
    filtered = contours.filter()
    boards: List[Union[NDArray, None]] = []
    centroids = []
    def centroid(m): return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
    for contour in filtered:
        contour_arr = np.squeeze(contour, 1)
        if correction:
            assert brdsize is not None
            assert isinstance(brdsize, int)
            quad = Quadrangle.get_quad(img, contour_arr)
            if quad:
                b = quad.perspective_corr(img, brdsize, brdsize)
            else:
                b = None
        else:
            x, y, w, h = cv2.boundingRect(contour_arr)
            b = img[y:y+h, x:x+w]
        boards.append(b)
        centroids.append(np.array(centroid(cv2.moments(contour))))

    if grid:

        assert len(grid) == 2
        numrows, numcols = grid
        img_height, img_width, _ = img.shape
        grid_height = img_height // numrows
        grid_width = img_width // numcols
        centroids_arr = np.array(centroids)

        def label(index: int) -> tuple[int, int]:
            try:
                col = centroids_arr[index][0] // grid_width
                row = centroids_arr[index][1] // grid_height
            except IndexError:
                return MISSING_BOARD
            return row, col

        labels: Union[List[GridCoord], List[int]] = [label(i) for i in range(numrows*numcols)]

    else:
        labels = list(range(len(boards)))

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
    parser.add_argument('-p', '--print', action='store_true',
                        help="just print the number of chess boards found; don't save")
    parser.add_argument('-o', '--offset', type=int, default=0, help="offset label by constant")

    args = parser.parse_args()

    if args.rows and not args.cols:
        print('Missing column spec')
        sys.exit()

    if args.cols and not args.rows:
        print('Missing rows spec')
        sys.exit()

    import time
    start = time.time()
    # scalene_profiler.start()

    if args.rows and args.cols:
        numboards = args.rows * args.cols
    else:
        numboards = UNKNOWN

    missing: Dict[str, List[Tuple[np.signedinteger, ...]]] = dict()
    for filenum, filename in enumerate(args.filenames):
        image = cv2.imread(filename)

        if args.rows and args.cols:
            boards, labels = extract_boards(image, grid=(args.rows, args.cols), priority=args.label)
        else:
            boards, labels = extract_boards(image)

        if args.print:
            print(f"{filename}: {len(boards)}")

        else:
            print(f"{filename}")
            if numboards != UNKNOWN:

                board_idx = 0
                found = []
                for i in range(numboards):

                    if labels[i] == MISSING_BOARD:
                        continue

                    row, col = labels[i]  # type: ignore
                    board_num = col + row*args.cols if args.label == "row" else row + col*args.rows
                    label = filenum*numboards + board_num + 1
                    savefile = f"{args.offset + label:04d}.jpg"
                    b = boards[board_idx]
                    if b is None:
                        continue
                    if args.crop:
                        cv2.imwrite(savefile, b[args.crop:-args.crop, args.crop:-args.crop])
                    else:
                        cv2.imwrite(savefile, b)
                    board_idx += 1
                    found.append(labels[i])

                for j in range(numboards):
                    coord = np.unravel_index(j, (args.rows, args.cols))
                    if coord in found:
                        continue
                    if filename in missing:
                        missing[filename].append(coord)
                    else:
                        missing[filename] = [coord]

            else:
                pass

    if missing:
        print("Missing boards:")
        total_missing = 0
        for i, m in enumerate(missing.items()):
            print(f"{i+1:04d}. {m[0]}, Board: {m[1]}")
            total_missing += len(m[1])
        print(f"Total missing = {total_missing} boards")
        print(f"Miss percentage = {total_missing/(len(args.filenames)*numboards)*100:.1f}%")

    end = time.time()
    print(f"Time taken = {end - start:.3f} seconds")
    # scalene_profiler.stop()
