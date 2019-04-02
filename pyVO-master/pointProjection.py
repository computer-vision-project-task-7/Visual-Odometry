import numpy as np
import cv2

from typing import Tuple

fx = 520.9
fy = 521.0
cx = 325.1
cy = 249.7


def project_points(ids: np.ndarray, points: np.ndarray, depth_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects the 2D points to 3D using the depth image and the camera instrinsic parameters.

    :param ids:          A N vector point ids.
    :param points:       A 2xN matrix of 2D points
    :param depth_img:    The depth image. Divide pixel value by 5000 to get depth in meters.

    :return:             A tuple containing a N vector and a 3xN vector of all the points that where successfully projected.
    """

        # listify ponts array, construct z vector
        points_list = zip(p[0], p[1])
        z = []

        # getting z-values for each 2D point
        for x,y, in points_list:
            z.append( depth_img[x][y] )

        # adding z values to the 2D points, making them 3D. (3xN matrix)
        points_3d = np.vstack( (points, z) )
        # returning a tuple of ids and 3D points
        return (ids, points_3d)
