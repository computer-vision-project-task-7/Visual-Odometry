import numpy as np
import cv2

from typing import Tuple

fx = 520.9  #f/ro_w
fy = 521.0  #f/ro_h
cx = 325.1  #u0
cy = 249.7  #v0


def project_points(ids: np.ndarray, points: np.ndarray, depth_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects the 2D points to 3D using the depth image and the camera instrinsic parameters.

    :param ids:          A N vector point ids.
    :param points:       A 2xN matrix of 2D points
    :param depth_img:    The depth image. Divide pixel value by 5000 to get depth in meters.

    :return:             A tuple containing a N vector and a 3xN vector of all the points that where successfully projected.
    """
    #points values  (0-640, 0-480)
    print(depth_img.shape)
    #(480, 640)
    N = points.shape[1]
    # 30
    depths = np.zeros(N)

    for i in range(N):
        print(points[:, i])
        print( "0", points[0,i], "1", points[1, i])
        depths[i] = depth_img[ points[1, i].astype(int), points[0, i].astype(int)  ]/5000

    depths = depths.reshape(1, N)
    print( depths.shape )
    # (1, 30)

    enere = np.ones( (1, N) )
    # legger til en rad med enere, points går fra 2xN til 3xN
    points_homogen = np.vstack( (points, enere) )
    # cameramatrix
    K = np.array([   [fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]  ])

    # finner s_hatt
    s_hatt = np.linalg.inv(K) @ points_homogen
    # (3,30)
    s_linjer = np.sqrt(s_hatt[1]**2 + s_hatt[0]**2)
    alphas = np.arctan( s_linjer ).reshape(1, N)
    print("alphas", alphas.shape)
    print("np.cos(alphas)", np.cos(alphas).reshape(1,N).shape )


    Zs = np.cos(alphas).reshape(1, N) * depths

    print("Zs", Zs.shape, "s_hatt", s_hatt.shape)
    # (1, 30),    (3, 30)
    r = Zs * s_hatt
    # ( 3, 30)

    return (ids, r)
