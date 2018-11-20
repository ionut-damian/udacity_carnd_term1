import numpy as np
import cv2

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

def computeCalibMatrix(images, nx, ny, show_calib_result=False):
    """
    Computes the calibration matrix and distortion coefficients from a list of chessboard images
    Returns the matrix and the coefficients
    """
    #arrays to store points for all images
    objpoints = [] #describes real positions of chessboard corners
    imgpoints = [] #describes actual positions of chessboard corners in image

    #define an array to store the object points of one image
    #will be the same for every image since the checkerboard does not change
    objp = np.zeros((nx*ny,3), np.float32)

    #initialize the array with the indices of the chessboard corners
    # [0,0]
    # [1,0]
    # [2,0]
    # ...
    # [0,1]
    # [1,1]
    # ...
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    
    shape = 0
    for fname in images:
        img = cv2.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            if show_calib_result:
                img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
        
    #compute calibration matrix and distortion params
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape[::-1], None, None)

    return mtx, dist

def undistort(img, mtx, dist):    
    return cv2.undistort(img, mtx, dist, None, mtx)

def warp(img, M):   
    """
    warp image according to a perspective tranformation
    """ 
    shape = img.shape[1::-1]
    return cv2.warpPerspective(img, M, shape, flags=cv2.INTER_LINEAR)

def computePerspectiveTransform(img, draw_roi=False):  
    """ 
    Computes perspective transform, returns M
    """    
    shape = img.shape[1::-1]  
    rect_src = np.float32(
        # [[0.452 * shape[0], 0.63 * shape[1]], #top-left
        # [0.547 * shape[0], 0.63 * shape[1]], #top-right
        # [0.87 * shape[0], 0.95 * shape[1]], #bottom-right
        # [0.15 * shape[0], 0.95 * shape[1]]] #bottom-left
        [[560, 475], #top-left
        [725, 475], #top-right
        [1100, 719], #bottom-right
        [216, 719]] #bottom-left
    )

    rect_dst = np.float32(
        [[0.2 * shape[0], 0.15 * shape[1]], #top-left
        [0.8 * shape[0], 0.15 * shape[1]], #top-right
        [0.8 * shape[0], 0.9 * shape[1]], #bottom-right
        [0.2 * shape[0], 0.9 * shape[1]]] #bottom-left
    )

    if draw_roi:
        cv2.polylines(img, np.int32([rect_src]), True, (0,0,255), 4)

    M = cv2.getPerspectiveTransform(rect_src, rect_dst)
    M_inv = cv2.getPerspectiveTransform(rect_dst, rect_src)
    
    return M, M_inv