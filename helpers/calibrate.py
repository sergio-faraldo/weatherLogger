#!/usr/bin/env python
import json

import cv2
import numpy as np
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('./calibration-images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        raise RuntimeError(f"Could not read calibration image: {fname}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if prev_img_shape is None:
        prev_img_shape = gray.shape
    elif gray.shape != prev_img_shape:
        raise ValueError("Calibration images must share one resolution")
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow('img',img)
    cv2.waitKey(1000)

cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the
detected corners (imgpoints)
"""
if not objpoints or prev_img_shape is None:
    raise RuntimeError("No usable checkerboard detections were found")
image_size = (prev_img_shape[1], prev_img_shape[0])  # (width, height)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)
calibration = {
    "ret": ret,
    "mtx": mtx.tolist(),
    "dist": dist.tolist(),
    "rvecs": [r.tolist() for r in rvecs],
    "tvecs": [t.tolist() for t in tvecs]
}

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

with open("calibration.json", "w", encoding="utf-8") as file:
    json.dump(calibration, file, indent=4)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)
