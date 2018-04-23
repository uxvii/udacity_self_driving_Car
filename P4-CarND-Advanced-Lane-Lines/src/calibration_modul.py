import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg

def camera_calibration():
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
	objp = np.zeros((9*6,3), np.float32)
	objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.

	# Make a list of calibration images
	images = glob.glob('../camera_cal/calibration*.jpg')

	# Step through the list and search for chessboard corners
	for idx, fname in enumerate(images):
	    img = cv2.imread(fname)
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	    # Find the chessboard corners
	    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

	    # If found, add object points, image points
	    if ret == True:
	        objpoints.append(objp)
	        imgpoints.append(corners)



 	#calculate distortion matrix and camera instinct matrix
	
	img = cv2.imread('../camera_cal/calibration2.jpg')
	img_size = (img.shape[1], img.shape[0])

	# Do camera calibration given object points and image points
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


	dst = cv2.undistort(img, mtx, dist, None, mtx)
	cv2.imwrite('../output_images/test_undist.jpg',dst)


	# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
	dist_pickle = {}
	dist_pickle["mtx"] = mtx  #camera instincts
	dist_pickle["dist"] = dist  #distortion matrix
	pickle.dump( dist_pickle, open( "wide_dist_pickle.p", "wb" ) )

	return 0


def undistortion(img):
	dist_pickle=pickle.load(open("wide_dist_pickle.p","rb"))
	mtx=dist_pickle["mtx"]
	dist=dist_pickle["dist"]

	dst = cv2.undistort(img, mtx, dist, None, mtx)
	return dst




if __name__ == '__main__':

    img = cv2.imread('../test_images/test1.jpg')
    
    img_undistorted = undistortion(img)

    cv2.imwrite('../output_images/test_calibration_before.jpg', img)
    cv2.imwrite('../output_images/test_calibration_after.jpg', img_undistorted)


