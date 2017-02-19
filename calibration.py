# import all required libraries here
import numpy as np
import cv2
import glob
import pickle
import utils
import os

# Set default file paths
CALIBRATION_FILES = './camera_cal/calibration*.jpg'
OUT_FILE = './data/calibration.p'


def calibrate_camera(path=CALIBRATION_FILES, out=OUT_FILE, force=False):
    '''
    Function to calibrate the camera. Accepts a set of images and uses opencv's functions to return the distortion co-efficients and camera matrix.
    :param path: path where the images exist
    :param out: name of the file where calculated values should be stored
    :param force: boolean to force re-calibrate
    :return: the distortion co-efficients and camera matrix
    '''

    # if recalibration is not necessary return the calculated values from the stored file
    if not force:
        if os.path.isfile(out):
            calibration_data = pickle.load(open("./data/calibration.p", "rb"))
            mtx = calibration_data['matrix']
            dist = calibration_data['distortion']

            print('Calibration Data successfully loaded.')
            return mtx, dist

    # load the file and convert to grayscale
    files = glob.glob(path)
    calibration_images = [utils.convert_bgr_grayscale(utils.load_image(file)) for idx, file in enumerate(files)]

    # display the first four images loaded
    # we can see there are 9 horizontal inside corners and 6 vertical inside corners in the calibration image
    utils.display_image_grid(calibration_images[0:4], colormap='gray')

    # prepare the object points for each image
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Lists to store object points and image points for all the images.
    objpoints = []
    imgpoints = []

    image_shape = (calibration_images[0].shape[0], calibration_images[0].shape[1])

    # for each of the image use opencv's findChessboardCorners function to get the image points
    for idx, image in enumerate(calibration_images):

        if (image.shape[0] != image_shape[0]) or (image.shape[1] != image_shape[1]):
            print("Image {} has a different shape {},{}".format(idx, image.shape[0], image.shape[1]))
            continue

        ret, corners = cv2.findChessboardCorners(image, (9, 6), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("Could not find chessboard corners for {}".format(idx))
            continue

    # use the object and image points to get the required calculations
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

    # save the calculated value
    calibration_data = {}
    calibration_data['matrix'] = mtx
    calibration_data['distortion'] = dist
    calibration_data['rvecs'] = rvecs
    calibration_data['tvecs'] = tvecs
    pickle.dump(calibration_data, open(out, "wb"))

    print('Calibration Data successfully saved.')

    # return the camera matrix and distortion co-efficient
    return mtx, dist
