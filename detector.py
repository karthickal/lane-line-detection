# load all the necessary libraries
import cv2
import numpy as np
import calibration
import utils


class LaneDetector(object):
    # define all the constants and parameters required
    # define the sobel threshold parameters
    SOBEL_X_THRESH_MIN = 50
    SOBEL_X_THRESH_MAX = 255
    SOBEL_Y_THRESH_MIN = 100
    SOBEL_Y_THRESH_MAX = 255

    # define the direction parametes
    DIRECTION_THRESH_MIN = np.pi / 5
    DIRECTION_THRESH_MAX = np.pi / 2

    # define the magnitude parameters
    MAGNITUDE_THRESH_MIN = 12
    MAGNITUDE_THRESH_MAX = 255

    # define the satiration parameters
    SAT_THRESH_MIN = 85
    SAT_THRESH_MAX = 150

    # define the parameters to improve the image
    GAMMA_FACTOR = 2
    BLUR_KERNEL = 5
    BLUR_SIGMA = 1
    VISIBILITY_GAIN = 1
    VISIBILITY_BIAS = 5

    # define parameters required for lane detection
    ROI_CHANGE = 5
    ROI_MIN_CONFIDENCE = 0.6
    AVERAGE_FRAME_COUNT = 20
    MIN_CONF = 0.7
    WINDOW_SLICE_COUNT = 10
    LANE_MIN_PIX = 250
    WINDOW_MARGIN = 50
    ACTUAL_LANE_WEIGHT = 1.5

    # define other utility parameters
    ENABLE_LOGS = False

    def __init__(self):

        # initialize all the default values needed
        self.pers_mat = None
        self.inv_pers_mat = None
        self.processed_frames = 0
        self.lane_detected = False
        self.prev_overlay = None
        self.prev_left_rad = None
        self.prev_right_rad = None

        self.left_fit_list = []
        self.right_fit_list = []
        self.left_fit_conf_list = []
        self.right_fit_conf_list = []
        self.offset_list = []

        self.prev_conf_left_fit = None
        self.prev_conf_right_fit = None

        self.current_roi_src = []
        self.current_roi_dest = []

        self.conf_left_fit = None
        self.conf_right_fit = None
        self.total_confidence = 1

        self.lane_distances = {}
        self.lane_weights = {}

        # calibrate the camera and get the matrix and distortion co-efficients
        self.mtx, self.dist = calibration.calibrate_camera()
        src, dest = self.__get_default_roi()
        self.__load_perspective_matrix(src, dest)

    def __dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        '''
        Method to threshold the image based on the direction of the lines
        :param img: the image to threshold
        :param sobel_kernel: the kernel size of the sobel operator
        :param thresh: the min and max values of the direction
        :return: the thresholded image
        '''

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        grad_dir = np.arctan2(abs_sobely, abs_sobelx)

        binary_output = np.zeros_like(gray)
        binary_output[(grad_dir > thresh[0]) & (grad_dir < thresh[1])] = 1

        return binary_output

    def __sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        '''
        Method to apply the sobel operator on the image
        :param img: the image to apply sobel on
        :param orient: orientation of the sobel function
        :param sobel_kernel: the kernel size of the sobel operator
        :param thresh: the min and max values to retain
        :return: the thresholded image
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / (np.max(abs_sobel)))

        binary_output = np.zeros_like(gray)
        binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1

        return binary_output

    def __mag_threshold(self, img, sobel_kernel=3, thresh=(0, 255)):
        '''
        Method to threshold the image based on the magnitude of the pixels
        :param img: the image to apply the threshold on
        :param sobel_kernel: the kernel size of the sobel operator
        :param thresh: the min and max values to retain
        :return: the thresholded image
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        abs_sobelxy = np.sqrt(np.square(abs_sobelx) + np.square(abs_sobely))
        scaled_sobel = np.uint8(255 * abs_sobelxy / (np.max(abs_sobelxy)))

        binary_output = np.zeros_like(gray)
        binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1

        return binary_output

    def __color_threshold(self, img):
        '''
        Method to threshold the image based on the color
        :param img: the image to apply the threshold on
        :return: the thresholded image
        '''

        # convert the image to HLS space
        hls = utils.convert_rgb_hls(img)
        s_channel = hls[:, :, 2]
        binary = np.zeros_like(hls[:, :, 0])
        binary[((s_channel > self.SAT_THRESH_MIN) & (s_channel < self.SAT_THRESH_MAX))] = 1

        return binary

    def __adjust_visibility(self, img, gain=1, bias=25):
        '''
        Method to increase the visibility of the image
        :param img: the image to process
        :param gain: multiplication factor
        :param bias: addition factor
        :return: the processed image
        '''
        return (img * gain) + bias

    def __remove_noise(self, img):
        '''
        Method to remove noise from the image by applying a Gaussian Blur
        :param img: the image to process
        :return: the processed image
        '''
        return cv2.GaussianBlur(img, (self.BLUR_KERNEL, self.BLUR_KERNEL), self.BLUR_SIGMA, self.BLUR_SIGMA)

    def __adjust_gamma(self, img, gamma=1.0):
        '''
        Method to adjust the gamma factor of the image
        :param img: the image to process
        :param gamma: gamma factor
        :return: the processed image
        '''
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(img, table)

    def __prepocess(self, img):
        '''
        Method to preprocess the image and threshold it to reveal lane lines
        :param img: the image to preprocess
        :return: processed image
        '''

        # remove noise from the image, improve visibility and gamma factor
        noise_removed = self.__remove_noise(img)
        visible = self.__adjust_visibility(noise_removed, self.VISIBILITY_GAIN, self.VISIBILITY_BIAS)
        gamma_processed = self.__adjust_gamma(visible, self.GAMMA_FACTOR)

        # get the thresholded binary from various channels
        color_bin = self.__color_threshold(gamma_processed)
        binary_x = self.__sobel_thresh(gamma_processed, 'x', thresh=(self.SOBEL_X_THRESH_MIN, self.SOBEL_X_THRESH_MAX))
        binary_y = self.__sobel_thresh(gamma_processed, 'y', thresh=(self.SOBEL_Y_THRESH_MIN, self.SOBEL_Y_THRESH_MAX))
        dir_binary = self.__dir_threshold(gamma_processed,
                                          thresh=(self.DIRECTION_THRESH_MIN, self.DIRECTION_THRESH_MAX))
        mag_binary = self.__mag_threshold(gamma_processed,
                                          thresh=(self.MAGNITUDE_THRESH_MIN, self.MAGNITUDE_THRESH_MAX))

        # create the final image from the thresholded binaries
        final_image = np.zeros_like(img[:, :, 0])
        final_image[
            (((binary_x == 1) & (binary_y == 1)) | ((dir_binary == 1) & (mag_binary == 1))) | (color_bin == 1)] = 1

        if self.ENABLE_LOGS:
            print('Pre-processed Image Successfully')

        return final_image

    def __undistort(self, img, mtx, dist):
        '''
        Method to undistort the image using the camera calibration values
        :param img: the image to undistort
        :param mtx: camera matrix
        :param dist: distortion co-efficients
        :return: the undistorted image
        '''

        dest = cv2.undistort(img, mtx, dist, None, mtx)
        return dest

    def __load_perspective_matrix(self, src, dest):
        '''
        Method to get the perspective matrix from the source and destination points
        :param src: source points
        :param dest: destination points
        :return: the matrix and inverse matrix
        '''

        self.pers_mat = cv2.getPerspectiveTransform(src, dest)
        self.inv_pers_mat = cv2.getPerspectiveTransform(dest, src)

        return self.pers_mat, self.inv_pers_mat

    def __get_default_roi(self):
        '''
        Method to get the default ROI. Must be set based on the video
        :return: the default ROI
        '''
        src = np.float32([
            (550, 470),
            (770, 470),
            (1090, 640),
            (270, 640)
        ])
        dest = np.float32([
            (0, 0),
            (1280, 0),
            (1280 * 1.0, 720),
            (0, 720)
        ])

        return src, dest

    def __get_current_roi(self):
        '''
        Method to get the current ROI
        :return: the source and desination points
        '''
        if self.current_roi_src == []:
            return self.__get_default_roi()

        return self.current_roi_src, self.current_roi_dest

    def __set_current_roi(self, src, dest):
        '''
        Method to set the current ROI
        :param src: the source points
        :param dest: the destination points
        :return: None
        '''
        self.current_roi_src, self.current_roi_dest = src, dest

    def __reset_current_roi(self):
        '''
        Method to reset the ROI points to the default values
        :return: None
        '''
        self.current_roi_src, self.current_roi_dest = self.__get_default_roi()

    def __predict_next_roi(self, conf):
        '''
        Method to set the next ROI based on the confidence of fits
        :param conf: the confidence value
        :return: the source and destination points
        '''
        conf = conf / (self.WINDOW_SLICE_COUNT + 1)
        if conf < self.ROI_MIN_CONFIDENCE:
            # print(conf)
            src, dest = self.__get_current_roi()
            src[0][0] = src[0][0] - (self.ROI_CHANGE)
            src[3][0] = src[3][0] - (self.ROI_CHANGE)

            src[1][0] = src[1][0] + (self.ROI_CHANGE)
            src[2][0] = src[2][0] + (self.ROI_CHANGE)

            self.__set_current_roi(src, dest)

        return self.__get_current_roi()

    def __perspective_transform(self, img, mat):
        '''
        Method to transform the image
        :param img: the image to be transformed
        :param mat: the perspective matrix
        :return: the transformed image
        '''
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, mat, img_size, cv2.INTER_LINEAR)

        return warped

    def __get_peaks(self, img):
        '''
        Method to get the left and right peak points
        :param img: the image to find peak points
        :return: left and right peak points
        '''
        histogram = np.sum(img[img.shape[0] / 2:, :], axis=0)

        mid = np.int(histogram.shape[0] / 2)
        left_peak = np.argmax(histogram[:mid])
        right_peak = np.argmax(histogram[mid:]) + mid

        return left_peak, right_peak

    def __get_non_zeros(self, img):
        '''
        Method to get the nonzero points of an image
        :param img: the input image
        :return: the x and y nonzero points
        '''

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        return nonzerox, nonzeroy

    def __get_conf_fits(self, left_fit, left_conf, right_fit, right_conf):
        '''
        Method to smooth the fits detected based on the average and confidence values from the previous fits.
        :param left_fit: the current left fit
        :param left_conf: the confidence of left fit
        :param right_fit: the current right fit
        :param right_conf: the confidence of right fit
        :return: the smooth left and right fits
        '''

        # reset the fits if the frame counter has reached
        if self.processed_frames % self.AVERAGE_FRAME_COUNT == 0:
            self.left_fit_list = []
            self.right_fit_list = []
            self.left_fit_conf_list = []
            self.right_fit_conf_list = []
            self.prev_conf_left_fit = None
            self.prev_conf_right_fit = None

        # append the current fit if it meets the minimum threshold
        if (len(self.left_fit_conf_list) == 0):
            if left_conf > 0:
                self.left_fit_list.append(left_fit)
                self.left_fit_conf_list.append(left_conf)
        if (len(self.right_fit_conf_list) == 0):
            if right_conf > 0:
                self.right_fit_list.append(right_fit)
                self.right_fit_conf_list.append(right_conf)

        # return the average of the fits based on the confidence level
        return np.average(self.left_fit_list, weights=self.left_fit_conf_list, axis=0), np.average(self.right_fit_list,
                                                                                                   weights=self.right_fit_conf_list,
                                                                                                   axis=0)

    def __get_fits(self, leftx, lefty, left_conf, rightx, righty, right_conf):
        '''
        Method to get the polyfits based on the x and y values
        :param leftx: the x values of the left lane
        :param lefty: the y values of the left lane
        :param left_conf: the confidence value of detection
        :param rightx: the x values of the right lane
        :param righty: the y values of the right lane
        :param right_conf: the confidence value of detection
        :return:
        '''
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.conf_left_fit = left_fit
        self.conf_right_fit = right_fit

        return self.__get_conf_fits(left_fit, left_conf, right_fit, right_conf)

    def __full_detect_indices(self, img):
        '''
        Method to detect the lane lines from the image
        :param img: the input image
        :return: the lane points and the confidence value
        '''

        # get the peak values and the nonzero values of the pixels in the image
        leftx_current, rightx_current = self.__get_peaks(img)
        nonzerox, nonzeroy = self.__get_non_zeros(img)

        # initialize the window and the confidence values
        window_height = np.int(img.shape[0] / self.WINDOW_SLICE_COUNT)
        left_lane_inds = []
        right_lane_inds = []
        left_conf = 0
        right_conf = 0

        # for each of the window
        for window in range(self.WINDOW_SLICE_COUNT):

            # get the average lane distance for the particular window based on the previous averages.
            # This is helpful when a lane is not detected confidently
            if window in self.lane_distances:
                avg_px_dist = np.average(self.lane_distances[window], weights=self.lane_weights[window])
            else:
                avg_px_dist = -1

            if self.ENABLE_LOGS:
                print('average dist is {}'.format(avg_px_dist))

            # calculate the window points from the current position
            left_found = False
            right_found = False
            window_y_low = img.shape[0] - ((window + 1) * window_height)
            window_y_high = img.shape[0] - (window * window_height)
            window_xleft_low = leftx_current - self.WINDOW_MARGIN
            window_xleft_high = leftx_current + self.WINDOW_MARGIN
            window_xright_low = rightx_current - self.WINDOW_MARGIN
            window_xright_high = rightx_current + self.WINDOW_MARGIN

            # get all the left and right indices with nonzero values and append it to a growing list
            good_left_inds = (nonzeroy >= window_y_low) & (nonzeroy < window_y_high) & (
                nonzerox >= window_xleft_low) & (nonzerox < window_xleft_high)
            good_left_inds = good_left_inds.nonzero()[0]
            good_right_inds = (nonzeroy >= window_y_low) & (nonzeroy < window_y_high) & (
                nonzerox >= window_xright_low) & (nonzerox < window_xright_high)
            good_right_inds = good_right_inds.nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # if the number of pixels in the window meets the threshold, reposition the window over the peaks
            if len(good_left_inds) > self.LANE_MIN_PIX:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                left_found = True
                left_conf += 1
            if len(good_right_inds) > self.LANE_MIN_PIX:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                right_found = True
                right_conf += 1

            # if the window did not have minimum pixels, approximate the new lane lines based on the average value
            if not left_found:
                if right_found and avg_px_dist > 0:
                    leftx_current = np.int(rightx_current - avg_px_dist)
            if not right_found:
                if left_found and avg_px_dist > 0:
                    rightx_current = np.int(leftx_current + avg_px_dist)

            # if both the lanes were found, append the new distance value to the list
            if left_found and right_found:
                if window in self.lane_distances:
                    self.lane_distances[window].append(rightx_current - leftx_current)
                    self.lane_weights[window].append(
                        self.ACTUAL_LANE_WEIGHT * ((right_conf + left_conf) / (window + 1)))
                else:
                    self.lane_distances[window] = [rightx_current - leftx_current]
                    self.lane_weights[window] = [self.ACTUAL_LANE_WEIGHT]

        # calculate the confidence values
        right_conf = (right_conf + 1 / self.WINDOW_SLICE_COUNT + 1)
        left_conf = (left_conf + 1 / self.WINDOW_SLICE_COUNT + 1)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # get the left and right points from the full list
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # predict the next ROI based on the confidence values
        self.__predict_next_roi((right_conf + left_conf) / 2)
        self.lane_detected = True

        # return the output
        return leftx, lefty, left_conf, rightx, righty, right_conf

    def __draw_overlay(self, undistorted, warped, left_fit, right_fit, mat):
        '''
        Method to draw an overlay to indicate lane lines in the original image
        :param undistorted: the undistorted image
        :param warped: the transformed image
        :param left_fit: the left fit
        :param right_fit: the right fit
        :param mat: the perspective matrix
        :return: the new image with lane lines drawn
        '''

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # if the fit is not empty draw the lane lines and fill the area
        if left_fit != None and right_fit != None:
            ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
            left_fitx = (left_fit[0] * (ploty ** 2)) + (left_fit[1] * ploty) + (left_fit[2])
            right_fitx = (right_fit[0] * (ploty ** 2)) + (right_fit[1] * ploty) + (right_fit[2])
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

            self.prev_overlay = self.__perspective_transform(color_warp, mat)

        # Combine the result with the original image
        if self.prev_overlay != None:
            return cv2.addWeighted(undistorted, 1, self.prev_overlay, 0.3, 0)
        else:
            return undistorted

    def __get_radius(self, leftx, lefty, rightx, righty):
        '''
        Method to get the radius of curvature in meters
        :param leftx: the x points of the left lane
        :param lefty: the y points of the left lane
        :param rightx: the x points of the right lane
        :param righty: the y points of the right lane
        :return: the radius of curvature in meters
        '''
        ym_per_pix = 30 / 720  # 30 meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # 3.7 meters per pixel in x dimension

        # evaluate at the points near the car
        y_left_eval = np.max(lefty)
        y_right_eval = np.max(lefty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radius
        left_rad = ((1 + (2 * left_fit_cr[0] * y_left_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_rad = ((1 + (
            2 * right_fit_cr[0] * y_right_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        # return the left and right radius
        return np.round(left_rad, 2), np.round(right_rad, 2)

    def __write_radius(self, img, radius):
        '''
        Method to write the radius of curvature
        :param img: the input image
        :param radius: the radius of curvature
        :return: the processed image
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        if radius is not None:
            cv2.putText(img, 'Radius {}'.format(radius), (100, 100),
                        font, 0.5, (255, 255, 255), 2)

        return img

    def __write_offset(self, img, left_fit, right_fit):
        '''
        Method to calculate and write the offset from the center
        :param img: the input image
        :param left_fit: the left fit
        :param right_fit: the right fit
        :return: the processed image
        '''

        if left_fit!=[] and right_fit!=[]:

            # calculate the actual center by getting the x values at the bottom
            leftx = (left_fit[0] * (self.image_height ** 2)) + (left_fit[1] * self.image_height) + (left_fit[2])
            rightx = (left_fit[0] * (self.image_height ** 2)) + (right_fit[1] * self.image_height) + (right_fit[2])
            self.offset_list.append(np.absolute((leftx - rightx) / 2))
            actual_center = np.average(self.offset_list)

            # caluclate offset from original in meters
            ideal_center = self.image_width / 2
            offset = np.round(np.absolute((actual_center - ideal_center) / 2) * 3.7 / 700, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            if offset is not None:
                cv2.putText(img, 'Offset {}'.format(offset), (int(img.shape[1] ) - 100, 100),
                            font, 0.5, (255, 255, 255), 2)

        if self.processed_frames % self.AVERAGE_FRAME_COUNT ==0 :
            self.offset_list = []

        return img

    def run(self, img, convert_to_rgb=True):
        '''
        Method to detect lanes on the image. Runs the image through the pipeline and draws information on the processed image.
        :param img: image to detect lane lines on
        :param convert_to_rgb: boolean to indicate if conversion to rgb is required
        :return: the processed image with lane lines (hopefully)
        '''

        # increment the processed counter
        self.processed_frames += 1

        if convert_to_rgb:
            img = utils.convert_bgr_rgb(img)

        self.image_width = img.shape[1]
        self.image_height = img.shape[0]

        if self.ENABLE_LOGS:
            cv2.imwrite('./test/def_' + str(self.processed_frames) + '.jpg', img)

        # undistort the image using the calibrated values
        undistorted = self.__undistort(img, self.mtx, self.dist)

        # pass the distortion corrected image through the threshold pipeline
        preprocessed = self.__prepocess(undistorted)

        if self.ENABLE_LOGS:
            cv2.imwrite('./debug/preprocessed_' + str(self.processed_frames) + '.jpg', img)

        # get the current region of interest and load the perspective and inverse perspective matrix
        src, dest = self.__get_current_roi()
        pers_mat, inv_mat = self.__load_perspective_matrix(src, dest)

        # transform the thresholded image to a bird's eye view
        warped = self.__perspective_transform(preprocessed, pers_mat)

        # get the points and confidence values during lane detection
        leftx, lefty, left_conf, rightx, righty, right_conf = self.__full_detect_indices(warped)

        # if lanes lines were detected use the value to get the fits
        if leftx != [] and rightx != []:
            left_fit, right_fit = self.__get_fits(leftx, lefty, left_conf, rightx, righty, right_conf)

        # else use the previous confident fits to draw the lane lines
        else:
            left_fit, right_fit = self.conf_left_fit, self.conf_right_fit

        # calculate the radius if this is the first time we are calculating it
        if self.prev_left_rad is None:
            self.prev_left_rad, self.prev_right_rad = self.__get_radius(leftx, lefty, rightx, righty)

        # if method has processed the average frames for smoothing the fits, reset the roi values and also re-calculate the radius of curvature
        if self.processed_frames % self.AVERAGE_FRAME_COUNT == 0:
            self.__reset_current_roi()
            self.prev_left_rad, self.prev_right_rad = self.__get_radius(leftx, lefty, rightx, righty)

        # draw the warped polyfill and write the information
        overlay = self.__draw_overlay(undistorted, warped, left_fit, right_fit, inv_mat)
        overlay = self.__write_radius(overlay, np.round((self.prev_left_rad + self.prev_right_rad) / 2, 2))
        overlay = self.__write_offset(overlay, left_fit, right_fit)

        if self.ENABLE_LOGS:
            cv2.imwrite('./test/overlay_' + str(self.processed_frames) + '.jpg', overlay)

        # return the processed image
        return utils.convert_bgr_rgb(overlay)
