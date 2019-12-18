"""
Stereo calibration for Leopard OV580-OV7251
"""
import pickle
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calibrate_camera(corner3d, corner_pix, shape, *args):
    """
    Wrapper of OpenCV calibrateCamera, returning CameraParameters object.
    Extra args:
    [cameraMatrix[, distCoeffs[, rvecs[, tvecs[, flags[, criteria]]]]]]
    """
    rep_err, K, dc, rv, tv = cv2.calibrateCamera(
        corner3d, 
        corner_pix, 
        shape,
        None,
        None, 
        *args
    )
    return CameraParameters(K, dc, rv, tv, rep_err)


class CameraParameters:

    def __init__(self, K, dc, rv=None, tv=None, repr_err=None):
        self.K = K # Camera matrix
        self.dc = dc # Distortion parameters
        self.rv = rv # Rotation vectors, output of camera calibration
        self.tv = tv # Translation vectors, output of camera calibration
        self.reprojection_error = repr_err

    def __str__(self):
        ret1 = 'K: {}\n'.format(self.K)
        ret2 = 'Distortion coefs [k1,k2,p1,p2,k3]: {}\n'.format(self.dc)
        ret3 = 'Intrinsic error: {}\n'.format(self.reprojection_error)
        return ret1 + ret2 + ret3

    def __repr__(self):
        ret1 = 'K: {}\n'.format(repr(self.K))
        ret2 = 'Distortion coefs [k1,k2,p1,p2,k3]: {}\n'.format(repr(self.dc))
        ret3 = 'Intrinsic error: {}\n'.format(repr(self.reprojection_error))
        return ret1 + ret2 + ret3

    def reproject_corners(self, rows, cols, corner3d, corner_pix):
        """
        Reprojects 3D checkerboard positions to pixel coordinates.

        Returns two IxJx2 arrays, where I is number of image pairs, J is the
        number of checkerboard corners.
        """
        num_images = len(corner3d)
        num_corners = rows*cols
        pix_measured = np.zeros((num_images, num_corners, 2))
        pix_reprojected = np.zeros((num_images, num_corners, 2))

        for i in range(num_images):
            project, _ = cv2.projectPoints(
                corner3d[i],
                self.rv[i],
                self.tv[i],
                self.K,
                self.dc
            )
            pix_reprojected[i, :, :] = np.squeeze(project)
            pix_measured[i, :, :] = np.squeeze(corner_pix[i])
        return pix_reprojected, pix_measured


class LeopardStereoCal:

    def __init__(self, directory=None, rows=5, cols=5, ext='.pgm', size=5, **kwargs):
        self.dir_path = directory
        self.rows = rows
        self.cols = cols
        # All images found in directory
        self.all_imgs = [i for i in os.listdir(directory) if i.endswith(ext)]
        # Image names used in intrinsic calibration
        self.cal_imgs = []
        # Indices of all images for which corners are found.
        self.cal_inds_all = []
        # Cal inds to remove
        self.removed_inds = []
        if len(self.all_imgs):
            print(
                'Found {} images:\n'.format(len(self.all_imgs)), 
                self.all_imgs
            )
        # Length of checkerboard square in mm.
        self.size = size
        # Shape of single image view
        self.view_shape = ((480, 640))
        
        # Arrays to store object points and image points from all images.
        self.objpoints = [] # 3d points in checkerboard space
        self.points1 = [] # pixel coords of corners in img1
        self.points2 = [] # pixel coords of corners in img2

        self.params1 = None
        self.params2 = None
        self.stereo_error = None
        self.R = None
        self.t = None
        self.stereo_reprojection_errors = None

        self.intrinsics_recalibrated = False

    def __str__(self):
        ret1 = 'Camera parameters1:\n {}\n'.format(self.params1)
        ret2 = 'Camera parameters2:\n {}\n'.format(self.params2)
        ret3 = 'Stereo repr error: {}\n'.format(self.stereo_error)
        ret4 = 'R: {}\n'.format(self.R)
        ret5 = 't: {}\n'.format(self.t)
        return ret1 + ret2 + ret3 + ret4 + ret5

    def __repr__(self):
        ret1 = 'Camera parameters1:\n {}\n'.format(repr(self.params1))
        ret2 = 'Camera parameters2:\n {}\n'.format(repr(self.params2))
        ret3 = 'Stereo repr error: {}\n'.format(repr(self.stereo_error))
        ret4 = 'R: {}\n'.format(repr(self.R))
        ret5 = 't: {}\n'.format(repr(self.t))
        return ret1 + ret2 + ret3 + ret4 + ret5

    @property
    def cal_inds(self):
        """
        Indices of images in which checkerboard images were found, and also 
        have not been manually removed as outliers.
        """
        return np.setdiff1d(self.cal_inds_all, self.removed_inds)

    @property
    def images_used(self):
        """
        Subset of images used for stereo calibration.
        """
        cal_imgs = np.array(self.cal_imgs)
        return cal_imgs[self.cal_inds]

    @property
    def P1(self):
        """
        Camera projection matrix for view 1.
        """
        K1 = self.params1.K
        return np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))

    @property
    def P2(self):
        """
        Camera projection matrix for view 2.
        """
        K2 = self.params2.K
        R = self.R
        t = self.t
        return np.dot(K2, np.hstack((R, t.reshape(3,1))))

    def view_calibrate(self):
        """
        Calibrate intrinsic parameters for each view.
        """
        # Prepare corner points: (0,0,0), (1,0,0) etc.
        objp = np.zeros((self.rows*self.cols, 3), np.float32)
        objp[:, :2] = self.size*np.mgrid[0:self.rows, 
                                         0:self.cols].T.reshape(-1,2)

        for img in self.all_imgs:
            print('Processing ...', img)
            imgpath = os.path.join(self.dir_path, img)
            mat = cv2.imread(imgpath, 0) # grayscale
            mat1 = mat[:, 640:]
            mat2 = mat[:, :640]

            # Find corners
            flag1, corners1 = self.find_corners(mat1)
            flag2, corners2 = self.find_corners(mat2)

            # Process only if corners found for both views.
            if (flag1 and flag2):
                self.cal_imgs.append(img)
                self.objpoints.append(objp)
                self.points1.append(corners1)
                self.points2.append(corners2)

        self.objpoints = np.array(self.objpoints)
        self.points1 = np.array(self.points1)
        self.points2 = np.array(self.points2)
        self.cal_inds_all = [i for i in range(len(self.cal_imgs))]
        print('Calculating intrinsics:')
        self.params1 = calibrate_camera(self.objpoints, self.points1,
            self.view_shape)
        self.params2 = calibrate_camera(self.objpoints, self.points2,
            self.view_shape)
        print('Reprojection error camera 1 cal:', 
              self.params1.reprojection_error)
        print('Reprojection error camera 2 cal:', 
              self.params2.reprojection_error)

    def view_recalibrate(self):
        """
        Re-calibrate intrinsic parameters with exclusions.
        """
        inds = self.cal_inds
        print('Recalculating intrinsics:')
        self.params1 = calibrate_camera(
            self.objpoints[inds], 
            self.points1[inds],
            self.view_shape
        )
        self.params2 = calibrate_camera(
            self.objpoints[inds], 
            self.points2[inds],
            self.view_shape
        )
        print('Reprojection error camera 1 cal:', 
              self.params1.reprojection_error)
        print('Reprojection error camera 2 cal:', 
              self.params2.reprojection_error)
        
        # Set this to true for calculating stereo reprojection error.
        self.intrinsics_recalibrated = True

    def _calibrate(self, recalibrate_intrinsics=True):
        if (self.params1 is None) or (self.params2 is None):
            self.view_calibrate()
        elif recalibrate_intrinsics:
            self.view_recalibrate()

        print('Calibrating stereo:')
        stereo_error, R, t = self._stereo_calibrate()
        print('Stereo error:', stereo_error)
        self.stereo_error = stereo_error
        self.R = R
        self.t = t
        self.calculate_stereo_reprojection_errors()

    def calibrate(self, output_path=None, output_corners=None, 
        show_corners=False, output_error_plot=None, 
        recalibrate_intrinsics=True):
        """
        Stereo calibration with command line input.
        """
        while(True):
            self._calibrate(recalibrate_intrinsics)
            self.draw_checker_corners(output_corners, show_corners)
            self.show_errors(output_error_plot)
            prompt = 'Enter indices to remove separated by comma:\n'

            try:
                command = input(prompt)
                command = command.split(',')
                remove_inds = [int(_.strip()) for _ in command]
                self.removed_inds.extend(remove_inds)
            except ValueError:
                print('Exiting.')
                if output_path is not None:
                    print('Saving calibration to', output_path)
                    self.save(output_path)
                break

    def draw_checker_corners(self, output_dir=None, show=False):
        """
        Draws corners on images used for intrinsic calibrations.
        """
        inds = self.cal_inds
        if output_dir or show:
            for i in inds:
                img = self.cal_imgs[i]
                imgpath = os.path.join(self.dir_path, img)
                mat = cv2.imread(imgpath)
                mat1 = mat[:, 640:]
                mat2 = mat[:, :640]

                cv2.drawChessboardCorners(
                    mat1,
                    (self.rows, self.cols),
                    self.points1[i],
                    1
                )
                cv2.drawChessboardCorners(
                    mat2,
                    (self.rows, self.cols),
                    self.points2[i],
                    1
                )
                mat_corners = np.hstack((mat2, mat1))
                if show:
                    winlabel = '{} index: {}'.format(img, i)
                    cv2.imshow(winlabel, mat_corners)
                    cv2.waitKey(0)
                if output_dir:
                    output_path = os.path.join(
                        output_dir, 
                        'corners_{}.png'.format(img)
                    )
            cv2.destroyAllWindows()

    def _stereo_calibrate(self):
        """
        Wrapper for cv2.stereoCalibrate()
        """
        inds = self.cal_inds
        stereo_error, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(
            self.objpoints[inds],
            self.points1[inds],
            self.points2[inds],
            self.params1.K,
            self.params1.dc,
            self.params2.K, 
            self.params2.dc,
            self.view_shape)
        #     cv2.CALIB_FIX_INTRINSIC
        # )
        return stereo_error, R, t

    def calculate_stereo_reprojection_errors(self):
        """
        Calculate stereo reprojection error for each image pair.
        """
        rms1 = np.zeros(len(self.cal_imgs))
        rms2 = np.zeros(len(self.cal_imgs))

        if self.intrinsics_recalibrated:
            inds = np.arange(len(self.cal_inds), dtype=int)
        else:
            inds = self.cal_inds

        for i in inds:
            # Convert rotation vec1 to matrix
            rM1, _ = cv2.Rodrigues(self.params1.rv[i])
            rM2_new = np.dot(self.R, rM1)
            # Convert transformed rotation mat2 to vector
            rv2_new, _ = cv2.Rodrigues(rM2_new)
            tv2_new = self.t + np.dot(self.R, self.params1.tv[i])
            reprojections1, _ = cv2.projectPoints(
                self.objpoints[i],
                self.params1.rv[i],
                self.params1.tv[i],
                self.params1.K,
                self.params1.dc
            )
            reprojections2, _ = cv2.projectPoints(
                self.objpoints[i],
                rv2_new,
                tv2_new,
                self.params2.K,
                self.params2.dc
            )
            errors1 = np.squeeze(self.points1[i] - reprojections1)
            errors2 = np.squeeze(self.points2[i] - reprojections2)
            rms1[i] = np.mean(np.sqrt(errors1[:, 0]**2 + errors1[:, 1]**2))
            rms2[i] = np.mean(np.sqrt(errors2[:, 0]**2 + errors2[:, 1]**2))

        df = pd.DataFrame(np.array([rms1[inds], rms2[inds]]).T, columns=['V1', 'V2'],
                          index=self.cal_inds)
        self.stereo_reprojection_errors = df

    def show_errors(self, output_path=None):
        """
        Show barplot of stereo reprojection errors.
        """
        self.stereo_reprojection_errors.plot.bar()
        plt.ylabel('RMS reprojection error (pixels)')
        plt.title('Stereo calibration RMS reprojection error')
        if output_path is not None:
            plt.savefig(output_path)
        plt.show()
        plt.close()

    def find_corners(self, img):
        """
        Find checkerboard corners with subpixel accuracy.
        If corners not found, returns flag = 0.
        """
        # Termination criteria for getting sub-pixel corner positions.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                    30, 0.001)
        flag, corners = cv2.findChessboardCorners(
            img,
            (self.rows, self.cols),
            None
        )
        if flag:
            corners = cv2.cornerSubPix(img, corners, (5,5), (-1, 1), 
                criteria)
        return flag, corners

    def save(self, cal_file):
        """
        Pickle calibration object.
        """
        with open(cal_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, cal_file):
        """
        Load saved calibration object.
        """
        with open(cal_file, 'rb') as f:
            tmp_dict = pickle.load(f)

        self.__dict__.clear()
        self.__dict__.update(tmp_dict)



if __name__ == '__main__':
    onedrive = os.environ['OneDrive']
    testdir = os.path.join(onedrive, 'Phd/Projects/dense_stereo_reconstruction/stereo_cal/tattoo_cal_2409_4_0')
    rows = 11
    cols = 17
    cal = LeopardStereoCal(testdir, 11, 17)
    cal.calibrate(output_path='stereo_cal')
