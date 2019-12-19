"""
Tools for compatibility with MATLAB stereo calibration.
"""
import os
import cv2
from scipy import io


def cmkdir(directory):
    """Creates directory if it doesn't exist."""
    if not os.path.isdir(directory):
        os.makedirs(directory)


def split_images(imagedir, ext='.pgm'):
    """
    Splits all images in directory into two new directories view1 and view2.
    """
    view1dir = os.path.join(imagedir, 'view1')
    cmkdir(view1dir)
    view2dir = os.path.join(imagedir, 'view2')
    cmkdir(view2dir)

    images = [img for img in os.listdir(imagedir) if img.endswith(ext)]
    for i, image in enumerate(images):
        print('Splitting', image)
        image_path = os.path.join(imagedir, image)
        img = cv2.imread(image_path, 0)
        view1 = img[:, 640:]
        view2 = img[:, :640]

        view1_outpath = os.path.join(view1dir, '{}.pgm'.format(i))
        cv2.imwrite(view1_outpath, view1)
        view2_outpath = os.path.join(view2dir, '{}.pgm'.format(i))
        cv2.imwrite(view2_outpath, view2)


def read_stereotoolbox(matfile):
    """
    Reads stereo calibration results of a Bouguet camera calibration toolbox
    .mat results file.
    """
    data = io.loadmat(matfile)
    K1 = data['KK_left']
    K2 = data['KK_right']
    dc1 = data['kc_left'].T
    dc2 = data['kc_right'].T
    R = data['R']
    t = data['T']
    return K1, K2, dc1, dc2, R, t


def read_stereoapp(matfile):
    """
    Make sure matlab variables have the proper names!
    """
    data = io.loadmat(matfile)
    K1 = data['K1']
    K2 = data['K2']
    dc1 = data['dc1']
    dc2 = data['dc2']
    R = data['R']
    t = data['t']
    return K1, K2, dc1, dc2, R, t




