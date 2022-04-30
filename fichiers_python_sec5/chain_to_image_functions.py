'''
chain_to_image_functions.py
Uses the package https://github.com/galtay/hilbertcurve

2020 Hugo Gangloff
'''
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib.pyplot as plt

def get_hilbertcurve_path(image_border_length):
    '''
    Given image_border_length, the length of a border of a square image, we
    compute path of the hilbert peano curve going through this image

    Note that the border length must be a power of 2.

    Returns a list of the coordinates of the pixel that must be visited (in
    order !)
    '''
    path = []
    p = int(np.log2(image_border_length))
    hilbert_curve = HilbertCurve(p, 2)
    path = []
    print("Compute path for shape ({0},{1})".format(image_border_length,
        image_border_length))
    for i in range(image_border_length ** 2):
        #coords = hilbert_curve.coordinates_from_distance(i)
        coords = hilbert_curve.point_from_distance(i)
        path.append([coords[0], coords[1]])

    return path

def chain_to_image(X_ch):
    '''
    X_ch is an unidimensional array (a chain !) whose length is 2^(2*N) with N non negative
    integer.
    We transform X_ch to a 2^N * 2^N image following the hilbert peano curve
    '''
    image_border_length = int(np.sqrt(len(X_ch)))
    path = get_hilbertcurve_path(image_border_length)

    X_img = np.empty((image_border_length, image_border_length))
    for idx, coords in enumerate(path):
        X_img[coords[0], coords[1]] = X_ch[idx]

    return X_img

def image_to_chain(X_img):
    '''
    X_img is a 2^N * 2^N image with N non negative integer.
    We transform X_img to a 2^(2*N) unidimensional vector (a chain !)
    following the hilbert peano curve
    '''
    path = get_hilbertcurve_path(X_img.shape[0])

    X_ch = []
    for idx, coords in enumerate(path):
        X_ch.append(X_img[coords[0], coords[1]])

    return np.array(X_ch)

if __name__ == '__main__':
    # Test
    s = 128
    X_chain = np.random.randn(s * s)
    X_image = chain_to_image(X_chain)
    X_chain_back = image_to_chain(X_image)
    # check that the transformation and its reverse give the same vector:
    assert np.count_nonzero(X_chain != X_chain) == 0

    #plot hilbert peano path


    path = get_hilbertcurve_path(128)
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1])
    plt.show()
