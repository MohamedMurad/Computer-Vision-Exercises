#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt


# Stereo Matching function as per the given paper

def stereoMatching(leftImg, rightImg):

    # image shape

    (rows, cols) = leftImg.shape

    # matrices to store disparities: left and right

    left_disp = np.zeros((rows, cols))
    right_disp = np.zeros((rows, cols))

    # some constants

    sig = 2
    c0 = 5

    # Pick a row in the image to be matched

    for c in range(0, rows):

        # Disparity path matrix

        disp_matrix = np.zeros((cols, cols))

        # Cost matrix

        cost_matrix = np.zeros((cols, cols))

        # Initialize the cost matrix

        for i in range(0, cols):
            cost_matrix[i][0] = i * c0
            cost_matrix[0][i] = i * c0

        # Iterate the row in both the images to find the path using dynamic programming

        for k in range(0, cols):
            for j in range(0, cols):

                # calculate matcing cost

                match_cost = ((leftImg[c][k] - rightImg[c][j]) ** 2 )/ (sig ** 2)

                # Finding minimum cost

                min1 = cost_matrix[k - 1][j - 1] + match_cost
                min2 = cost_matrix[k - 1][j] + c0
                min3 = cost_matrix[k][j - 1] + c0

                cost_matrix[k][j] = min(min1, min2, min3)

                # marking the path

                if cost_matrix[k][j] == min1:
                    disp_matrix[k][j] = 1
                if cost_matrix[k][j] == min2:
                    disp_matrix[k][j] = 2
                if cost_matrix[k][j] == min3:
                    disp_matrix[k][j] = 3

        # backtracking and update the disparity value

        i = cols - 1
        j = cols - 1

        while i != 0 and j != 0:
            if disp_matrix[i][j] == 1:
                left_disp[c][i] = np.absolute(i - j)
                right_disp[c][j] = np.absolute(j - i)
                i = i - 1
                j = j - 1
            elif disp_matrix[i][j] == 2:
                left_disp[c][i] = 0
                i = i - 1
            elif disp_matrix[i][j] == 3:
                right_disp[c][j] = 0
                j = j - 1

    return left_disp, right_disp


def main():

    # determine path to read images

    path = 'samples/sample2/'

    # Read images.

    leftImg = cv2.imread(path + 'img1.png', 0)
    leftImg = np.asarray(leftImg, dtype=np.float)
    rightImg = cv2.imread(path + 'img2.png', 0)
    rightImg = np.asarray(rightImg, dtype=np.float)

    # Call disparity matching algorithm

    left_disp, right_disp = stereoMatching(leftImg, rightImg)

    # save images

    cv2.imwrite(path + 'left_disparity' + '.png', left_disp)
    cv2.imwrite(path + 'right_disparity' + '.png', right_disp)


main()