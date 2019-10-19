import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import cv2


# Manually identifies corresponding points from two views
def getCorrespondence(image1, image2, numOfPoints_=8, manualSelection_=True):
    manualSelection = manualSelection_;
    numOfPoints = numOfPoints_;

    if(numOfPoints<8):
        print("Error: Num of paris must be greater or equal 4")
        return

    if(manualSelection):
        # Display images, select matching points
        fig = plt.figure()
        fig1 = fig.add_subplot(1,2,1)
        fig2 = fig.add_subplot(1,2,2)
        # Display the image
        fig1.imshow(image1)
        fig2.imshow(image2)
        plt.axis('image')
        pts = plt.ginput(n=numOfPoints, timeout=0)
        pts = np.reshape(pts, (2, int(numOfPoints/2), 2))
        # print(pts);
        return pts
    else:
        """
        automatic selection process
        """
        numOfPoints=20

        img1 = cv2.imread('../images/mountain/image1.jpg',0)
        img2 = cv2.imread('../images/mountain/image2.jpg',0)

        # Initiate SIFT detector
        orb = cv2.ORB()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        print(matches)
        return matches


    # 10 to 10 mountain points 
    pts = np.array([(290.99673461310101, 168.49247971502086), (351.77211557490591, 132.47743914506236), (381.03433603799715, 139.23025925192962), (455.31535721353646, 161.7396596081536), (471.07193746289317, 148.23401939441919), (511.58885810409652, 175.24529982188801), (507.08697803285168, 359.82238274292507), (403.54373639422113, 319.30546210172179), (390.03809618048672, 341.81486245794582), (290.99673461310101, 359.82238274292507), (5.9867999208391893, 139.23025925192951), (80.267821096378498, 109.96803878883827), (107.27910152384732, 121.22273896695026), (177.05824262814178, 161.73965960815349), (195.06576291312103, 148.23401939441908), (224.32798337621227, 175.2452998218879), (208.57140312685544, 353.06956263605764), (98.275341381357748, 303.54888185236484), (82.518761132000918, 326.05828220858882), (3.7358598852168825, 353.06956263605764)])
    pts = np.reshape(pts, (2, int(numOfPoints/2), 2))
    return pts

# Return x coordinate of point index from the selected points in image 1
def getStartX(pts, index):
    return pts[0][index][0]

# Return y coordinate of point index from the selected points in image 1
def getStartY(pts, index):
    return pts[0][index][1]

# Return x coordinate of matched point index from image 2 
def getEndX(pts, index):
    return pts[1][index][0]

# Return y coordinate of matched point index from image 2
def getEndY(pts, index):
    return pts[1][index][1]

# Compute sub matrix of A for a certain point and its matched one
def getSubAMatrix(x, y, x_, y_):
    subMatrix = np.zeros((2,8))
    subMatrix[0] = np.array([x, y,1,0,0,0, -x*x_, -y*x_])
    subMatrix[1] = np.array([0,0,0,x, y,1, -x*y_, -y*y_])
    return subMatrix


# compute x_ from x point using transform h
def transform(point, h):
    # reshape h to matrix 3x3
    h_2d = np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            if 3*i+j < 8:
                h_2d[i][j] = h[3*i+j]
    h_2d[2][2] = 1
    
    # compute x_
    ret = np.dot(h_2d, point)

    # normalize by dividing by w
    ret[0] = ret[0]/ret[2]
    ret[1] = ret[1]/ret[2]
    ret[2] = 1
    return ret

# compute x from x_ point using transform h
def invTransform(point, h):
    # reshape h to matrix 3x3
    h_2d = np.zeros((3,3))
    for i in range(0,3):
        for j in range(0,3):
            if 3*i+j < 8:
                h_2d[i][j] = h[3*i+j]
    h_2d[2][2] = 1

    # compute x
    ret = np.dot(np.linalg.inv(h_2d), point)

    # normalize by dividing by w
    ret[0] = ret[0]/ret[2]
    ret[1] = ret[1]/ret[2]
    ret[2] = 1
    return ret

# calcuate H matrix using n matched points from get_correspondence
def calcH(pts):
    # SOLVE A h = b
    # find number of points
    n = int(pts.size/4)
    
    # build A
    A = np.zeros((2*n, 8))
    for i in range(0, n):
        subMatrix = getSubAMatrix(getStartX(pts, i), getStartY(pts, i), getEndX(pts, i), getEndY(pts, i))
        A[2*i] = subMatrix[0]
        A[2*i + 1] = subMatrix[1]
    
    # build b
    b = np.zeros((2*n, 1))
    for i in range(0, n):
        b[2*i] = getEndX(pts, i)
        b[2*i+1] = getEndY(pts, i)

    #solve equation
    h = np.linalg.lstsq(A, b)[0]

    # for i in range(0,n):
    #     iniPoint = np.array([[getStartX(pts, i)], [getStartY(pts, i)], [1]])
    #     print(' point ' + str(i) + '\n\t'+ str(np.array([[getEndX(pts, i)], [getEndY(pts, i)], [1]])) + '\n\t' + str(transform(iniPoint, h)))
    print('shapes for A, h, b, reconstructed ' + str(A.shape) + str(h.shape) + str(b.shape))
    return h


def wrapAndMergeImage(sourceImage , h, refImage):
    # here we return new sourceImage with (2*width,2*height), 
    # and transform our image into destination .. with interpolating
    height = sourceImage.shape[0]
    width = sourceImage.shape[1]

    minMappedI = minMappedJ = int(100000)
    maxMappedI = maxMappedJ = int(-100000)

    # calculate corners of transformed image
    print('Image A size ' + str(sourceImage.shape))

    # calcuate transformed corners positions
    corners = np.array([[0,0],[height-1, 0],[0, width-1],[height-1, width-1]]);
    for k in range(0,4):
        i = corners[k][0]
        j = corners[k][1];

        mappedPos = transform(np.array([[j],[i],[1]]), h);
        mappedJ = int(mappedPos[0][0])
        mappedI = int(mappedPos[1][0])

        # update corners
        if mappedI < minMappedI:
            minMappedI = mappedI
        if mappedI > maxMappedI:
            maxMappedI = mappedI
        if mappedJ < minMappedJ:
            minMappedJ = mappedJ
        if mappedJ > maxMappedJ:
            maxMappedJ =mappedJ
    
    newHeight = (maxMappedI-minMappedI+1);
    newWidth = (maxMappedJ-minMappedJ+1);

    shiftHeight = -minMappedI;
    shiftWidth = -minMappedJ;

    print('Bounds ' + str(minMappedI) + ' ' + str(maxMappedI) + ' ' + str(minMappedJ) + ' ' + str(maxMappedJ));
    print('shiftHeight ' + str(shiftHeight));
    print('shiftWidth ' + str(shiftWidth));


    destinationImage = np.zeros((newHeight, newWidth,3), dtype=np.uint8);
    print('Destination image size ' + str(destinationImage.shape))

    # method1: transform each pixels from image2 to image1, 
    # which cause black holes
    for i in range(0,height):
        for j in range(0, width):
            mappedPos = transform(np.array([[j],[i],[1]]), h);
            mappedJ = int(mappedPos[0][0])
            mappedI = int(mappedPos[1][0])
            destinationImage[mappedI+shiftHeight][mappedJ+shiftWidth] = sourceImage[i][j];

    im = Image.fromarray(destinationImage)
    im.save("with_holes.jpg")

    # method2: calculate for each pixel in image1 the correspondence in image2
    # we need to interpolate, it removes the black holes
    for i in range(0, newHeight):
        for j in range(0, newWidth):
            # may be done in more neat way!
            if int(destinationImage[i][j][0]) == 0 and int(destinationImage[i][j][1]) == 0 and int(destinationImage[i][j][2]) == 0:
                # it's black let's get back to it's inverse!
                invMappedPos = invTransform(np.array([[(j - shiftWidth)], [(i - shiftHeight)],[1]]), h)
                invMappedJ = invMappedPos[0][0]
                invMappedI = invMappedPos[1][0]
                if invMappedI <= height-1 and  invMappedI >= 0 and invMappedJ <= width-1 and invMappedJ >= 0:
                    # using bilinear interpolation
                    low_i = int(invMappedI);
                    low_j = int(invMappedJ);
                    dist_i = invMappedI - low_i;
                    dist_j = invMappedJ - low_j;
                    destinationImage[i][j] = \
                    (1-dist_i)*(1-dist_j)*sourceImage[low_i][low_j] + \
                    (1-dist_i)*(dist_j)*sourceImage[low_i][low_j+1] + \
                    (dist_i)*(1-dist_j)*sourceImage[low_i+1][low_j] + \
                    (dist_i)*(dist_j)*sourceImage[low_i+1][low_j+1]
                    # destinationImage[i][j] = sourceImage[int(invMappedI)][int(invMappedJ)]

    im = Image.fromarray(destinationImage)
    im.save("without_holes.jpg")

    # merge the warped image and the reference one
    refImageHeight = refImage.shape[0]
    refImageWidth = refImage.shape[1]
    print('Ref-image size: ' + str(refImage.shape))

    # calculate merged width and height
    mergedImageHeight = refImageHeight + shiftHeight
    if newHeight > mergedImageHeight:
        mergedImageHeight = newHeight

    mergedImageWidth = refImageWidth + shiftWidth
    if newWidth > mergedImageWidth:
        mergedImageWidth = newWidth

    # make a new image of the new width and height
    mergedImage = np.zeros((mergedImageHeight, mergedImageWidth, 3), dtype=np.uint8);

    # sketch the reference image
    for i in range(0, refImageHeight):
        for j in range(0, refImageWidth):
            mergedImage[i + shiftHeight][j + shiftWidth] = refImage[i][j]

    # sketch the destination image (warped image)
    for i in range(0, newHeight):
        for j in range(0, newWidth):
            if not( int(destinationImage[i][j][0]) == 0 and \
                int(destinationImage[i][j][0]) == 0 and \
                int(destinationImage[i][j][2]) == 0 ):
                mergedImage[i][j] = destinationImage[i][j]

    im = Image.fromarray(mergedImage)
    im.save("result.jpg")

    return destinationImage



if __name__ == "__main__":
    # Read images
    # file1 = '../images/mountain/image1.jpg'
    # file2 = '../images/mountain/image2.jpg'
    file1 = '../images/building/image1.jpg'
    file2 = '../images/building/image2.jpg'
    image1 = mpimg.imread(file1)
    image2 = mpimg.imread(file2)
    
    # get correspondences
    pts = getCorrespondence(image1, image2, manualSelection_=False)
    # pts = getCorrespondence(image1, image2, numOfPoints_=8, manualSelection_=True)
    print(pts)

    # calculate h
    # h = calcH(pts)
    # print(h)

    # warp and merge
    # wrapAndMergeImage(image2, h, image1)
