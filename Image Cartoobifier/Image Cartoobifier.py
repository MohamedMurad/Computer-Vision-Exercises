
# coding: utf-8

# # Image Cartoonifier
# ## Applying Image Processing Filters For Image Cartoonifying

# In[1]:


# import needed libraries
import cv2
import matplotlib.pyplot as plt
import glob


# ---
# ##  1. Generating black and white sketch and Noise Reduction Using Median Filter
# 

# In[2]:


# default color is BGR
def plot_image(image,color,title):
    plt.title(title)
    if(color=="GRAY"):
        plt.imshow(image,cmap='gray')
    else:
        plt.imshow(image)
    plt.xticks([])
    plt.yticks([])


# In[3]:


image = cv2.imread('KATE.JPG')


# In[4]:


# plot the original image as RGB
image_RBG = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plot_image(image_RBG,"RBG","Original RGB Image")


# In[5]:


image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plot_image(image_gray,"GRAY","Original RGB Image")


# In[6]:


# smoothing the image using median filter with window size 7
median_image = cv2.medianBlur(image_gray, ksize=7)
plot_image(median_image,"GRAY","Smoothed Grayscale Image")


# ---
# ## 2. Edge Detection Using Laplacian Filter

# In[7]:


laplacian_image = cv2.Laplacian(median_image, ddepth=-1,ksize=5)
plot_image(laplacian_image,"GRAY","Edge Detection")


# #### applying threshold with lower bound 125 and upper bound 255

# In[8]:


ret,thresh = cv2.threshold(laplacian_image,125,255,cv2.THRESH_BINARY_INV)
plot_image(thresh,"GRAY","Image Thresholding")


# ---
# ## 3. Generating a color painting and a cartoon

# In[9]:


# resize the image to half of its original size
resized_image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
plot_image(cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB),"RBG","Resized Image")


# #### applying 7 iterations of bilateral fiter

# In[10]:


diterations = 7
for i in range(iterations):
    resized_image = cv2.bilateralFilter(resized_image, d=9, sigmaColor=9, sigmaSpace=7)
bilateral_image=cv2.resize(resized_image,(image.shape[1],image.shape[0]))
plot_image(cv2.cvtColor(bilateral_image,cv2.COLOR_BGR2RGB),"RGB","Bilateral Filter")


# #### combine the edges withe the bilateral filtered image

# In[11]:


cartoon = cv2.bitwise_and(bilateral_image,cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR))
plot_image(cv2.cvtColor(cartoon,cv2.COLOR_BGR2RGB),"RGB","Final Output")


# ---
# ## 4. Final Function

#  composite all steps in one function
#  1. convert the image to gray
#  2. reduce noise on the image using medianBlur filter
#  3. use the function Laplacian() to detect edges
#  4. put a threshold on the result and invert pixels
#  5. apply 7 iterations of bilateral filtering
#  6. combine the result cartoon effect with the original image using bitwise_and function 
#  and return the final result

# In[12]:


def image_cartoonifier_converter(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    median_image = cv2.medianBlur(image_gray, ksize=7)
    laplacian_image = cv2.Laplacian(median_image, ddepth=-1,ksize=5)
    ret,thresh = cv2.threshold(laplacian_image,125,255,cv2.THRESH_BINARY_INV)
    iterations = 7
    resized_image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    for i in range(iterations):
        resized_image = cv2.bilateralFilter(resized_image, d=9, sigmaColor=9, sigmaSpace=7)
    bilateral_image = cv2.resize(resized_image,(image.shape[1],image.shape[0]))
    cartoon = cv2.bitwise_and(bilateral_image,cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR))        
    return cartoon


# In[25]:


def plot_side_by_side(img_1, img_2, size=(16,16)):
    fig = plt.figure(figsize=(size))
    fig.add_subplot(1, 2, 1)
    plt.axis('off')
    for i, img in enumerate([img_1, img_2]):
        ax = fig.add_subplot(1,2,i+1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')


# In[13]:


images = [cv2.imread(file) for file in glob.glob('test/*')]


# In[15]:


i = 0
for image in images:
    i += 1
    cartoon = image_cartoonifier_converter(image)
    cv2.imwrite("test/results/"+str(i)+".JPG", cartoon)


# In[29]:


for image in images:
    plot_side_by_side(image, image_cartoonifier_converter(image))

