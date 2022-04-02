import tensorlayer as tl
import numpy as np
import os
import nibabel as nib
import threading
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
from PIL import Image
import skimage.measure

nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # 
training_data_path = "CT-0"
preserving_ratio = 0.25 # filter out 2d images containing < 25% non-zeros


def getImage(src):
    img_path = src
    # print(img_path)
    img = nib.load(img_path).get_fdata()
    # print(img.shape)
    img_3d_max = np.amax(img)
    img = img / img_3d_max * 255  # Normalization
    # for i in range(img.shape[2]):   # 
    #     img_2d = img[:, :, i]  # take one piece of image
    #    # plt.imshow(img_2d) show image
    #    # plt.pause(0.001)
    #     # filter out 2d images containing < 10% non-zeros
    #     # print(np.count_nonzero(img_2d))
    #     #print("before process:", img_2d.shape)
    #     if float(np.count_nonzero(img_2d)) / img_2d.size >= preserving_ratio:  # 
    #         img_2d = img_2d / 127.5 - 1  # 
    #         img_2d = np.transpose(img_2d, (1, 0))  # rotate 90 degrees
    #         # plt.imshow(img_2d)
    #         # plt.pause(0.01)

    img_arr = np.squeeze(img)[:, :, 0]

    return img_arr

def loadRawData():
    # %%
    # The directory of given Dataset, you can change it
    dataDirectory = 'COVID19_1110/studies/'
    # Label for each class
    classNames = os.listdir(dataDirectory)
    imageFiles = [[] for i in range(len(classNames))]
    images = [[] for i in range(len(classNames))]
    layerNum = [[] for i in range(len(classNames))]
    imageFilesList = []
    imageClass = [[] for i in range(len(classNames))]

    for i in range(len(classNames)):
        for x in os.listdir(os.path.join(dataDirectory, classNames[i])):
            imageFiles[i].append(os.path.join(dataDirectory, classNames[i], x))

    # Number of data in each category

    for i in range(len(classNames)):
        for j in range(len(imageFiles[i])):
            # too large, play a demo dataset first
            tmp = nib.load(imageFiles[i][j]).get_fdata()[:, :, :2]

            layerNum[i].append(tmp.shape[2])
            images[i].append(np.array(tmp).swapaxes(0, 2).swapaxes(1, 2))
    numEachClass = [sum(layerNum[i]) for i in range(len(classNames))]

    for i in range(len(classNames)):
        imageFilesList.extend(imageFiles[i])
        imageClass[i] = (np.array([i] * numEachClass[i]).reshape(-1, 1))
    # Total number of images
    numTotal = 0
    for i in numEachClass:
        numTotal += i

        # The dimensions of each image
    Width, Height = getImage(imageFilesList[0]).shape

    print("Total number of images is " + str(sum(numEachClass)))
    print("There are " + str(len(classNames)) +" classes in total:")
    print(classNames)

    print("Corresponding data sample number for each class:")
    print(numEachClass)

    print("Image size:")
    print("(" + str(Width) + ", " +str(Height) + ")")
    # img = getImage(src)
    # print(img.shape)
    # plt.imshow(img, "gray")
    # plt.show()

    return images, imageClass, numEachClass

if __name__ == '__main__':
    images = loadRawData()
