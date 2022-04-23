import tensorlayer as tl
import numpy as np
import os
import nibabel as nib
import cv2

nib.Nifti1Header.quaternion_threshold = - np.finfo(np.float32).eps * 10  # loose the limit
training_data_path = "CT-0"
preserving_ratio = 0.25 # filter out 2d images containing < 25% non-zeros


def getImage(src):
    img_path = src
    # print(img_path)
    img = nib.load(img_path).get_fdata()
    # print(img.shape)
    img_3d_max = np.amax(img)
    img = img / img_3d_max * 255

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
            tmp = nib.load(imageFiles[i][j]).get_fdata()[:, :, :10]

            layerNum[i].append(tmp.shape[2])
            img_list = []
            for id in range(tmp.shape[2]):
                # image size control
                tmp_img = process_image(tmp[:, :, id], 224)
                img_list.append(tmp_img)

            images[i].append(np.array(img_list))
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

def process_image(img, min_side):
    size = img.shape
    h, w = size[0], size[1]
    #rescale longest side to min_side
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    else:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目

    return pad_img

if __name__ == '__main__':
    # images = loadRawData()
    tmp = np.ones((512, 512, 2))
    img_list = []
    for id in range(tmp.shape[2]):
        tmp_img = process_image(tmp[:, :, id], 128)
        img_list.append(tmp_img)

    print(np.array(img_list).shape)