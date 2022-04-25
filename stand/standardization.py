"""
This is the standardization module of the MosMed CT image datasets
* select lung area layers
* unify to the same number of layers
* normalization
* abnormal layer detect and fixing


Author: Gupta Shreyash, Yinuo Wang
Date: 2022.04.02

"""


import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from tqdm import tqdm
import shutil
import time


"""
Load nifti image from dataset
"""


def nifti_loader(img):
	# print("loading file...")
	return nib.load(img).get_fdata()


"""
Save the processed CT data for later use
"""


def nifti_saver(img, dir_path, img_path):
	path = os.path.join(dir_path, img_path)
	# print("converting...")
	nifti_file = nib.Nifti1Image(img, np.eye(4))
	# print("saving...")
	nib.save(nifti_file, path)
	
	
"""
Normalize the ct image into 0-1 range
"""


def normalize(img):
	w,h,z = img.shape
	
	for k in range(z):
		vmin = np.min(img[:, :, k])
		vmax = np.max(img[:, :, k])
		img[:, :, k] = (img[:, :, k] - vmin) / (vmax - vmin + 1e-8)
	return img


"""
Lung area layer selector
"""


def select(img, threshold=0.3):
	slices = img.shape[2]
	img_lung = img[:, :, int(threshold * slices) : int((1-threshold) * slices)]
	return img_lung



"""
Unify the number of CT slices
"""


def unify(img, stand_size, stand_slices):
	
	scale = float(stand_size / img.shape[0])
	slices = img.shape[2]
	
	if (stand_slices - slices) == 1:
		return nd.zoom(input=img[:, :, :slices - 1], zoom=(scale, scale, 1), order=3)

	return nd.zoom(input=img, zoom=(scale, scale, float(stand_slices / slices)), order=3)


"""
Convert CT image to grayscale
"""


def ct2gray(image):
	minv = np.min(image)
	maxv = np.max(image)
	return np.floor((image - minv) / (maxv - minv + 1e-8) * 255)


def ct2gray2(image):
	for i in range(image.shape[2]):
		minv = np.min(image[:, :, i])
		maxv = np.max(image[:, :, i])
		image[:, :, i] = ((image[:, :, i] - minv) / (maxv - minv + 1e-8))
	return image


"""
Compute the histogram of image
"""


def computeHist(gray):
	if len(gray.shape) != 2:
		print("length error")
		return None
	w, h = gray.shape
	hist = {}
	for k in range(256):
		hist[k] = 0
	for i in range(w):
		for j in range(h):
			if hist.get(gray[i][j]) is None:
				gray[i][j] = 0
			hist[gray[i][j]] += 1
	hist[0] = 0
	# normalize
	n = w * h
	for key in hist.keys():
		hist[key] = float(hist[key]) / n
	return hist


"""
Implement histogram equalization
"""


def equalization(grayArray, h_s, nums):
	tmp = 0.0
	h_acc = h_s.copy()
	for i in range(nums):
		tmp += h_s[i]
		h_acc[i] = tmp
	
	if len(grayArray.shape) != 2:
		print("length error")
		return None
	w, h = grayArray.shape
	des = np.zeros((w, h), dtype=np.uint8)
	for i in range(w):
		for j in range(h):
			des[i][j] = int((nums - 1) * h_acc[grayArray[i][j]] * 1.25)
	return des


"""
Image Standardization with histogram
"""


def standardize(img):
	w, h, n = img.shape
	stand_img = np.zeros([w, h, n])
	for i in range(n):
		gray_img = ct2gray(img[:, :, i])
		hist = computeHist(gray_img)
		stand_img[:, :, i] = equalization(gray_img, hist, 256)
	# print(stand_img.shape)
	return stand_img


"""
Detect and fix abnormal layers
"""


def abnormal_detect(img):
	h, w, z = img.shape
	# imgg = ct2gray2(img)
	img_mean = []
	for i in range(z):
		sumi = 0
		cnt = 0
		for j in range(w):
			for k in range(h):
				if img[k, j, i] != 0:
					sumi += img[k, j, i]
					cnt += 1
		img_mean.append(sumi / cnt)
	
	for i in range(1, z - 1):
		
		sliding_avg = (img_mean[i - 1] + img_mean[i + 1]) * 0.5
		
		if abs(sliding_avg - img_mean[i]) > 0.1:
			# print(str(i) + " "+ str(sliding_avg - img_mean[i]))
			offset = abs(sliding_avg - img_mean[i])
			
			for k in range(h):
				for j in range(w):
					if img[k, j, i] != 0:
						img[k, j, i] = min(img[k, j, i] + offset, 1.0)
			img_mean[i] = sliding_avg
	return img




"""
Plot Images
"""


def sample_viewer(img):
	h, w, s = img.shape
	fig = plt.figure(figsize=(15, 15))
	for i in range(s):
		fig.add_subplot(int(s / 5)+1, 5, i + 1)
		plt.imshow(img[:, :, i], cmap="gray")
		plt.axis("off")
		plt.subplots_adjust(left=None, bottom=None, right=0.9, top=0.8, wspace=0.05, hspace=0.08)
	plt.show()


"""
Implement the standardization pipeline
"""


def standardization(img_path, stand_size, stand_slices):

	# load image
	img = nifti_loader(img_path)
	# print(str(img_path) + "  layers:" + str(img.shape[2]))
	
	# plot original data
	# sample_viewer(img)
	
	# rotate the image
	img_rot = nd.rotate(img, 90)
	
	# layer selector
	img_sel = select(img_rot)
	
	# layer unifier
	img_uni = unify(img_sel, stand_size, stand_slices)
	
	# normalization
	img_norm = normalize(img_uni)
	# print(img_norm[200:210,200:210,2])
	
	# abnormal fixing
	img_stand = abnormal_detect(img_norm)
	
	# plot standardized data
	sample_viewer(img_stand)
	
	# save standardized image
	# nifti_saver(img_stand, outdir, img_path)
	
	return img_stand


if __name__ == '__main__':
	debug = True
	if debug:
		standardization("study_1050.nii.gz", "./standardized", 512, 10)
	
	else:
		now = int(time.time())
		timeStamp = time.strftime("%Y-%m-%d-%H%M%S", time.localtime(now))
		
		dirpath = "./COVID19_1110/studies"
		outpath = "./Standardization/standard" + str(timeStamp)
		# initialize output directory
		if os.path.isdir(outpath) is True:
			shutil.rmtree(outpath)
		os.mkdir(outpath)
		
		# Processing parameters
		standard_size = 128
		standard_slice = [10, 10, 10, 10]
		
		# initialize the data
		imageList = []
		imageClass = []
		classType = os.listdir(dirpath)
		classType.remove('CT-4')
		classNum = [[] for i in range(len(classType))]
		
		for i in range(len(classType)):
			files = [f for f in sorted(os.listdir(os.path.join(dirpath, classType[i])))]
			num = 0
			for image_path in files:
				imageList.append(os.path.join(dirpath, classType[i], image_path))
				imageClass.append(i)
				num += 1
			classNum[i] = num
		print("There are " + str(len(classType)) + " classes in the dataset:")
		print(classType)
		print("Corresponding CT image in each class:")
		print(classNum)
		
		# pre-process all CT images
		classpath = []
		for i in range(len(classType)):
			classpath.append(outpath + '/CT-' + str(i))
			os.mkdir(classpath[i])
		
		start = 0
		for image in tqdm(imageList[start:start+classNum[0]], ncols=50):
			standardization(image, classpath[0], standard_size, standard_slice[0])
		start += classNum[0]
		for image in tqdm(imageList[start:start + classNum[1]], ncols=50):
			standardization(image, classpath[1], standard_size, standard_slice[1])
		start += classNum[1]
		for image in tqdm(imageList[start:start + classNum[2]], ncols=50):
			standardization(image, classpath[2], standard_size, standard_slice[2])
		start += classNum[2]
		for image in tqdm(imageList[start:], ncols=50):
			standardization(image, classpath[3], standard_size, standard_slice[3])