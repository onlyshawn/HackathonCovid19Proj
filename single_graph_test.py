import nibabel as nib
import torch
from model.ResNet import resnet18
from torch.autograd import Variable
import cv2
import numpy as np
from scipy import ndimage as nd




# def clear():
#     output_valueText.delete(1.0, 'end')

# def exit():
#     root.destroy()
def concatenate(list):
    head = list[0]

    for i in range(1, len(list)):
        head = np.concatenate((head, list[i]), 0)

    return np.array(head)

    """
Downsample the image and unify the number of CT slices
"""
def resize(img, stand_size, stand_slices):
    # img = nd.rotate(img, 90, reshape=False) # unnecessary to rotate images
    scale = float(stand_size / img.shape[0])
    slices = img.shape[2]
    if (stand_slices-slices)==1:
        return nd.zoom(input = img[:,:,:slices-1], zoom=(scale, scale, 1), order=1)
    # elif (slices-stand_slices)==1:
    #     img = nd.zoom(input = img, zoom=(scale, scale, 1), order=1)
    #     last = img[:,:,slices-1]
    #     img.append(last)
    #     return img
    return nd.zoom(input = img, zoom=(scale, scale, float(stand_slices/slices)), order=1)

"""
    resize image
"""
def process_image(img, min_side):
    size = img.shape
    h, w = size[0], size[1]
    # rescale the minside
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # minside * minside
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    else:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=[0,0,0]) #
    return pad_img


"""
Use model to diagnose
"""
def diagnose():
    global ct_img

    slices = ct_img.shape[2]
    diag = np.zeros(4)

            # images = [[]]
            # img_list = []
            # for id in range(tmp.shape[2]):
            #     # image size control
            #     tmp_img = process_image(tmp[:, :, id], 64)
            #     img_list.append(tmp_img)
            # images[i].append(np.array(img_list))

    img_list = []
    images = [[]]

    print(ct_img.shape)
    for i in range(30):
        img_list.append(process_image(ct_img[:, :, i],128))
    images[0].append(np.array(img_list))#.swapaxes(1,2))

    images[0] = concatenate(images[0])
    np.random.shuffle(images[0])

    img_set = images[0][:,np.newaxis].astype(np.float32)
    print("Shape of data: " + str(img_set.shape))
    loader = torch.utils.data.DataLoader(img_set, batch_size=4)
    output = []
    for i, (img_data) in enumerate(loader):
        img_data = img_data.cuda()
        res = net(img_data)

        output.extend(np.argmax(res.detach().cpu().numpy(), 1))
        # all_preds =

    # output = torch.tensor(np.array(output))
    print(output)
    # _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(output)
    for i in range(len(preds)):
        if preds[i] == 0:
            diag[0]+=1
        elif preds[i] == 1:
            diag[1]+=1
        elif preds[i] == 2:
            diag[2]+=1
        elif preds[i] == 3:
            diag[3]+=1
    print(diag)



# ct
ct_img = None

# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net = resnet18(pretrained=False, b_RGB=False)
net = torch.load('./checkpoint_LDAM/best_ckpt_cnn_60_0.959.pth',map_location=torch.device(device))#.module.state_dict())
# print(net)
net = net.cuda()
net.eval()
ct_img = nib.load("./COVID19_1110/studies/CT-3/study_1074.nii.gz").get_fdata()
diagnose()
