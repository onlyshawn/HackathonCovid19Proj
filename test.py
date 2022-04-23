import nibabel as nib
import torch
from model.ResNet import resnet18
from torch.autograd import Variable
import cv2
import numpy as np
from scipy import ndimage as nd
from utils import *

# ct
ct_img = None
metric_basic = ['acc']
metric_ad = ['MCC']

# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = resnet18(pretrained=False, b_RGB=False)
model = torch.load('./checkpoint_LDAM/best_ckpt_cnn_60_0.959.pth',map_location=torch.device(device))#.module.state_dict())
# print(net)
model = model.cuda()
model.eval()
# ct_img = nib.load("study_0256.nii.gz").get_fdata()

X_test = np.load("./cache/X_test.npy")
Y_test = np.load("./cache/Y_test.npy")

X_test = X_test[:, np.newaxis].astype(np.float32);
Y_test = np.squeeze(Y_test.astype(np.int64))


test_dataset = SPDataset(X_test, Y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
output = []
labels_test = []

for i, (input, target) in enumerate(test_loader):
    target_test = target

    input = input.cuda()
    target_test = target_test.cuda()
    o1 = model(input)
    output.extend(o1.cpu().detach().numpy())
    labels_test.extend(target_test.cpu().detach().numpy())

output = torch.tensor(np.array(output))
labels_test = np.array(labels_test)

for key in dic.keys():
    for m in metric_basic:
        result = metric(m, output, labels_test, key)
result = metric("acc", output, labels_test)

# test type output
for m in metric_ad:
    y_pred = pred(output).cpu()
    result_ad = metric_advanced(m, y_pred, labels_test)
    if (m == "MCC"):
        best_mcc = result_ad


print()
print("BEST MCC:" + str(best_mcc))

diag = np.zeros(4)
p_diag = np.zeros(4)

labels_test = labels_test.reshape(-1, 1)
output = torch.argmax(output, 1).reshape(-1, 1).numpy()
for i in range(4):
    diag[i] = np.sum(labels_test==i)
    p_diag[i] = np.sum(output==i)


print(diag)
print(p_diag)

torch.cuda.empty_cache()