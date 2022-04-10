# -*- coding:utf-8 -*-
"""
This is the GUI of COVID-19 CT image diagnosis project.
Author: Yinuo Wang
Date: Apr 8, 2022

"""
# todo: Add pre-trained model in diagnose() function

import tkinter
import tkinter.filedialog
from tkinter import *
import matplotlib as plt
from matplotlib.pyplot import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
plt.use('TkAgg')
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import nibabel as nib
import torch
from model.ResNet import resnet18
from torch.autograd import Variable
import cv2
# create main window
root = tkinter.Tk()
root.title('COVID-19 AI Diagnostic system')

# window config
root.geometry('800x600+100+50')  # width x height + widthoffset + heightoffset
root.configure(bg='white')
root.resizable(False, False)
root.focusmodel()

# logo
img = ImageTk.PhotoImage(Image.open('img.png'))
label = Label(image=img, border=False)
label.place(x=530, y=0)

# ct
ct_img = None
# load model
net = resnet18(pretrained=False, b_RGB=False)
net.load_state_dict(torch.load('best_ckpt_cnn_60_0.959.pth',map_location=torch.device('cpu')).module.state_dict())
# print(net)


def clear():
    output_valueText.delete(1.0, 'end')

def exit():
    root.destroy()

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
generate dataset
"""
def concatenate(list):
    head = list[0]

    for i in range(1, len(list)):
        head = np.concatenate((head, list[i]), 0)

    return np.array(head)
"""
Use model to diagnose
"""
def diagnose():
    global ct_img
    slices = ct_img.shape[2]
    diag = np.zeros(4)
    img_set = []
    for i in range(slices):
        img_set.append(process_image(ct_img[i],128))
    img_set = np.array(img_set).swapaxes(1,2)
    # print(str(img_set.shape))

    np.random.shuffle(img_set)
    img_set = img_set[:,np.newaxis].astype(np.float32)
    # print("Shape of data: " + str(img_set.shape))
    loader = torch.utils.data.DataLoader(img_set, batch_size=32, shuffle=True)
    output = []
    for i, (img_data) in enumerate(loader):

        res = net(img_data)

        output.extend(res.detach().numpy())

    output = torch.tensor(np.array(output))
    print(output)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
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
    if np.max(diag) == diag[0]:
        output_valueText.insert('end',"Normal lung tissue, no CT-signs of viral pneumonia (CT-0)")
    elif np.max(diag) == diag[1]:
        output_valueText.insert('end',"Several ground-glass opacifications, involvement of lung parenchyma is less than 25% (CT-1)")
    elif np.max(diag) == diag[2]:
        output_valueText.insert('end',"Ground-glass opacifications, involvement of lung parenchyma is between 25 and 50% (CT-2)")
    elif np.max(diag) == diag[3]:
        output_valueText.insert('end',"Ground-glass opacifications and regions of consolidation, involvement of lung parenchyma is between 50 and 75% (CT-3)")
    else:
        output_valueText.insert('end',"Cannot get the diagnosis")




"""
plot CT images in the GUI
"""
def plotCT(img,image_label):
    h, w, s = img.shape
    fig = figure(figsize=(15,15))

    image_label = Label(root, bg='gray')
    image_label.place(x=0, y=0, width=500, height=600)
    for i in range(s):
        a = fig.add_subplot(int(s/6)+1, 6, i+1)
        a.imshow(img[:,:,i], cmap="gray")
        xticks([])
        yticks([])
        axis("off")

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=image_label)
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
    canvas.draw()


"""
Choose CT image file from disk
"""
def choosepic():
    global ct_img
    clear()

    path_ = askopenfilename()
    path.set(path_)
    ct_img = nib.load(file_entry.get()).get_fdata()
    plotCT(ct_img,image_label)


# Load CT
path = StringVar()
loadImg = tkinter.Button(root, text='Select CT File', font=('verdana',10,'bold'), command=choosepic)
loadImg.place(x=520, y=170, w=250, h=40)
file_entry = Entry(root, state='readonly', text=path)
file_entry.pack()
image_label = Label(root, bg='gray')
image_label.place(x=0, y=0, width=500, height=600)

# Diagnostic Result
output_valueTitle = StringVar()
output_valueTitle.set('Diagnostic Result')
label_outputTitle = tkinter.Label(root, textvariable=output_valueTitle,font=('verdana',10,'bold'), bg='white')
label_outputTitle.place(x=520, y=330)

output_valueText = tkinter.Text(root, width=12, bg="white", wrap=WORD, relief="sunken", borderwidth=2)
output_valueText.place(x=520, y=350, width=250, height=100, anchor=NW)

# Diagnose
comand = tkinter.Button(root, text="Diagnose", font=('verdana',10,'bold'), command=diagnose, width=10, height=2)
comand.place(x=520, y=250, width=250, height=40, anchor=NW)

# exit
exit_ = tkinter.Button(root,text="Exit",font=('verdana',10,'bold'),borderwidth=2,command=exit)
exit_.place(x=520, y=500, width=250, height=40,)
# start main loop
root.update()
root.mainloop()

