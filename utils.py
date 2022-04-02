from torch.utils.data import Dataset
import torch
import numpy as np
import math
import os
from sklearn import metrics
from enum import Enum


dic = {'CT-0': 0, 'CT-1': 1, 'CT-2': 2, 'CT-3': 3}
labels=[0, 1, 2, 3]


class SPDataset(Dataset):
    """
        Initialize Dataset
    """

    def __init__(self, X, Y ):
        self.x_data = torch.tensor(X)
        self.y_data = torch.tensor(Y)
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def accuracy(output ,target):
    y_true = target.numpy()
    y_pred = torch.argmax(output, axis=-1).numpy()
    acc = np.equal(y_true, y_pred)

    acc = float(np.sum(acc))/(float(y_true.shape[0]))
    return acc

def mcc(TP, TN, FP, FN):
    return ((TP*TN-TP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

def pred(output):
    return torch.argmax(output, axis=-1)

# save the model and relevant parameters
def save_checkpoint(args, state, is_best):
    if not is_best:
        filename = './checkpoint/ckpt.pth.tar'
    else:
        filename = './checkpoint/best_ckpt.pth.tar'
    torch.save(state, filename)

def save_model(model, epoch, is_best, args, value):
    path = './checkpoint_%s' % (args.loss_type)
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    if not is_best:
        filename = './checkpoint_%s/ckpt_cnn_%s_%s.pth' % (args.loss_type, str(epoch), str(value))
    else:
        filename = './checkpoint_%s/best_ckpt_cnn_%s_%s.pth' % (args.loss_type, str(epoch), str(value))
    torch.save(model, filename, _use_new_zipfile_serialization=False)

def make_directory(keys):
    for key in keys:

        path = "./log/%s" % (key)
        isExists = os.path.exists(path)
        if not isExists:
            # If directory doesn't exist, build up a new directory
            print(path + ' build up successfully')
            os.makedirs(path)
        else:
            # Otherwise
            print(path + ' already exists')

def metric(mode, output, labels, cls=None):
    #print(X)

    # different modes
    if mode=='acc':

        all_preds=np.argmax(output.detach().cpu().numpy(), 1)
        if ( cls!=None):
            index_pred = np.argwhere(np.array(all_preds) == dic[cls]).squeeze(1).tolist()
            index_true = np.argwhere(np.array(labels) == dic[cls]).squeeze(1).tolist()
            acc = float(len([i for i in index_pred if i in index_true]))/float(len(index_true))

            print("     test acc for " + cls + ": " + str(acc))
        else:
            acc = float(np.sum(np.equal(all_preds, np.array(labels))))/float(len(labels))
            print("     test acc for all: " + str(acc))
        return acc

    elif mode=='F1_score':

        all_preds = np.argmax(output.detach().cpu().numpy(), 1)
        index_pred = np.argwhere(np.array(all_preds) == dic[cls]).squeeze(1).tolist()
        index_true = np.argwhere(np.array(labels) == dic[cls]).squeeze(1).tolist()
        # TP
        TP = float(len([i for i in index_pred if i in index_true]))
        # FP
        FP = float(len(index_pred))-TP
        FN = float(len(index_true))-TP

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1_score = 2.0*precision*recall/(precision+recall)

        print("     test F1 score for "+cls+": "+str(F1_score))
        return F1_score

    else :
        # take MCC as metric by default

        all_preds = np.argmax(output.detach().cpu().numpy(), 1)
        index_pred = np.argwhere(np.array(all_preds) == dic[cls]).squeeze(1).tolist()
        index_true = np.argwhere(np.array(labels) == dic[cls]).squeeze(1).tolist()

        index_pred_n = np.argwhere(np.array(all_preds) != dic[cls]).squeeze(1).tolist()
        index_true_n = np.argwhere(np.array(labels) != dic[cls]).squeeze(1).tolist()
        # TP
        TP = float(len([i for i in index_pred if i in index_true]))
        TN = float(len([i for i in index_pred_n if i in index_true_n]))
        # FP
        FP = float(len(index_pred)) - TP
        FN = float(len(index_true)) - TP

        MCC = mcc(TP, TN, FP, FN)

        print("     test MCC for " + cls + ": " + str(MCC))
        return MCC

def metric_advanced(mode, y_pred, labels):

    # micro average/mactro average
    # options for state:
    #   macro, micro, weighted

    y_test = labels

    result=0.0
    print()
    if (mode=="F1_score"):
        result = metrics.f1_score(y_test, y_pred, labels=labels)

    elif (mode=="precision"):
        result = metrics.precision_score(y_test, y_pred, labels=labels)

    elif (mode=="recall"):
        result = metrics.recall_score(y_test, y_pred, labels=labels)

    elif (mode=="MCC"):
        result = metrics.matthews_corrcoef(y_test, y_pred)

    elif (mode=="AUC_ROC"):
        result = metrics.roc_auc_score(y_test, y_pred)

    elif (mode == "Kappa"):
        result = metrics.cohen_kappa_score(y_test, y_pred)

    elif (mode == "SN"):
        confusion = metrics.confusion_matrix(y_test, y_pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        result=(TP / float(TP + FN))

    elif (mode == "SP"):
        confusion = metrics.confusion_matrix(y_test, y_pred)
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        result=(TN / float(TN + FP))

    elif (mode == "balanced_accuracy"):
        result = metrics.balanced_accuracy_score(y_test, y_pred)

    print("test "+mode + " :" + str(result))

    return result

def concatenate(list):
    head = list[0]

    for i in range(1, len(list)):
        head = np.concatenate((head, list[i]), 0)

    return np.array(head)
