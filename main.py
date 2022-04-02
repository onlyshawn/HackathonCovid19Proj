from __future__ import print_function
import numpy.random as random
import torch.backends.cudnn as cudnn
import torch.nn as nn
from utils import *
from DataLoader import *
from Losses import *
import argparse
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from model.ResNet import resnet18

dic = {'CT-0': 0, 'CT-1': 1, 'CT-2': 2, 'CT-3': 3}
rev_dic = {0 : 'CT-0', 1 : 'CT-1', 2 : 'CT-2', 3 : 'CT-3'}
metric_basic = ['acc', 'F1_score']
metric_ad = ['MCC']
avg = ['micro', 'macro']
classes = len(dic)

def createTrainTestData(test_split=0.04, val_split=0.20, seed = 1997
              ):
    # Initialize
    if (not os.path.exists("./cache")):
        image_data, labels, cls_num_list = loadRawData()
        image_data_per_class = [[] for i in range(len(image_data))]
        for i in range(classes):
            image_data_per_class[i] = concatenate(image_data[i])
            print("Shape for data samples in class " + rev_dic[i])
            print(image_data_per_class[i].shape)
            print(labels[i].shape)
            np.random.shuffle(image_data_per_class[i])

        # Split training set, validation set and test set
        X_train = np.vstack( image_data_per_class[i][:int(image_data_per_class[i].shape[0] * (1 - val_split))]
                            for i in range(classes)
                             )
        Y_train = np.vstack( labels[i][:int(image_data_per_class[i].shape[0] * (1 - val_split))]
                            for i in range(classes)
                             )

        X_val = np.vstack( image_data_per_class[i][int(image_data_per_class[i].shape[0] * (1 - val_split)):
                                                   int(image_data_per_class[i].shape[0] * (1 - test_split))]
                            for i in range(classes)
                             )
        Y_val = np.vstack( labels[i][int(image_data_per_class[i].shape[0] * (1 - val_split)):
                                                   int(image_data_per_class[i].shape[0] * (1 - test_split))]
                            for i in range(classes)
                             )

        X_test = np.vstack( image_data_per_class[i][int(image_data_per_class[i].shape[0] * (1 - test_split)):]
                            for i in range(classes)
                             )
        Y_test= np.vstack( labels[i][int(image_data_per_class[i].shape[0] * (1 - test_split)):]
                            for i in range(classes)
                             )
        cls_num_list_np = np.array(cls_num_list)
        os.makedirs("./cache")
        np.save("./cache/X_train.npy", X_train)
        np.save("./cache/Y_train.npy", Y_train)
        np.save("./cache/X_val.npy", X_val)
        np.save("./cache/Y_val.npy", Y_val)
        np.save("./cache/X_test.npy", X_test)
        np.save("./cache/Y_test.npy", Y_test)
        np.save("./cache/cls.npy", cls_num_list_np)

    else:
        X_train = np.load("./cache/X_train.npy")
        Y_train = np.load("./cache/Y_train.npy")
        X_val = np.load("./cache/X_val.npy")
        Y_val = np.load("./cache/Y_val.npy")
        X_test = np.load("./cache/X_test.npy")
        Y_test = np.load("./cache/Y_test.npy")
        cls_num_list_np = np.load("./cache/cls.npy")
        cls_num_list = cls_num_list_np.tolist()
    X_train = X_train[:, np.newaxis].astype(np.float32);
    X_val = X_val[:, np.newaxis].astype(np.float32);
    X_test = X_test[:, np.newaxis].astype(np.float32);
    Y_train = np.squeeze(Y_train.astype(np.int64))
    Y_val = np.squeeze(Y_val.astype(np.int64))
    Y_test = np.squeeze(Y_test.astype(np.int64))
    print("============Statistical Information===========")
    print("Training Set: ")
    print("Shape of data: " + str(X_train.shape) + "; Shape of labels: " + str(Y_train.shape))
    print("Validation Set: ")
    print("Shape of data: " + str(X_val.shape) + "; Shape of labels: " + str(Y_val.shape))
    print("Test Set: ")
    print("Shape of data: " + str(X_test.shape) + "; Shape of labels: " + str(Y_test.shape))
    # shuffle

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), cls_num_list[:classes]


def train(train_loader, model, criterion, optimizer, epoch, args):
    # 训练模式
    model.train()

    end=time.time()

    for i, (input, target) in enumerate(train_loader):
        target = target.reshape(-1)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        else:
            input = input.cuda()
            target = target.cuda()


        # print('input:')
        # print(input)
        # compute output
        output = model(input)

        # print('output:')
        # print(output)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        duration = time.time()-end
        end = time.time()
        losses=loss.item()
        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}]\t'   
                      'Loss {loss:.4f} \t'
                      'Duration:{duration:.3f}'.format(
                epoch, i, len(train_loader),loss=losses, duration=duration))  # TODO
            #print(output)

def validate(train_loader, model, criterion, epoch, args, best_acc):

    # 评估模式，模型不再更新参数
    model.eval()
    all_preds = []
    all_targets = []

    index_pred = {'CT-0': [], 'CT-1': [], 'CT-2': [], 'CT-3': []}
    index_true = {'CT-0': [], 'CT-1': [], 'CT-2': [], 'CT-3': []}

    with torch.no_grad():
        end = time.time()

        for i, (input, target) in enumerate(val_loader):

            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            else:
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = accuracy(output.cpu(), target.cpu())
            losses = loss.item()

            # measure elapsed time
            duration=time.time()-end
            end = time.time()

            all_preds.extend(pred(output).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss:.4f} \t'
                          'Accuracy:{acc:.3f}\t'
                          'Duration:{duration:.3f}\t'.format(
                    epoch, i, len(train_loader), loss=losses, acc=acc, duration=duration))  # TODO
                #print(output)


        # 获得预测中为0元素的索引
        for key in index_pred.keys():

            index_pred[key] = np.argwhere(np.array(all_preds)==dic[key]).squeeze(1).tolist()
            index_true[key] = np.argwhere(np.array(all_targets)==dic[key]).squeeze(1).tolist()
            set1 = set(index_pred[key])
            set2 = set(index_true[key])
            iset = list(set1.intersection(set2))
            true_neg = len(iset)
            total_neg = len(index_true[key])
            print("     accuracy for "+key+": "+str(float(true_neg/total_neg)))

        total_acc=np.sum(np.equal(np.array(all_preds), np.array(all_targets)))/len(all_preds)

        print("total_acc:"+str(total_acc))

        return best_acc

def test(model, loader, out_f, out_f_all_acc, out_f_ad, best_mcc, args):

    output=[]
    labels_test=[]

    for i, (input, target) in enumerate(loader):
        target_test = target

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target_test = target_test.cuda(args.gpu, non_blocking=True)
        else:
            input = input.cuda()
            target_test = target_test.cuda()

        o1, _, o2 = model(input)

        output.extend(o1.cpu().detach().numpy())
        labels_test.extend(target_test.cpu().detach().numpy())

    output = torch.tensor(np.array(output))
    labels_test = np.array(labels_test)

    for key in dic.keys():
        for m in metric_basic:
            result = metric(m, output, labels_test, key)
            out_f[key][m].write(str(result) + "\n")
    result = metric("acc", output, labels_test)
    out_f_all_acc.write(str(result) + "\n")

    # test type output
    for m in metric_ad:
        y_pred = pred(output).cpu()
        result_ad = metric_advanced(m, y_pred, labels_test)
        out_f_ad[m].write(str(result_ad) + "\n")
        if (m == "MCC"):
            if (result_ad > best_mcc):
                best_mcc = result_ad
                save_model(model, epoch+1 , True, args, float('%.4f' % best_mcc))
                print("best model saved!")

    print()
    print("BEST MCC:" + str(best_mcc))

    torch.cuda.empty_cache()

    return best_mcc


if __name__ == '__main__':

    # Specify the used GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 参数定义
    parser = argparse.ArgumentParser(description='Baseline model for COVID19')
    parser.add_argument('--model_arch', default="ResNet", type=str, help='the architecture of model')
    parser.add_argument('--loss_type', default="Normal", type=str, help='loss type')
    parser.add_argument('--train_rule', default="Reweight", type=str, help='train rule')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--attention', default=False, type=bool,
                        help='Use attention or not')

    args = parser.parse_args()
    if args.seed is not None:
        # set the random seed
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # 创建模型
    print("Creating the model……")
    if (args.model_arch=="ResNet"):
        model = resnet18(pretrained=False, b_RGB=False)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # 创建存放log的文件夹
    make_directory(dic.keys())

    # 加载数据
    print('Loading data...')
    train_set, val_set, test_set, cls_num_list = createTrainTestData()
    X_train, Y_train = train_set
    X_val, Y_val = val_set
    X_test, Y_test = test_set

    train_dataset = SPDataset(X_train, Y_train)
    val_dataset = SPDataset(X_val, Y_val)
    test_dataset = SPDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)


    print("Number of test samples:")
    print(len(X_test))
    # 定义优化器

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.weight_decay)


    print("Loss type is:" + args.loss_type)
    print("================================================================")
    out_f = {'CT-0': {}, 'CT-1': {},'CT-2':{}, 'CT-3':{}}
    out_f_ad = {'F1_score': None, 'MCC': None, "AUC_ROC": None, "Kappa": None}

    for key in dic.keys():
        for m in metric_basic:
            file_name = "./log/%s/model_%s_%s.txt" % (key, args.loss_type, m)
            out_f[key][m] = open(file_name, 'w')

    out_f_all_acc = open("./log/model_%s_%s.txt" % (args.loss_type, "acc"), 'w')

    for key in out_f_ad.keys():
        out_f_ad[key] = open("./log/model_%s_%s.txt" % (args.loss_type, key), 'w')

    best_acc = 0.0
    best_mcc = 0.0
    # 开始训练
    for epoch in range(args.epochs):

        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)

            # effective_num_kingdom = 1.0 - np.power(beta, cls_kingdom_num_list)
            # per_cls_weights_kingdom = (1.0 - beta) / np.array(effective_num_kingdom)
            # per_cls_weights_kingdom = per_cls_weights_kingdom / np.sum(per_cls_weights_kingdom) * len(per_cls_weights_kingdom)
            # per_cls_weights_kingdom = torch.FloatTensor(per_cls_weights_kingdom).cuda(args.gpu)
            # kingdomloss = LDAMLoss(cls_num_list=cls_kingdom_num_list, max_m=0.3, s=150, weight=per_cls_weights_kingdom).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)

        # 定义损失函数
        if args.loss_type == 'LDAM':
            # 给的几个参数：
            # 每个列的sample数量， 每个列的权重，max_m，以及s
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.3, s=150, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM_CB':
            criterion = LDAMLoss_CB(cls_num_list=cls_num_list, max_m=0.3, s=150).cuda(args.gpu)

        elif args.loss_type == 'Focal':
            criterion = FocalLoss(gamma=2.0).cuda(args.gpu)
        elif args.loss_type == 'Normal':
            criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
        else:
            raise Exception("Wrong loss type!")

        print("================================")
        print("epoch:" + str(epoch + 1))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        # evaluate on validation set
        best_acc = validate(val_loader, model, criterion, epoch, args, best_acc)

        # test
        model.eval()
        if((epoch+1)>=10):
            if ((epoch + 1) % 1 == 0):
                best_mcc = test(model, test_loader, out_f, out_f_all_acc, out_f_ad,
                                                                    best_mcc,
                                                                    args)

        print("================================")

    for key in dic.keys():
        for m in metric_basic:
            out_f[key][m].close()
    out_f_all_acc.close()

    for key in out_f_ad.keys():
        out_f_ad[key].close()

