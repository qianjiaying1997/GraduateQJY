import sys
sys.path.append('..')

import torch
import torch.nn.parallel
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd

import random
import os
import copy
import argparse
from tqdm import tqdm
import numpy as np
from cifar_train_val_test import CIFAR10, CIFAR100
from termcolor import cprint
from prototype import ProNet, regularization

from preact_resnet_cifar10 import preact_resnet32_cifar


def sample_selection(f_x, th=0.5):
    """
    Sample selection based on the gap between the first and the second largest prob
    """
    pure_idx = []
    for i in range(int(len(f_x))):
        top2, _ = torch.topk(f_x[i],2)
        gap = torch.abs(top2[0]-top2[1])
        if gap > th:
            pure_idx.append(i)
    return th, np.array(pure_idx)

# random seed related
def _init_fn(worker_id):
    np.random.seed(77 + worker_id)
#    np.random.seed(0)



def mixup_data(x, y,sample_weight, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_x_weight = lam * sample_weight + (1-lam) * sample_weight[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam,mixed_x_weight


def mixup_criterion(criterion, pred, y_a, y_b, lam, weight):

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)




# @profile
def main(args):
    random_seed = args.seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well



    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters Setting
    batch_size: int = args.Batch_Size
    num_workers: int = 4
    train_val_ratio: float = 0.9
    lr: float = args.lr
    current_th: float = args.th
    tao: float = args.tao

    noise_label_path = os.path.join('noisy_labels', args.noise_label_file)
    if args.noise_type == 'other':
        noise_y = list(pd.read_csv(noise_label_path)['label_noisy'].values.astype(int))
        noise_y = np.array(noise_y)
        train_num = int(len(noise_y)*train_val_ratio)
        noise_y = noise_y[:train_num]
    else:
        noise_y = np.load(noise_label_path)
    print('Load noisy label from {}'.format(noise_label_path))

    which_data_set = args.noise_label_file.split('-')[0]

    # data_ augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(root='data', split='train', train_ratio=train_val_ratio, trust_ratio=0, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              worker_init_fn=_init_fn)

    valset = CIFAR10(root='data', split='val', train_ratio=train_val_ratio, trust_ratio=0, download=False, transform=transform_test)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    testset = CIFAR10(root='data', split='test', download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_class = 10
    in_channel = 3
    

    print('train data size:', len(trainset))
    print('validation data size:', len(valset))
    print('test data size:', len(testset))

    # -- Sanity Check --
    num_noise_class = len(np.unique(noise_y))
    assert num_noise_class == num_class, "The data class number between the generate noisy label and the selected dataset is incorrect!"
    assert len(noise_y) == len(trainset), "The number of noisy label is inconsistent with the training data number!"

    # -- generate noise --
    gt_clean_y = copy.deepcopy(trainset.get_data_labels())
    noise_y_train = noise_y.copy()
    trainset.update_corrupted_label(noise_y_train)


    real_noise_level = np.sum(noise_y_train != gt_clean_y) / len(noise_y_train)
    print('\n>> Real Noise Level: {}'.format(real_noise_level))
    y_train_tilde = copy.deepcopy(noise_y_train)


    # -- set network, optimizer, scheduler, etc
    f = preact_resnet32_cifar()
    ch_in = 3
    n_units = 16
    student = ProNet(channel_in = ch_in, num_hidden_units = n_units, num_classes = num_class,s=tao)

    file_name = '(' + which_data_set + ')' + args.noise_label_file + '.txt'
    log_dir = 'log/logs_txt_' + str(random_seed)
    os.makedirs(log_dir, exist_ok=True)
    file_name = os.path.join(log_dir, file_name)
    saver = open(file_name, "w")
    saver = open(file_name, "w")


    print("\n")
    print("============= Parameter Setting ================")
    print("Using Data Set : {}".format(which_data_set))
    print("Training Epoch : {} | Batch Size : {} | Learning Rate : {} ".format(args.nepoch, batch_size, lr))
    print("================================================")
    print("\n")

    print("============= Start Training =============")
    print("-- Start Sample Selection at Epoch : {} --\n".format(args.warm_up))

    criterion = torch.nn.NLLLoss(reduction='none')
    optimizer = torch.optim.SGD(f.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    # optimizer_s = torch.optim.SGD(student.parameters(),lr = 0.01,momentum = 0.9, nesterov = True, weight_decay = 5e-4)
    optimizer_s = torch.optim.Adam(student.parameters(),lr=0.01)
    scheduler_s = MultiStepLR(optimizer_s, milestones = [40,80],gamma = 0.5)
    scheduler = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.5)
    f = f.to(device)
    student = student.to(device)

    f_record = torch.zeros([args.rollWindow, len(y_train_tilde), num_class])
    f_pred_record = torch.zeros([len(y_train_tilde)]).long()
    weight_record = torch.zeros([len(y_train_tilde)]).cuda()
    prob_record = torch.ones([args.rollWindow, len(y_train_tilde),num_class]).cuda()
    p_weight = torch.ones([len(y_train_tilde),num_class]).cuda()
    pure_idx = np.zeros(len(y_train_tilde))

    best_acc = 0
    s_best_acc = 0
    best_epoch = 0
    s_best_epoch = 0

    for epoch in range(args.nepoch):
        train_loss = 0
        train_correct = 0
        train_total = 0

        f.train()
        student.train()
        for _, (features, labels, indices) in enumerate(tqdm(trainloader, ascii=True, ncols=50)):
            if features.shape[0] == 1:
                continue

            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            fea,outputs = f(features)
            outputs = F.softmax(outputs,dim=1)
            currentweight = torch.ones(outputs.shape).cuda()/float(len(labels))


            train_total += features.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

            f_record[epoch % args.rollWindow, indices] = outputs.detach().cpu()  #save 5 epoch for rolling len
            f_pred_record[indices]=predicted.detach().cpu()
            f_pred = f_record.mean(0)[indices]
            _, p_label = f_pred.max(1)
            if epoch > args.warm_up:
                update_idx = np.isin(indices,pure_idx)
                update_idx = np.arange(labels.shape[0])[update_idx]
                if len(update_idx)>0:
                    optimizer_s.zero_grad()
                    feat, centers,distance,outputs_pro = student(fea.detach())
                    if epoch > args.stage2:
                      currentweight = p_weight[indices]
                    _, preds = torch.max(distance, 1)
                    loss1 = F.nll_loss((torch.log(outputs_pro[update_idx])),predicted[update_idx].detach())
                    loss2=regularization(feat[update_idx], centers, predicted[update_idx].detach())
                    p_loss=loss1+1e-2*loss2
                    p_loss.backward()
                    optimizer_s.step()
            y_one_hot = torch.zeros(features.shape[0],num_class).scatter_(1,labels.cpu().unsqueeze(1),1).cuda()
            sample_weight = currentweight.mul(y_one_hot).sum(1)
            sample_weight_sum = torch.sum(sample_weight)
            sample_weight = sample_weight/sample_weight_sum
            weight_record[indices] = sample_weight

            sample_loss = criterion(torch.log(outputs),labels)
            loss = torch.sum(sample_weight.mul(sample_loss))

            if epoch > args.stage2:
                inputs, targets_a, targets_b, lam, mixed_weight = mixup_data(features,labels,sample_weight)
                inputs, targets_a, targets_b = map(Variable,(inputs, targets_a, targets_b))
                _, outputs = f(inputs)
                outputs = F.softmax(outputs)
                sample_loss = lam * criterion(torch.log(outputs),targets_a) + \
                (1-lam)*criterion(torch.log(outputs), targets_b)
                loss += torch.sum(mixed_weight.mul(sample_loss))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()


        train_acc = train_correct / train_total * 100
        cprint("Epoch [{}|{}] \t  Train loss:{}:.3f \t Train Acc {:.3f}%".format(epoch+1, args.nepoch, train_loss, train_acc), "yellow")
        print('debug show weight max',torch.max(weight_record))
        print('debug show weight min',torch.min(weight_record))

        if epoch >= args.warm_up:
            f_x = f_record.mean(0)
            y_tilde = trainset.targets
            current_th, pure_idx = sample_selection(f_x,th=current_th)
            print("pure region",len(pure_idx))
            purify = float(np.sum(np.array(f_pred_record)[pure_idx] == np.array(gt_clean_y)[pure_idx])) / float(len(pure_idx))
            print('correct after correction:{}'.format(purify))
            student.eval()
            f.eval()
            with torch.no_grad():
                for i, (images, labels, indices) in enumerate(trainloader):
                    images,labels = images.to(device),labels.to(device)
                    fea,_ = f(images)
                    _,_,distance,_ = student(fea)
                    outputs_prob = torch.exp(distance/tao)
                    prob_record[epoch % args.rollWindow,indices] = outputs_prob
            p_x = prob_record.mean(0)  #n*c
            p_norm = torch.sum(p_x, dim=0)
            p_norm = p_norm.repeat(p_x.shape[0],1)
            p_weight = (p_x/p_norm)*(float(len(y_train_tilde))/float(num_class))

        # --- validation --
        val_total = 0
        val_correct = 0
        pro_correct = 0
        f.eval()
        student.eval()
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(valloader):
                images, labels = images.to(device), labels.to(device)

                fea,outputs = f(images)
                feat, centers,distance,outputs_pro = student(fea)
                _, preds = torch.max(distance, 1)

                val_total += images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                pro_correct += preds.eq(labels).sum().item()

        val_acc = val_correct / val_total * 100.
        pro_acc = pro_correct / val_total * 100.

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(f.state_dict())
            torch.save(best_weights,'./best_model/best_classifier.pkl')
        last_weights = copy.deepcopy(f.state_dict())
        if pro_acc > s_best_acc:
            s_best_acc = pro_acc
            s_best_epoch = epoch
            s_best_weights = copy.deepcopy(student.state_dict())
            torch.save(s_best_weights,'./best_model/best_prototype.pkl')
        s_last_weights = copy.deepcopy(student.state_dict())

        cprint('val accuracy: {}'.format(val_acc), 'cyan')
        cprint('prototype accuracy:{}'.format(pro_acc),'cyan')
        saver.write('Val Accuracy: {}\n'.format(val_acc))
        saver.write('prototype Accuracy: {}\n'.format(pro_acc))
        cprint('>> best accuracy: {}\n>> best epoch: {}\n'.format(best_acc, best_epoch), 'green')
        cprint('>> best prototype accuracy: {}\n>> best epoch: {}\n'.format(s_best_acc, s_best_epoch), 'green')
        scheduler.step()
        scheduler_s.step()
#
    # -- Final testing
    cprint('>> testing using best validation model <<', 'cyan')
    test_total = 0
    test_correct = 0
    pro_test_correct = 0
#
    f.load_state_dict(last_weights)
    student.load_state_dict(s_last_weights)
    f.eval()
    student.eval()

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            fea,outputs = f(images)
            fea, centers,distance,outputs_pro = student(fea)
            _, preds = torch.max(distance, 1)

            test_total += images.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
            pro_test_correct += preds.eq(labels).sum().item()

    test_acc = test_correct / test_total * 100.
    pro_test_acc = pro_test_correct / test_total * 100.
    cprint('>> test accuracy last: {}'.format(test_acc), 'cyan')
    cprint('>> prototype test accuracy last:{}'.format(pro_test_acc),'cyan')
    saver.write('>> Final test accuracy: {}\n'.format(test_acc))

    print("Final Test Accuracy last {:3.3f}%".format(test_acc))
    print("Final Prototype Test Accuracy last {:3.3f}".format(pro_test_acc))
    # print("Final Delta Used {}".format(current_delta))

   #=================test with best model =============
    f.load_state_dict(torch.load('./best_model/best_classifier.pkl'))
    student.load_state_dict(torch.load('./best_model/best_prototype.pkl'))
    f.eval()
    student.eval()
    test_total = 0
    test_correct = 0
    pro_test_correct = 0

    with torch.no_grad():
        for i, (images, labels,_) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            fea,outputs = f(images)
            feat, centers,distance,outputs_pro = student(fea)
            _, preds = torch.max(distance, 1)

            test_total += images.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
            pro_test_correct += preds.eq(labels).sum().item()

    test_acc = test_correct / test_total * 100.
    pro_test_acc = pro_test_correct / test_total * 100.
    cprint('>> test accuracy best: {}'.format(test_acc), 'cyan')
    cprint('>> prototype test accuracy best:{}'.format(pro_test_acc),'cyan')

    print("Final Test Accuracy best {:3.3f}%".format(test_acc))
    print("Final Prototype Test Accuracy best {:3.3f}".format(pro_test_acc))
    return test_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_label_file', default='cifar10-dependent0.4.csv', help='noise label file', type=str)
    parser.add_argument('--noise_type', default='other', help='noise type [uniform | asym]', type=str)
    parser.add_argument('--noise_level', default=0.0, help='noise level [for additional uniform/asymmetric noise applied to the PMD noise]', type=float)
    parser.add_argument("--Batch_Size", default=128, help="batch size", type=int)
    parser.add_argument("--th", default=0.5, help="entropy threshold", type=float)
    parser.add_argument("--nepoch", default=120, help="number of training epochs", type=int)
    parser.add_argument("--rollWindow", default=5, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--gpus", default=1, help="which GPU to use", type=int)
    parser.add_argument("--warm_up", default=15, help="warm-up period", type=int)
    parser.add_argument("--stage2", default=30, help="stage2", type=int)
    parser.add_argument("--seed", default=8, help="random seed", type=int)
    parser.add_argument("--lr", default=0.01, help="initial learning rate", type=float)
    parser.add_argument("--tao", default=5, help="tao", type=float)
    args = parser.parse_args()
    main(args)
