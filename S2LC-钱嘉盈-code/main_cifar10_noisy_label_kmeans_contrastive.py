'''Train CIFAR10 with PyTorch.'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import losses

import os
import argparse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from datetime import datetime
import math
from sklearn.decomposition import PCA
from preact_resnet_cifar10 import preact_resnet32_cifar
from BestMap import *
from sklearn.mixture import GaussianMixture


#from models import *
#from utils import progress_bar
import math 
import random


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



class ImageSet_dataset(Dataset):
    def __init__(self, mat_path, noise_type='asym', transform = None, train = True,noisy_rate = 0.1):
        # Exp_Type must be 'Train', 'Test' or 'Valid'
        img = np.zeros([50000,3,1024])
        labels = np.zeros(50000)
        if train == True:
            for i in range(5):
                data_dict = unpickle(mat_path[i])
                data = data_dict[b'data']
                img[(i*10000):(i+1)*10000,0,:] = data[:,:1024]
                img[(i*10000):(i+1)*10000,1,:] = data[:,1024:2048]
                img[(i*10000):(i+1)*10000,2,:] = data[:,2048:3072]
                labels[(i*10000):((i+1)*10000)] = data_dict[b'labels']


            #==============asymmetric noise================

            Img = img
            Labels = torch.from_numpy(labels).long()
            real_Labels = torch.from_numpy(labels).long()
            Noisy = torch.zeros(50000).long()

            if noise_type == 'asym':
                for i in range(50000):
                    r = np.random.random()
                    if r < noisy_rate:
                        if Labels[i] == 9:
                            Labels[i] = 1
                            Noisy[i] = 1
                        elif Labels[i] == 2:
                            Labels[i] = 0
                            Noisy[i] = 1
                        elif Labels[i] == 4:
                            Labels[i] = 7
                            Noisy[i] = 1
                        elif Labels[i] == 3:
                            Labels[i] = 5
                            Noisy[i] = 1
                        elif Labels[i] == 5:
                            Labels[i] = 3
                            Noisy[i] = 1
            
                    

            #=======================symmetric noise============
            else:
               for c in range(10):
                   idx_s = np.argwhere(labels==c).squeeze()
                   num_sam = idx_s.shape[0]
                   noise_sam = int(num_sam*noisy_rate)
                   Img[idx:(idx+num_sam),:,:] = img[idx_s,:,:]
                   Labels[idx:(idx+noise_sam)] = torch.round(c*torch.rand(idx_s.shape))
                   Labels[(idx+noise_sam):(idx+num_sam)] = c*torch.ones(num_sam-noise_sam)
                   Noisy[idx:(idx+noise_sam)] = torch.ones(noise_sam)
                   Noisy[(idx+noise_sam):(idx+num_sam)] = torch.zeros(num_sam-noise_sam)
                   real_Labels[idx:(idx+num_sam)] = c*torch.ones(num_sam)
                   idx = idx+num_sam
                
        else:
            data_dict = unpickle(mat_path)
            data = data_dict[b'data']
            img = np.zeros([10000,3,1024])
            img[:,0,:] = data[:,:1024]
            img[:,1,:] = data[:,1024:2048]
            img[:,2,:] = data[:,2048:3072]
            labels = np.array(data_dict[b'labels'])
            Img = img
            Labels = torch.from_numpy(labels)
            real_Labels = torch.from_numpy(labels)
            Noisy = torch.zeros(10000)
        self.transform = transform
        self.img = Img
        self.label = Labels.long()
        self.noisy = Noisy.long()
        self.real_label = real_Labels.long()
#        self.label = label - 1
        del Labels, Img,Noisy,real_Labels
        
    def __getitem__(self, idx):
        img_data = self.img[idx,:]
        Batch_data = np.zeros([32,32,3])
        Batch_data[:,:,0] = img_data[0,:].reshape(32,32)
        Batch_data[:,:,1] = img_data[1,:].reshape(32,32)
        Batch_data[:,:,2] = img_data[2,:].reshape(32,32)
        Batch_data = Image.fromarray(Batch_data.astype('uint8')).convert("RGB")
        Batch_label = self.label[idx]
        Batch_noisy = self.noisy[idx]
        Batch_real_label = self.real_label[idx]
        if self.transform is not None:
            Batch_data = self.transform(Batch_data)
        
        return Batch_data, Batch_label,Batch_noisy,Batch_real_label,idx
    
    def __len__(self):
        return len(self.label)



    
def Make_Loader(mat_path, noise_type='asym', transform = None,Batch_size = 256,train = True,noisy_rate = 0.1):
    data_set = ImageSet_dataset(mat_path,noise_type, transform,train,noisy_rate)
    new_loader = DataLoader(data_set, Batch_size, shuffle=True, num_workers=2)
    return new_loader





parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')  # 0.1 for pretrain in the begining 0.1 decay every 5 epochs 
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--noise_type', default='asym', help='noise type [sym | asym]', type=str)
parser.add_argument('--noise_level', default=0.0, help='noise level [for symmetric/asymmetric noise]', type=float)
parser.add_argument("--Batch_Size", default=128, help="batch size", type=int)
parser.add_argument("--nepoch", default=120, help="number of training epochs", type=int)
parser.add_argument("--warm_up", default=20, help="warm-up period", type=int)
parser.add_argument("--stage2", default=55, help="stage2", type=int)
parser.add_argument("--seed", default=8, help="random seed", type=int)
args = parser.parse_args()


method = 'noisy label kmeans network split contrastive'
#writer = SummaryWriter(comment = 'cifar10-noisy_label 0.3'+method)
crf_step = args.stage2 #(15 ) for 0.4
contrastive_step = args.warm_up
exp_times = 3
alpha = 0.7
BatchSize = args.Batch_Size
test_BatchSize = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_crf_acc = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

lr = args.lr



# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
#    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
])

transform_test = transforms.Compose([
#        transforms.Resize((224,224)),
        transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
])


testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_BatchSize, shuffle=False, num_workers=2)
    #==============unbalanced setting===============
train_path = ['./data/data_batch_1','./data/data_batch_2','./data/data_batch_3','./data/data_batch_4'\
              ,'./data/data_batch_5']
test_path = './data/test_batch'


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

#net = ResNet18(num_classes=10)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = DLA()





# Training
def train(epoch,adjust_label,adjust_label2,use_idx):
#    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    c_loss = 0
    con_loss = 0
    correct = 0
    total = 0
    psuedo_correct = 0
    outputs_correct = 0
    
    feats_epoch = np.zeros((50000,64))   # presnet32 FEATURE DIM 64
    targets_arr = np.zeros(50000).astype(np.long)
    outputs_pred = np.zeros(50000)
    outputs_prob = np.zeros(50000)
    noisy_arr = np.zeros(50000)
    real_label_arr = np.zeros(50000)
    contrastive_loss = torch.zeros(1).squeeze().cuda()
    classification_loss = torch.zeros(1).squeeze().cuda()
    
    for batch_idx, (inputs, targets, noisy,real_label,idx) in enumerate(trainloader):
        inputs, targets= inputs.to(device), targets.to(device)
        
        noisy_arr[idx] = noisy.detach().numpy()
        targets_arr[idx] = targets.detach().cpu().numpy()
        real_label = real_label.detach().numpy()
        real_label_arr[idx] = real_label 
        
        optimizer.zero_grad()
        fea,outputs_o = net(inputs)
        outputs = F.softmax(outputs_o,dim=1)
        outputs_arr = outputs.detach().cpu().numpy()
        outputs_prob[idx] = np.max(outputs_arr,axis = 1)
        pred = torch.argmax(outputs,1)
        outputs_pred[idx] = pred.detach().cpu().numpy() 
        
        fea_norm = fea.pow(2).detach().sum(1).pow(1/2).unsqueeze(1)
        fea = (fea.mul(1/(fea_norm+1e-6)))*1
        
        feats_epoch[idx,:] = fea.detach().cpu().numpy()
        outputs_correct += np.sum(np.equal(outputs_pred[idx],real_label))
        if epoch < contrastive_step:                            #pretrain before epoch crf_step
            adjust_label[idx] = targets
            adjust_label2[idx] = targets
            loss = criterion(torch.log(outputs+1e-4),targets)
        else:                                #training with psuedo label from poch crf_step
            if epoch < crf_step:
                adjust_label[idx] = targets
                adjust_label2[idx] = targets
                idx = idx.cpu().numpy()
                c_idx = np.isin(idx,use_idx)
                c_idx = np.arange(targets.shape[0])[c_idx]
                contrastive_loss = 1e-1*cont_loss_func(fea[c_idx], targets[c_idx])
                classification_loss = criterion(torch.log(outputs+1e-6),targets)
                loss = contrastive_loss + classification_loss
            else:
                idx = idx.cpu().numpy()
                c_idx = np.isin(idx,use_idx)
                c_idx = np.arange(targets.shape[0])[c_idx]
                contrastive_loss = 1e-1*cont_loss_func(fea[c_idx], targets[c_idx])
                classification_loss = (1-alpha)*criterion(torch.log(outputs+1e-6),adjust_label[idx]) +\
                    alpha*criterion(torch.log(outputs+1e-6),adjust_label2[idx])
                loss = contrastive_loss + classification_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        c_loss += classification_loss.item()
        con_loss += contrastive_loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(torch.from_numpy(real_label).cuda()).sum().item()
        psuedo_correct += np.sum(targets.cpu().numpy()==real_label)

    adjust_label_arr = adjust_label.cpu().numpy()
    


    pca = PCA(n_components = 32)
    feats_epoch = pca.fit_transform(feats_epoch)
  
    #==============GMM===============
    fea_means = np.zeros((10,32))
    for k in range(10):
        idx = np.argwhere(outputs_pred==k).squeeze()
        fea_means[k,:] = np.mean(feats_epoch[idx,:],axis=0)

    y_pred = KMeans(n_clusters = 10).fit_predict(feats_epoch)   #,init = fea_means
    y_pred = BestMap(outputs_pred, y_pred)


    p_idx2 = np.argwhere((targets_arr != outputs_pred))

    noise_idx = p_idx2.squeeze()
    l_idx = np.setdiff1d(np.arange(50000),noise_idx)
    gmm_label_left = y_pred[l_idx]
    
    
    # select confident sample from kmeans noise samples=======
    feats_epoch_n = feats_epoch[noise_idx,:]
    outputs_pred_n = outputs_pred[noise_idx]
    outputs_prob_n = outputs_prob[noise_idx]
    Clean_idx = []
    for k in range(10):
        k_idx = np.argwhere(outputs_pred_n==k)
#                print('sample class',k_idx.shape[0])
        pred_prob_k = outputs_prob_n[(outputs_pred_n==k)]

        tmp_idx = np.argwhere((pred_prob_k>=np.mean(pred_prob_k))).squeeze()  #&(targets_arr[outputs_pred==k]==k)
        tmp_idx = k_idx[tmp_idx]
        Clean_idx.append(tmp_idx)
    C_len = 0
    for i in range(len(Clean_idx)):
        C_len += Clean_idx[i].shape[0]
    Clean_idx_array = np.zeros(C_len)
    idx = 0
    for i in range(len(Clean_idx)):
        len_i = Clean_idx[i].shape[0]
        Clean_idx_array[idx:(idx+len_i)] = Clean_idx[i].squeeze()
        idx = idx+len_i
    Clean_idx = Clean_idx_array.astype(np.int).squeeze()
    Clean_idx = noise_idx[Clean_idx]
    

    clean_arr = 1-noisy_arr
    correct_clean_num = np.sum(clean_arr[l_idx])
    correct_noisy_num = np.sum(noisy_arr[noise_idx])
    clean_acc = correct_clean_num/l_idx.shape[0]
    noisy_acc = correct_noisy_num/noise_idx.shape[0]
    

#    use_idx = l_idx.squeeze()
    crf_left_correct = np.sum(np.equal(gmm_label_left,real_label_arr[l_idx]))
    left_correct = np.sum(np.equal(adjust_label_arr[l_idx],real_label_arr[l_idx]))
    noise_correct = np.sum(np.equal(adjust_label_arr[noise_idx],real_label_arr[noise_idx]))
    noise_outputs_acc = noise_correct/noise_idx.shape[0]
    
       
    noisy_crf_correct = np.sum(np.equal(y_pred[noise_idx],real_label_arr[noise_idx]))
    left_network_correct = np.sum(np.equal(outputs_pred[l_idx],real_label_arr[l_idx]))
    noise_network_correct = np.sum(np.equal(outputs_pred[noise_idx],real_label_arr[noise_idx]))
    noisy_crf_acc = noisy_crf_correct/noise_idx.shape[0]
    left_crf_acc = crf_left_correct/l_idx.shape[0]
    left_outputs_acc = float(left_correct)/l_idx.shape[0]
    left_network_acc = float(left_network_correct)/l_idx.shape[0]
    noise_network_acc = noise_network_correct/noise_idx.shape[0]

    
    
    total_noisy_num = noise_idx.shape[0]
    use_idx = np.concatenate((l_idx,Clean_idx))
#    adjust_label[use_idx] = torch.from_numpy(outputs_pred[use_idx]).cuda().long()
    

    adjust_label[l_idx] = torch.from_numpy(outputs_pred[l_idx]).cuda().long()
    adjust_label[noise_idx] = torch.from_numpy(y_pred[noise_idx]).cuda().long()
    adjust_label2 = torch.from_numpy(outputs_pred).cuda().long()

    
    
#    total_acc = (left_correct + network_clean_pred)/(clean_sample_num+left_idx.shape[0])
    print('adjust label change',torch.sum(adjust_label!=torch.from_numpy(targets_arr).cuda()))
    total_acc = torch.sum(adjust_label==torch.from_numpy(real_label_arr).cuda().long()).float()/50000
                
    return 100.*correct/total,train_loss,c_loss,con_loss,noisy_crf_acc,noisy_acc,clean_acc,total_noisy_num,left_outputs_acc,noise_outputs_acc,\
noise_network_acc,total_acc,left_crf_acc,left_network_acc,adjust_label,adjust_label2,l_idx


def test(epoch,adjust_label,best_acc):
    global best_crf_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    AUC = 0
    Labels_one_hot = np.zeros((test_BatchSize*len(testloader),10))
    Outputs_arr = np.zeros((test_BatchSize*len(testloader),10))
    with torch.no_grad():
        correct_class = torch.zeros(10).cuda()
        num_class = torch.zeros(10).cuda()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            class_pro = torch.zeros(10).cuda()
            for c in range(10):
                num = torch.sum(targets==c)
                class_pro[c] = 1/(num.float()+1e-4)
            class_pro = class_pro.repeat(inputs.shape[0],1)
            _,outputs = net(inputs)
            outputs = F.softmax(outputs,dim=1)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            Labels_one_hot[batch_idx*test_BatchSize:(batch_idx+1)*test_BatchSize,:] = torch.zeros(test_BatchSize, 10).scatter_(1, (targets.cpu().view(-1,1)), 1).numpy()
            Outputs_arr[batch_idx*test_BatchSize:(batch_idx+1)*test_BatchSize,:] = outputs.detach().cpu().numpy()


    AUC = roc_auc_score(Labels_one_hot,Outputs_arr) 

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc :
        print('Saving..')
        print('=======================\n')
        print('best performance epoch:{}'.format(epoch))
        for p in optimizer.param_groups:
            learning_rate = p['lr']
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'adjust label':adjust_label,
            'batch size':BatchSize,
            'learning rate':learning_rate
        }
        f = open('result_cifar10_contrastive.txt','a')
        f.write('best epoch:{},total acc:{},AUC:{},,lr:{},BatchSize:{}\n'\
                .format(epoch,acc,AUC,args.lr,BatchSize))
        f.close()
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt'+method+str(noisy_rate)+".pth")
        best_acc = acc
    return acc,AUC,best_acc


f = open('result_'+'method'+'.txt','a')
f.write('Start '+method+' experiment'+str(datetime.now)+'\n')
f.close()

#file = open('RESULT_cifar10_%.1f.txt'%noisy_rate,'a')
#file.write('Start'+method+'experiment\n')
#file.close()

noise_type=args.noise_type
use_idx = np.arange(50000)

Best_ACC = np.zeros((5,exp_times))
Last_ACC = np.zeros((5,exp_times))
kmeans_clean = np.zeros((5,exp_times))
kmeans_noisy = np.zeros((5,exp_times))
network_clean = np.zeros((5,exp_times))
network_noisy = np.zeros((5,exp_times))
adjust_label = torch.zeros(50000).cuda().long()
adjust_label2 = torch.zeros(50000).cuda().long()

    # noisy_rate = 0.1*rate
noisy_rate = args.noise_level
trainloader = Make_Loader(train_path,noise_type,transform_train,BatchSize,True,noisy_rate)
for exp in range(exp_times):
    net = preact_resnet32_cifar()
    #net = ResNet34(num_classes=10)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

#    if args.resume:
#    # Load checkpoint.
#    print('==> Resuming from checkpoint..')
#    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#    checkpoint = torch.load('./checkpoint/ckpt.pth')
#    net.load_state_dict(checkpoint['net'])
#    best_acc = checkpoint['acc']
#    start_epoch = checkpoint['epoch']
#    adjust_label = checkpoint['adjust label']
#    BatchSize = checkpoint['batch size']
#        lr = checkpoint['learning rate']

    criterion = nn.NLLLoss()
    temperarure = 0.05
    cont_loss_func = losses.NTXentLoss(temperarure)
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    nepoch = args.nepoch
    best_acc = 0
    for epoch in range(nepoch):
        start_time = datetime.now()
        acc,train_loss,c_loss,con_loss,noisy_kmeans_acc,noisy_acc,clean_acc,total_noisy_num,left_outputs_acc,noise_outputs_acc,\
    noise_network_acc,total_acc,clean_kmeans_acc,clean_network_acc,adjust_label,adjust_label2,use_idx \
        = train(epoch,adjust_label,adjust_label2,use_idx)
        end_time = datetime.now()
        print('=================noisy rate:{}=============='.format(0.1*noisy_rate))
        print('=====================exp:{}============='.format(exp))
        print('====================epoch:{}=================='.format(epoch))
        print('classification loss:{}'.format(c_loss))
        print('contrastive loss:{}'.format(con_loss))
        print('train epoch time:{}s'.format((end_time-start_time).seconds))
        print('Train Accuracy(real label):{}%'.format(acc))
        print('total acc:{}'.format(total_acc))
        test_acc,AUC,best_acc = test(epoch,adjust_label,best_acc)
        print('current test accuracy:{}%'.format(test_acc))
        print("AUC:{}".format(AUC))
        print('best test accuracy:{}%'.format(best_acc))

        if test_acc == best_acc:
            Best_ACC[noisy_rate-1,exp] = best_acc
            kmeans_clean[noisy_rate-1,exp] = clean_kmeans_acc
            kmeans_noisy[noisy_rate-1,exp] = noisy_kmeans_acc
            network_clean[noisy_rate-1,exp] = clean_network_acc
            network_noisy[noisy_rate-1,exp] = noise_network_acc

        if epoch == 20:
            for p in optimizer.param_groups:
                p['lr'] = 0.01
        if epoch == 40:                     #start training with psuedo label
            for p in optimizer.param_groups:
                p['lr'] = 0.001
    Last_ACC[noisy_rate-1,exp] = test_acc
#    if epoch == crf_step-1:
#        for p in optimizer.param_groups:
#            p['lr'] = 0.0005
#    scheduler.step()
print('Best acc kmeans:')
for exp in range(exp_times):
    print('noise rate:{},exp:{},best acc:{}'.format(0.1*(noisy_rate),exp,Best_ACC[noisy_rate,exp]))
print('Last acc kmeans:')
for exp in range(exp_times):
    print('noise rate:{},exp:{},Last acc:{}'.format(0.1*(noisy_rate),exp,Last_ACC[noisy_rate,exp]))

results_file_name_best = './result/best' + noise_type+ str(noisy_rate)+'.csv'
results_file_name_last = './result/last' + noise_type+ str(noisy_rate)+'.csv'
np.savetxt(results_file_name_best,Best_ACC,delimiter=',')
np.savetxt(results_file_name_last,Last_ACC,delimiter=',')
