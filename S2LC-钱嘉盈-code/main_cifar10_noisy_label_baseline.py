'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import argparse
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from datetime import datetime
import math
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
#import model_utils
from BestMap import *


#from models import *
from preact_resnet_cifar10 import preact_resnet32_cifar
#from utils import progress_bar
import math 
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#random.seed(1)
#np.random.seed(1)
#torch.manual_seed(1)
method = 'noisy label crossent preact resnet32 baseline'
#writer = SummaryWriter(comment = 'cifar10-noisy-label 0.3 '+method)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



class ImageSet_dataset(Dataset):
    def __init__(self, mat_path,transform = None, train = True,noise_rate = 0.1):
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

#            Img = np.zeros([50000,3,1024])
#            Labels = torch.zeros(50000)
#            Noisy = torch.zeros(50000)
#            real_Labels = torch.zeros(50000)
#            idx=0 
            
            
            Img = img
            Labels = torch.from_numpy(labels)
            real_Labels = torch.from_numpy(labels)
            Noisy = torch.zeros(50000)
            
            
            #==============asymmetric noise================
            
            
            for i in range(50000):
                if np.random.random() < noise_rate:
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
#            for c in range(10):
#                idx_s = np.argwhere(labels==c).squeeze()
#                num_sam = idx_s.shape[0]
#                noise_sam = int(num_sam*0.4)
#                Img[idx:(idx+num_sam),:,:] = img[idx_s,:,:]
#                noise_label = np.random.choice(10,noise_sam,replace = False)
#                Labels[idx:(idx+noise_sam)] = torch.from_numpy(noise_label)
#                Labels[(idx+noise_sam):(idx+num_sam)] = c*torch.ones(num_sam-noise_sam)
#                Noisy[idx:(idx+noise_sam)] = torch.ones(noise_sam)
#                Noisy[(idx+noise_sam):(idx+num_sam)] = torch.zeros(num_sam-noise_sam)
#                real_Labels[idx:(idx+num_sam)] = c*torch.ones(num_sam)
#                idx = idx+num_sam
                
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



    
def Make_Loader(mat_path, transform = None,Batch_size = 256,train = True,noise_rate = 0.1):
    data_set = ImageSet_dataset(mat_path,transform,train,noise_rate)
    new_loader = DataLoader(data_set, Batch_size, shuffle=True, num_workers=2)
    return new_loader





parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')  # 0.1 for pretrain in the begining 0.1 decay every 5 epochs 
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

BatchSize = 128   #800 for densecrf 128 for pretrain
test_BatchSize = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lr = args.lr

#net = VGG('VGG11')
#net = preact_resnet32_cifar()
##lmcl_loss = model_utils.LMCL_loss(num_classes=10, feat_dim=512,m=0.4,s=30)
##net = PreActResNet34()
#net = net.to(device)
##lmcl_loss = lmcl_loss.to(device)
#if device == 'cuda':
#    net = torch.nn.DataParallel(net)
##    lmcl_loss = torch.nn.DataParallel(lmcl_loss)
#    cudnn.benchmark = True
#if args.resume:
#    # Load checkpoint.
#    print('==> Resuming from checkpoint..')
#    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#    checkpoint = torch.load('./checkpoint/ckpt_baseline.pth')
#    net.load_state_dict(checkpoint['net'])
#    best_acc = checkpoint['acc']
#    start_epoch = checkpoint['epoch']
#    BatchSize = checkpoint['batch size']
#    lr = checkpoint['learning rate']




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
    
#trainset = torchvision.datasets.CIFAR10(
#    root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(
#    trainset, batch_size=BatchSize, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_BatchSize, shuffle=False, num_workers=2)
    #==============unbalanced setting===============
train_path = ['./data/data_batch_1','./data/data_batch_2','./data/data_batch_3','./data/data_batch_4'\
              ,'./data/data_batch_5']
test_path = './data/test_batch'
    

#testloader = Make_Loader(test_path,transform_test,test_BatchSize,False)

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




#criterion = nn.NLLLoss()
#optimizer = optim.SGD(net.parameters(), lr=lr,
#                      momentum=0.9, weight_decay=5e-4)
#optimzer4center = optim.SGD(lmcl_loss.parameters(), lr=0.01)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [5,10,100], gamma = 0.1)


# Training
def train(epoch):
#    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    

#    feats_epoch = np.zeros((50000,64))
#    targets_arr = np.zeros(50000)
#    outputs_pred = np.zeros(50000)
    for batch_idx, (inputs, targets, noisy,real_label,idx) in enumerate(trainloader):
#    for batch_idx,(inputs,targets) in enumerate(trainloader):
        inputs, targets= inputs.to(device), targets.to(device)
        real_label = real_label.to(device)
#        targets_arr[idx] = targets.cpu().numpy()

        
        optimizer.zero_grad()
#        optimzer4center.zero_grad()
        fea,outputs_o = net(inputs)
        fea_norm = fea.pow(2).detach().sum(1).pow(1/2).unsqueeze(1)
        fea = (fea.mul(1/(fea_norm+1e-6)))*1e1
#        feats_epoch[idx] = fea.detach().cpu().numpy()
        outputs = F.softmax(outputs_o,dim=1)
        _, predicted = outputs.max(1)
#        outputs_pred[idx] = predicted.detach().cpu().numpy()
        
        fea_norm = fea.pow(2).detach().sum(1).pow(1/2).unsqueeze(1)
        fea = (fea.mul(1/(fea_norm+1e-6)))*1e1

        loss = criterion(torch.log(outputs+1e-4),targets)   #crossent
        #================crossent variants==============
#        logits, mlogits = lmcl_loss(fea, targets)
#        loss = criterion(torch.log(mlogits+1e-4), targets)
        #================crossent variants===============
        loss.backward()
        optimizer.step()
#        optimzer4center.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(real_label).sum().item()
#    pca = PCA(n_components = 16)
#    feats_epoch = pca.fit_transform(feats_epoch)
#    y_pred = KMeans(n_clusters = 10).fit_predict(feats_epoch)   #,init = fea_means
#    y_pred = BestMap(targets_arr, y_pred)
#    cluster_acc = np.sum(np.equal(y_pred,targets_arr))/50000
       
       
        
#    return 100.*correct/total,100.*cluster_acc,train_loss
    return 100.*correct/total,train_loss


def test(epoch,best_acc):
#    global best_acc
    net.eval()
    correct = 0
#    correct_min = 0
#    correct_maj = 0
    total = 0
#    total_min = 0
#    total_maj = 0
    AUC = 0
    Labels_one_hot = np.zeros((test_BatchSize*len(testloader),10))
    Outputs_arr = np.zeros((test_BatchSize*len(testloader),10))
    with torch.no_grad():
#        correct_class = torch.zeros(10).cuda()
#        num_class = torch.zeros(10).cuda()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
#            targets_arr = targets.detach().cpu().numpy()
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
            
            
#            target_arr = targets.detach().cpu().numpy()
#            for c in range(10):
#                idx = np.argwhere(target_arr==c)
#                if idx.shape[0]>1:
#                    idx = np.squeeze(idx)
#                correct_class[c] += (predicted[idx].eq(targets[idx]).sum())
#                num_class[c] += idx.shape[0]
#                if c%2 == 0:
#                    total_min += idx.shape[0]
#                    correct_min += predicted[idx].eq(targets[idx]).sum().item()
#                else:
#                    total_maj += idx.shape[0]
#                    correct_maj += predicted[idx].eq(targets[idx]).sum().item()
#    correct_class = correct_class/num_class
#    G_means = torch.prod(torch.pow(correct_class,1/10))
    AUC = roc_auc_score(Labels_one_hot,Outputs_arr) 
#    acc_min = 100.*correct_min/total_min
#    acc_maj = 100.*correct_maj/total_maj

#            progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
#                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        print('=======================\n')
        print('best performance epoch:{}'.format(epoch))
        for p in optimizer.param_groups:
            learning_rate = p['lr']
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'batch size':BatchSize,
            'learning rate':learning_rate
        }
        f = open('result_cifar10.txt','a')
        f.write('best epoch:{},total acc:{},AUC:{},lr:{},BatchSize:{}\n'\
                .format(epoch,acc,AUC,args.lr,BatchSize))
        f.close()
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_baseline.pth')
        best_acc = acc
    return acc,AUC,best_acc


f = open('result_cifar10.txt','a')
f.write('Start '+method+' experiment'+str(datetime.now)+'\n')
f.close()

Best_ACC = np.zeros((5,3))
Train_ACC = np.zeros((5,3))
Last_ACC = np.zeros((5,3))

for rate in range(1,6):
    noise_rate = 0.1*rate
    
    trainloader = Make_Loader(train_path,transform_train,BatchSize,True,noise_rate)
    for exp in range(3):
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
        optimizer = optim.SGD(net.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4)
        best_acc = 0
        for epoch in range(start_epoch, start_epoch+100):
            start_time = datetime.now()    
            acc,train_loss = train(epoch)
            end_time = datetime.now()
#            writer.add_scalar('train loss',train_loss,epoch)
            print('=========noise rate:{}========'.format(noise_rate))
            print('===========exp:{}================='.format(exp))
            print('====================epoch:{}=================='.format(epoch))
            print('train epoch time:{}s'.format((end_time-start_time).seconds))
            print('Train Accuracy:{}%'.format(acc))
            print('Train loss:{}'.format(train_loss))
            test_acc,AUC,best_acc = test(epoch,best_acc)
            Best_ACC[rate-1,exp] = best_acc
            if test_acc == best_acc:
                Train_ACC[rate-1,exp] = acc
#            writer.add_scalar('test acc',test_acc,epoch)
            print('current test accuracy:{}%'.format(test_acc))
            print("AUC:{}".format(AUC))
            print('best test accuracy:{}%'.format(best_acc))
#            if epoch == 20:
#                for p in optimizer.param_groups:
#                    p['lr'] = 0.01
#            if epoch == 40:
#                for p in optimizer.param_groups:
#                    p['lr'] = 0.001
        Last_ACC[rate-1,exp] = test_acc
print('Best acc baseline:')
for rate in range(5):
    for exp in range(3):
        print('noise rate:{},exp:{},best acc:{}'.format(0.1*(rate+1),exp,Best_ACC[rate,exp]))
print('Last acc baseline:')
for rate in range(5):
    for exp in range(3):
        print('noise rate:{},exp:{},last acc:{}'.format(0.1*(rate+1),exp,Last_ACC[rate,exp]))
print('Train acc baseline:')
for rate in range(5):
    for exp in range(3):
        print('noise rate:{},exp:{},train acc:{}'.format(0.1*(rate+1),exp,Train_ACC[rate,exp]))
np.savetxt('./result/best_acc_baseline.csv',Best_ACC)
np.savetxt('./result/train_acc_baseline.csv',Train_ACC)
np.savetxt('./result/last_acc_baseline.csv',Last_ACC)
np.savetxt('best_acc_baseline.csv',Best_ACC,)
#    scheduler.step()
