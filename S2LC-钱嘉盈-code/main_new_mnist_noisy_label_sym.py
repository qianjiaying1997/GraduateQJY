'''Train CIFAR10 with PyTorch.'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from tensorboardX import SummaryWriter
from sklearn.cluster import KMeans
from pytorch_metric_learning import losses

import os
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
import mnist_reader
# import model_utils
from sklearn.decomposition import PCA 
from BestMap import *
from resnet import ResNet18

# from utils import progress_bar
#np.random.seed(1)
#torch.manual_seed(1)
method = 'kmeans fmnist'
mat_path = ['./data/data_batch_1','./data/data_batch_2','./data/data_batch_3','./data/data_batch_4'\
              ,'./data/data_batch_5']
#writer = SummaryWriter(comment = method)
crf_step = 30  #55
alpha = 0.5
cont_step = 20  #25
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class ImageSet_dataset(Dataset):
    def __init__(self, mat_path,transform = None, train = True,noise_rate = 0.2):
        # Exp_Type must be 'Train', 'Test' or 'Valid'
        x_train, t_train = mnist_reader.load_mnist('data/fashion', kind='train')
        x_test, t_test = mnist_reader.load_mnist('data/fashion', kind='t10k')     
        if train == True:
            Img = x_train
            Labels = torch.from_numpy(t_train).long()
            Noisy = torch.zeros(60000).long()
            real_Labels = Labels.clone().long()
            
            #=================asymmetric noisy label===============
            for i in range(60000):
                r = np.random.random()
                if r < noise_rate:
                    if Labels[i] == 9:
                        Labels[i] = 7
                        Noisy[i] = 1
                    elif Labels[i] == 2:
                        Labels[i] = 0
                        Noisy[i] = 1
                    elif Labels[i] == 7:
                        Labels[i] = 5
                        Noisy[i] = 1
                    elif Labels[i] == 4:
                        Labels[i] = 3
                        Noisy[i] = 1
                    elif Labels[i] == 3:
                        Labels[i] = 4
                        Noisy[i] = 1

         #=============================symmetric noisy label===================
            # for i in range(60000):
            #     if np.random.random() < noise_rate:
            #         Labels[i] = np.random.randint(0, 10)
            #         Noisy[i] = 1
        else:
            Img = x_test
            Labels = torch.from_numpy(t_test)
            real_Labels = torch.from_numpy(t_test)
            Noisy = torch.zeros(10000)
        
            
        self.transform = transform
        self.img = Img
        self.label = Labels.long()
        self.noisy = Noisy.long()
        self.real_label = real_Labels.long()
        del Labels, Img , Noisy,real_Labels
        
    def __getitem__(self, idx):
        img_data = self.img[idx,:]
        Batch_data = np.zeros([28,28])
        Batch_data[:,:] = img_data.reshape(28,28)
        Batch_data = Image.fromarray(Batch_data.astype('uint8'))
        Batch_label = self.label[idx]
        Batch_noisy = self.noisy[idx]
        Batch_real_label = self.real_label[idx]
        if self.transform is not None:
            Batch_data = self.transform(Batch_data)
        
        return Batch_data, Batch_label,Batch_noisy,Batch_real_label,idx
    
    def __len__(self):
        return len(self.label)



    
def Make_Loader(mat_path, transform = None,Batch_size = 128,train = True,noise_rate = 0.2):
    data_set = ImageSet_dataset(mat_path,transform,train,noise_rate)
    new_loader = DataLoader(data_set, Batch_size, shuffle=True, num_workers=2)
    return new_loader




parser = argparse.ArgumentParser(description='PyTorch mnist Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

BatchSize = 200
test_BatchSize = 250

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
adjust_label = torch.zeros(60000).cuda().long()
lr = args.lr

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
#    transforms.Resize((224,224)),
    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
])

transform_test = transforms.Compose([
#        transforms.Resize((224,224)),
        transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
])
    
#trainset = torchvision.datasets.CIFAR10(
#    root='./data', train=True, download=True, transform=transform_train)
#trainloader = torch.utils.data.DataLoader(
#    trainset, batch_size=128, shuffle=True, num_workers=2)
#
testset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=test_BatchSize, shuffle=False, num_workers=2)
    #==============unbalanced setting===============
train_path = ['./data/data_batch_1','./data/data_batch_2','./data/data_batch_3','./data/data_batch_4'\
              ,'./data/data_batch_5']
test_path = './data/test_batch'
    

#testloader = Make_Loader(test_path,transform_test,test_BatchSize,False)


# Model
print('==> Building model..')

#net = LeNet()



# Training
def train(epoch,adjust_label,adjust_label2,use_idx):
#    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    outputs_correct = 0
    
    feats_epoch = np.zeros((60000,512))   # lenet FEATURE DIM 500
    targets_arr = np.zeros(60000).astype(np.long)
    outputs_pred = np.zeros(60000)
    outputs_prob = np.zeros(60000)
    noisy_arr = np.zeros(60000)
    real_label_arr = np.zeros(60000)
    
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
        fea = (fea.mul(1/(fea_norm+1e-6)))*1e1
        
        feats_epoch[idx,:] = fea.detach().cpu().numpy()
        outputs_correct += np.sum(np.equal(outputs_pred[idx],real_label))
#        if epoch < crf_step:                           
#            loss = criterion(torch.log(outputs+1e-4),targets)
#        else:
#            classification_loss = (1-alpha)*criterion(torch.log(outputs+1e-6),adjust_label[idx]) + \
#            alpha*criterion(torch.log(outputs+1e-6),adjust_label2[idx])
#            loss = classification_loss
        
        
        
        if epoch < cont_step:                           
            loss = criterion(torch.log(outputs+1e-4),targets)
        else:
            idx = idx.cpu().numpy()
            c_idx = np.isin(idx,use_idx)
            c_idx = np.arange(targets.shape[0])[c_idx]
            contrastive_loss = 1e-1*cont_loss_func(fea[c_idx], targets[c_idx])
            if epoch < crf_step:
                classification_loss = criterion(torch.log(outputs+1e-6),targets)
                loss = contrastive_loss + classification_loss
            else:
                classification_loss = (1-alpha)*criterion(torch.log(outputs+1e-6),adjust_label[idx]) + \
                alpha*criterion(torch.log(outputs+1e-6),adjust_label2[idx])
                loss = classification_loss + contrastive_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(torch.from_numpy(real_label).cuda()).sum().item()
        
        # if epoch>13:
        #     progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d) '\
        #                 % (train_loss/(batch_idx+1),100.*correct/total, correct,total))       
                
       
    pca = PCA(n_components = 16)
    feats_epoch = pca.fit_transform(feats_epoch)
    
    y_pred = KMeans(n_clusters = 10).fit_predict(feats_epoch)   #,init = fea_means
    y_pred = BestMap(outputs_pred, y_pred)
    
    p_idx2 = np.argwhere((targets_arr != outputs_pred))
    
    noise_idx = p_idx2.squeeze()
    l_idx = np.setdiff1d(np.arange(60000),noise_idx)
    
    gmm_label_left = y_pred[l_idx]
    crf_left_correct = np.sum(np.equal(gmm_label_left,real_label_arr[l_idx]))
    left_correct = np.sum(np.equal(targets_arr[l_idx],real_label_arr[l_idx]))
    noise_correct = np.sum(np.equal(targets_arr[noise_idx],real_label_arr[noise_idx]))
    noise_outputs_acc = noise_correct/noise_idx.shape[0]
    noisy_crf_correct = np.sum(np.equal(y_pred[noise_idx],real_label_arr[noise_idx]))
    left_network_correct = np.sum(np.equal(outputs_pred[l_idx],real_label_arr[l_idx]))
    noise_network_correct = np.sum(np.equal(outputs_pred[noise_idx],real_label_arr[noise_idx]))
    noisy_crf_acc = noisy_crf_correct/noise_idx.shape[0]
    left_crf_acc = crf_left_correct/l_idx.shape[0]
    left_outputs_acc = left_correct/l_idx.shape[0]
    left_network_acc = left_network_correct/l_idx.shape[0]
    noise_network_acc = noise_network_correct/noise_idx.shape[0]
    
    
    adjust_label[l_idx] = torch.from_numpy(outputs_pred[l_idx]).cuda().long()
    adjust_label[noise_idx] = torch.from_numpy(y_pred[noise_idx]).cuda().long()
    adjust_label2 = torch.from_numpy(outputs_pred).cuda().long()
    
    
    acc = torch.sum(adjust_label == torch.from_numpy(real_label_arr).cuda().long()).float()/60000
    t_acc = float(np.sum(np.equal(targets_arr,real_label_arr)))/60000

    return 100.*correct/total,train_loss,noise_outputs_acc,noisy_crf_acc,noise_network_acc,\
left_outputs_acc,left_crf_acc,left_network_acc,adjust_label,adjust_label2,l_idx,acc,t_acc


def test(epoch,best_acc):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
           
            _,outputs = net(inputs)
            outputs = F.softmax(outputs,dim=1)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
           
            

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
            'adjust label':adjust_label,
            'batch size':BatchSize,
            'learning rate':learning_rate
        }
        f = open('result_Fmnist_NoisyLabel.txt','a')
        f.write('best epoch:{},total acc:{}\n'\
                .format(epoch,acc))
        f.close()
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_fmnist_sym.pth')
        best_acc = acc
    return acc,best_acc


f = open('result_Fmnist_NoisyLabel.txt','a')
f.write('Start %s experiment\n'%method)
f.close()

best_acc_exp = np.zeros((4,3))
last_acc_exp = np.zeros((4,3))
idx = 0
for noise_rate in np.arange(0.1,0.5,0.1):
    for exp in range(3):
        trainloader = Make_Loader(train_path,transform_train,BatchSize,True,noise_rate)
        net = ResNet18()

        net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        
        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt_fmnist_0.4_resnet18_epoch26.pth')
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
            adjust_label = checkpoint['adjust label']
            BatchSize = checkpoint['batch size']
            lr = checkpoint['learning rate']
        
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4)
        temperarure = 0.05
        cont_loss_func = losses.NTXentLoss(temperarure)
        best_acc = 0
        adjust_label = torch.zeros(60000).cuda().long()
        adjust_label2 = torch.zeros(60000).cuda().long()
        use_idx = np.arange(60000)
        for epoch in range(start_epoch, start_epoch+100):
            start_time = datetime.now()
            acc,train_loss,noise_outputs_acc,noisy_kmeans_acc,noise_network_acc,\
        left_outputs_acc,clean_kmeans_acc,clean_network_acc,adjust_label,adjust_label2,use_idx,p_acc,t_acc = train(epoch,adjust_label,adjust_label2,use_idx)
            end_time = datetime.now()
            print('================noise rate:{}=========='.format(noise_rate))
            print('=====================exp:%d================='%exp)
            print('====================epoch:{}=================='.format(epoch))
            print('train epoch time:{}s'.format((end_time-start_time).seconds))
            print('Train Accuracy:{}%'.format(acc))
            print('Train loss:{}'.format(train_loss))
        #    writer.add_scalar('train loss',train_loss,epoch)
            # print('clean targets acc:{}'.format(left_outputs_acc))
            # print('clean kmeans acc:{}'.format(clean_kmeans_acc))
            # print('clean network acc:{}'.format(clean_network_acc))
            # print('kmeans prediction acc on noisy sample:{}'.format(noisy_kmeans_acc)) 
            # print('noise network acc:{}'.format(noise_network_acc))
            # print('noise target acc:{}'.format(noise_outputs_acc))
            # print('psuedo label acc:{}'.format(p_acc))
            # print('targets label acc:{}'.format(t_acc))
            test_acc,best_acc = test(epoch,best_acc)
            best_acc_exp[idx,exp] = best_acc
        #    writer.add_scalar('test acc',test_acc,epoch)
            print('current test accuracy:{}%'.format(test_acc))
            print('best test accuracy:{}%'.format(best_acc))
            if epoch == 20:
                for p in optimizer.param_groups:
                    p['lr'] = 0.01
            if epoch == 30:                     #start training with psuedo label
                for p in optimizer.param_groups:
                    p['lr'] = 0.001
        last_acc_exp[idx,exp] = test_acc
    idx += 1
for noise in range(4):
    for exp in range(3):
        print('Asym noise rate:{},exp:{}:'.format(np.arange(0.1,0.5,0.1)[noise],exp))
        print('best acc:{}'.format(best_acc_exp[noise,exp]))
        print('laet acc:{}'.format(last_acc_exp[noise,exp]))
np.savetxt('./result/f_mnist_asym_best_acc.csv',best_acc_exp)
np.savetxt('./result/f_mnist_asym_last_acc.csv',last_acc_exp)