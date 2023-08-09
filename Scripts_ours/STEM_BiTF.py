import numpy as np
import torch
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import DataLoader, Dataset

import torch.optim as optim
import time
import os
import argparse
from typing import Any,Callable,Dict,IO,Optional,Tuple,Union
import pandas as pd
import cv2
from PIL import Image
import subprocess
import collections
import matplotlib.pyplot as plt

import math
import torchvision.models as models
from torch import autograd

from torch.utils.tensorboard import SummaryWriter

cuda=torch.cuda.is_available()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class mylog():
    def __init__(self, log_path):
        self.log_path = log_path
    def save_log(self, txt):
        with open(self.log_path, 'a') as file:
            print(txt, file=file)

class ClassifyAccuracy(nn.Module):
    def __init__(self):
        super(ClassifyAccuracy,self).__init__()
        self.name = 'ClassifyAccuracy'
    
    def forward(self,output, target):
        acc = torch.sum(output == target).float()
        acc /= torch.tensor(len(output), dtype=float)
        return acc
    
class GarOFV_Video_Dataset(Dataset):
    def __init__(self,train_mode:str='train',transform:Optional[Callable]=None,target_transform:Optional[Callable]=None,opt=1,par=None)->None:
        super(GarOFV_Video_Dataset,self).__init__()
        self.train=train_mode
        self.img_path=par.img_path
        if self.train=='train':
            data_file=par.train_file
        elif self.train == 'val':
            data_file=par.vali_file
        elif self.train == 'test':
            data_file=par.test_file

        data=pd.read_csv(data_file)
        self.names=data.iloc[:,0]
        self.targets=data.iloc[:,opt]
        self.video_labels=data.iloc[:,3]

        self.transform=transform
        self.target_transform=target_transform
        self.rgb_depth = par.rgb_depth
        self.use_flow = par.use_flow
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        imgs_path=os.path.join(self.img_path, self.names[index])
        target=int(self.targets[index])
        video_label = self.video_labels.iloc[index]
        
        img_pair = []
        for img_i in range(len(os.listdir(imgs_path))):
            img_name = os.path.join(imgs_path, str(img_i+1).zfill(4)+'.png')
            if self.rgb_depth == 'depth':
                img=cv2.imread(img_name,0)
                img=Image.fromarray(img,mode='L')
            if self.rgb_depth == 'rgb':
                img=cv2.imread(img_name,-1)
                img=Image.fromarray(img,mode='RGB')

            if self.transform is not None:
                img=self.transform(img)
            if self.target_transform is not None:
                target=self.target_transform(target)
            if self.train:
                noise=0.01*torch.rand_like(img)
                img=img+noise
            img_pair.append(img)

        flow = None    
        if self.use_flow:
            flow = torch.load(os.path.join('./flow_256', str(video_label)+'.pth' ))
        img_pair = torch.stack(img_pair, dim=0)
        
        return img_pair, target-1, video_label, flow
    
    def __len__(self):
        return len(self.names)


class ConvGRU_2d(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=66):
        super(ConvGRU_2d, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        
    def forward(self, x, h):
        if h is None:
            state_size = [x.shape[0], self.hidden_dim, x.shape[2], x.shape[3]]
            if torch.cuda.is_available():
                h = Variable(torch.zeros(state_size)).cuda()
            else:
                h = Variable(torch.zeros(state_size))
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return q, h
    
class Fusion_Cell(nn.Module):
    def __init__(self, input_dim=512, num_classes=5):
        super(Fusion_Cell, self).__init__()
        self.feat_chn = input_dim
        self.conv = nn.Sequential(
            nn.Conv2d(self.feat_chn * 2, self.feat_chn, 1, padding=0),
            nn.BatchNorm2d(self.feat_chn),
            nn.ReLU(),
            nn.Conv2d(self.feat_chn, self.feat_chn, 3, padding=1),
            nn.BatchNorm2d(self.feat_chn),
            nn.ReLU(),
            nn.Conv2d(self.feat_chn, self.feat_chn, 1, padding=0),
            nn.BatchNorm2d(self.feat_chn),
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(self.feat_chn * 2, self.feat_chn, 1, padding=0),
            nn.BatchNorm2d(self.feat_chn),
        )

        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def classify(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def forward(self, feat_list, pred_list):
        feat_cat = torch.cat(feat_list, dim=1)# [BS, 512*2, 8, 8]
        feat_out = self.relu(self.conv(feat_cat) + self.shortcut(feat_cat))

        pred = self.classify(feat_out)
        return feat_out, pred

class STEM_BiTF(nn.Module):
    def __init__(self, img_chs,num_classes, rgb_pretrained_model_path) -> None:
        super(STEM_BiTF,self).__init__()
        self.resnet = models.resnet18(pretrained=False)# [BS, 512, 8, 8]
        # self.resnet = torch.load(os.path.join(rgb_pretrained_model_path, 'last_epoch30.pth'))
        # self.resnet = 
        # self.resnet = torch.load(os.path.join(rgb_pretrained_model_path, 'best_1.pth'))
        self.num_classes = num_classes
        conv1_out_features = self.resnet.conv1.out_channels
        self.resnet.conv1 = torch.nn.Conv2d(img_chs+2, conv1_out_features, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = torch.nn.Linear(512, self.num_classes, bias=True)
        
        self.myConvGRU_1 = ConvGRU_2d(64, 66)
        self.resize_fun = torch.nn.Upsample(scale_factor=1/4, mode='bilinear')
        
        self.weight_fusion = Fusion_Cell(512, self.num_classes)

    def feature_extract(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        return x
    
    def feature_extract_high(self, x):
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x
    
    def classify(self, x):
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x
    
    def forward(self,x, target, video_label, flow):
        hidden = None
        power_num = math.ceil(math.log(x.size(1), 2))
        feat_high_list = []
        y_list = []
        ###### STEM
        for t in range(x.size(1)):
            current_frame = Variable(x[:, t, :, :, :], requires_grad=True)
            if t == 0:
                flow_frame = torch.zeros([x.size(0), 2, x.size(3), x.size(4)]).to(device)
            else:
                flow_frame = flow[:, t-1, :, :, :].to(device)# [BS, 59, 2, 64, 64] -> [BS, 2, H, W]
            flow_frame_resize = self.resize_fun(flow_frame)

            input_concat = torch.cat([current_frame, flow_frame], dim=1)# [B, 3+2, 256, 256]
            output_feat = self.feature_extract(input_concat)  # [BS, 5, H, W] -> [BS, 64, 64, 64]

            feat_concat = torch.cat([output_feat, flow_frame_resize], dim=1)# [BS, 64+2, 64, 64]
            output_feature, hidden = self.myConvGRU_1(feat_concat, hidden)

            feat_high = self.feature_extract_high(output_feature)
            feat_high_list.append(feat_high)
            
            y = self.classify(feat_high)#[BS, 5]
            conf = torch.softmax(y, dim=1)# pred = torch.argmax(conf, dim=1)
            y_list.append(conf)


        tensor_zeros = torch.zeros_like(feat_high)
        fusion_feat_list, fusion_y_list = [], []
        for i in range(len(feat_high_list)):
            fusion_feat_list.append(feat_high_list[i].clone().detach())
            fusion_y_list.append(y_list[i].clone().detach())

        ###### BiTF
        grad_each_frame = []
        for forest_i in range(power_num):
            pair_mod = len(fusion_feat_list) % 2 # 0 or 1
            pair_num = (len(fusion_feat_list) + pair_mod) // 2# 64 / 2 = 32, 15%2=1, (15+1)//2=8

            if pair_mod == 1:
                fusion_feat_list.append(tensor_zeros)

            feature_num = len(fusion_feat_list)
            output_fusion_feat_list, output_fusion_y_list = [None] * pair_num, [None] * pair_num

            for index in range(0, feature_num, 2):
                output_fusion_feat_list[index // 2], output_fusion_y_list[index // 2] = self.weight_fusion(fusion_feat_list[index:index+2], fusion_y_list[index:index+2])

            fusion_feat_list, fusion_y_list = output_fusion_feat_list, output_fusion_y_list


        y = fusion_y_list[0]
        return y

def train_epoch(train_loader,model,loss_fn,optimizer,cuda,log_interval,metrics,accuracy_metric, logger=None):
    for metric in metrics:
        metric.reset()
    model.train()
    total_loss=0
    mean_acc = 0
    mean_acc_weight = 0

    for batch_idx,(data,target, video_label, flow) in enumerate(train_loader):
        target=target if len(target)>0 else None

        if cuda:
            data = data.cuda()
            if target is not None:
                target=target.cuda().long()
        
        optimizer.zero_grad()
        outputs=model(data, target, video_label, flow)


        loss_outputs=loss_fn(outputs, target)# + loss_fn(outputs, target_weight)
        
        total_loss+=loss_outputs.item()
        loss_outputs.backward()
        optimizer.step()

        pred = torch.argmax(outputs, dim=1)
        accuracies=accuracy_metric(pred, target)
        mean_acc = (mean_acc * batch_idx + accuracies) / (batch_idx + 1)

        if batch_idx%log_interval==0:
            message='Train:[{}/{}({:.0f}%)]\tloss:{:.6f}'.format(batch_idx*len(data[0]),len(train_loader.dataset),100*batch_idx/len(train_loader),np.mean(total_loss))
            if accuracy_metric is not None:
                message+='\t{}:{}'.format(accuracy_metric.name,mean_acc)
            
            logger.save_log (message)

    total_loss/=batch_idx+1
    return total_loss,metrics,mean_acc

def test_epoch(val_loader,model,loss_fn,cuda,metrics,accuracy_metric,logger=None):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
    model.eval()
    val_loss=0
    mean_acc = 0

    for batch_idx,(data,target, video_label, flow) in enumerate(val_loader):
        target=target if len(target)>0 else None

        if cuda:
            data = data.cuda()
            if target is not None:
                target=target.cuda().long()
        
        outputs=model(data, target, video_label, flow)

        loss_outputs=loss_fn(outputs, target)# + loss_fn(outputs, target_weight)
        val_loss+=loss_outputs.item()

        pred = torch.argmax(outputs, dim=1)
        accuracies=accuracy_metric(pred, target)
        mean_acc = (mean_acc * batch_idx + accuracies) / (batch_idx + 1)
        
    message='Valid:[{}/{}({:.0f}%)]\tloss:{:.6f}'.format(batch_idx*len(data[0]),len(val_loader.dataset),100*batch_idx/len(val_loader),np.mean(val_loss))
    if accuracy_metric is not None:
        message+='\t{}:{}'.format(accuracy_metric.name,mean_acc)
    
        logger.save_log (message)

    val_loss/=batch_idx+1
    return val_loss,metrics,mean_acc


def fit(train_loader,val_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval,ckpt_dir,time_str,metrics=[],start_epoch=0,par=None, logger=None, writer=None):
    best_acc, num_best = 0, 0
    best_deque = collections.deque([2, 3, 1], maxlen=3)
    accuracy_metric=ClassifyAccuracy()

    
    optimizer.step()
    for epoch in range(0,start_epoch):
        scheduler.step()
    for epoch in range(start_epoch,n_epochs):
        scheduler.step()
        train_loss,metrics,train_accuracy=train_epoch(train_loader,model,loss_fn,optimizer,cuda,log_interval,metrics,accuracy_metric, logger)
        message='Epoch {}/{}. Train set: Average loss:{:.4f} Accuracy:{:.4f}'.format(epoch+1,n_epochs,train_loss,train_accuracy)
        for metric in metrics:
            message+='\t{}:{}'.format(metric.name(),metric.value())
        
        val_loss,metrics,accuracy=test_epoch(val_loader,model,loss_fn,cuda,metrics,accuracy_metric,logger)
        val_loss/=len(val_loader)
        message+='\nEpoch {}/{}. Validation set: Average loss:{:.4f} Accuracy:{:.4f}'.format(epoch+1,n_epochs,val_loss,accuracy)
        for metric in metrics:
            message+='\t{}:{}'.format(metric.name(),metric.value())
        

        # 1. 记录这个epoch的loss值和准确率
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', accuracy, epoch)
        
        if accuracy > best_acc:# 某epoch的效果超过当前最佳
            if accuracy > 0.5:# accuracy有效
                for i in range(1,len(best_deque)+1):# 删除可能存在的上次最佳模型
                    if os.path.exists(os.path.join(ckpt_dir, 'best_{}.pth'.format(i))):
                        subprocess.run(['rm', os.path.join(ckpt_dir, 'best_{}.pth'.format(i))])
                    logger.save_log('delete saved models with accuracy accuracy: {}'.format(best_acc))
                # 保存新的最佳模型
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_1.pth'))
                logger.save_log('>>>>> save NEW best epoch so far, epoch: {}, accuracy: {}'.format(epoch+1, accuracy))
                best_deque = collections.deque([2, 3, 1], maxlen=3)
            best_acc = accuracy
        elif accuracy == best_acc and best_acc > 0.5:# 和已有有效的最佳结果一样
            pop_name = best_deque.popleft()
            best_deque.append(pop_name)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_{}.pth'.format(pop_name)))
            logger.save_log('------ save No.{} best epoch, epoch: {}, accuracy: {}'.format(pop_name, epoch+1, accuracy))

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'epoch{}.pth'.format(epoch+1)))
        
        logger.save_log (message)
        logger.save_log(' ')
    print('End training, best_acc = ', best_acc.item())

def video_classify(dataloader, model, Early_stop = False, Frame_optical_flag = True, \
                txt_file = "Frame_continuous-perception.txt", train_no = 0, \
                Early_stop_thresh = 0.8, n = 5, result_dir='./results/', frame_optical_from_csv = True, logger=None):
    if n==5:
        color=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd']
        label_plottings=['pant','shirt','sweater','towel','tshirt']
    if n==3:
        color=['#1f77b4','#ff7f01','#2ca02c','#1f77b4','#ff7f01','#2ca02c']
        label_plottings=['light','medium','heavy']
    
    correct_video_frame, correct_video_optical, total_video  = np.zeros((n)), np.zeros((n)), np.zeros((n))
    time_avg = 0
    for batch_idx,(data,target, video_label, flow) in enumerate(dataloader):
        target=target if len(target)>0 else None

        if cuda:
            data = data.cuda()
            if target is not None:
                target=target.cuda().long()
        
        time1 = time.time()
        outputs = model(data, target, video_label, flow)# [BS, 5]
        video_pred_frame, video_pred_optical = None, None
        video_pred_frame = torch.argmax(outputs, dim=1)
        
        time_avg += time.time() - time1
        

        if video_pred_frame == target:
            correct_video_frame[video_pred_frame] +=1
        if video_pred_optical == target:
            correct_video_optical[video_pred_optical] +=1
        total_video[target] += 1  

        if video_pred_frame is not None:
            video_pred_frame = label_plottings[video_pred_frame]
        if video_pred_optical is not None:
            video_pred_optical = label_plottings[video_pred_optical]

        with open(os.path.join(result_dir, txt_file.replace('.txt', '_detail.txt')), 'a') as file1:
            print ('[train]: no.',str(train_no+1).zfill(2), ',[video_idx],',video_label,',[Category],',label_plottings[target],\
                    ',[Pred_frame],',str(video_pred_frame),\
                    ',[Pred_optical],',str(video_pred_optical), file=file1)

    with open(os.path.join(result_dir, txt_file), 'a') as file0:
        print('[Frame], [train]: no.',str(train_no+1).zfill(2),  ',[category_acc], ', correct_video_frame[0]/total_video[0],',', correct_video_frame[1]/total_video[1],',',\
              correct_video_frame[2]/total_video[2],',',correct_video_frame[3]/total_video[3],',',correct_video_frame[4]/total_video[4], ',[video_acc],', correct_video_frame.sum()/total_video.sum(),\
              ',[Time per video]',str(round(time_avg/(batch_idx+1), 5)), file=file0)
    
    logger.save_log('video_acc_frame = {}, video_acc_optical = {}'.format(correct_video_frame.sum()/total_video.sum(), correct_video_optical.sum()/total_video.sum()))
    logger.save_log('#######################   End inference   #############################')
               

##########################################################################
def main():
    
    pars=argparse.ArgumentParser()
    pars.add_argument('--train_mode',type=str,default='train',help='train modes: train, val, test')
    pars.add_argument('--class_model',type=int,default=1,help='train for shape or weight classification: shape(1), weight(2)')
    pars.add_argument('--rgb_depth',type=str,default='rgb',help='train for shape or weight classification')
    pars.add_argument('--train_split',type=str,default='train_file/',help='train for shape or weight classification')
    pars.add_argument('--vali_split',type=str,default='vali_file/',help='train for shape or weight classification')
    pars.add_argument('--test_split',type=str,default='test_file/',help='train for shape or weight classification')
    pars.add_argument('--img_path',type=str,default='img/',help='train for shape or weight classification')
    pars.add_argument('--csv_path',type=str,default='target/target.csv',help='train for shape or weight classification')
    pars.add_argument('--model_path',type=str,default='Model/',help='train for shape or weight classification')
    pars.add_argument('--exp_name',type=str,default='Scripts_ours/',help='train for shape or weight classification')
    pars.add_argument('--use_flow',type=str,default='True',help='use optical flow or not')
    pars.add_argument('--use_early_stop',type=str,default='False',help='use optical flow or not')
    par=pars.parse_args()

    if par.class_model == 1:
        print('Train for shapes classification')
        train_name = 'shape_model'
        physnet_classes=['1','2','3','4','5']
        numbers=[1,2,3,4,5]
    elif par.class_model == 2:
        print('Train for weights classification')
        train_name = 'weight_model'
        physnet_classes=['1','2','3']
        numbers=[1,2,3]
    colors=['#1f77b4','#ff7f01','#2ca02c','#d62728','#9467bd','#7f7f7f','#bcbd22','#17becf','#585957','#7f7f7f']
    num_classes=len(physnet_classes)
    print ('physnet_claasses:',len(physnet_classes))
    print ('colors:',len(colors))
    if par.rgb_depth == 'rgb':
        img_chs = 3
        mean, std = (0.0097459145, 0.00797071, 0.00843916), (0.07246114, 0.06304047, 0.063311584)# rgb
    elif par.rgb_depth == 'depth':
        img_chs = 1
        mean,std=(0.00586554,), (0.03234654,)# backup for depth

    cuda=torch.cuda.is_available()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic=True

    rgb_depth_root = './Database/'
    par.img_path = os.path.join(rgb_depth_root, 'video_'+par.rgb_depth)

    network_name = 'STEM_BiTF'

    num_train=4
    for n in range(num_train):
        writer = SummaryWriter(comment=network_name+'_{}_No{}'.format(par.train_mode, str(n+1).zfill(2)))
        exp_path = os.path.join('./', par.model_path, par.exp_name, network_name)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        result_root_dir = os.path.join(exp_path, train_name)
        ckpt_dir = os.path.join(result_root_dir, 'no_'+str(n+1).zfill(2))
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        mylogger = mylog(log_path=os.path.join(ckpt_dir, 'log.txt'))
    
        if 'STEM' in network_name:
            batch_size=2
            rgb_pretrained_model_path = os.path.join('/home/hl/microscopy/codes/GarFusion_test/Model/Scripts_ours/STEM/shape_model/', 'no_'+str(n+1).zfill(2))
            model = globals()[network_name](img_chs,num_classes, rgb_pretrained_model_path)
        if cuda:
            model=model.cuda()
        
        mylogger.save_log('network_name = {}, batch_size = {}'.format(network_name, batch_size))

        params=[]
        for name,param in model.named_parameters():
            if param.requires_grad==True:
                params.append(param)
        # Getting the list of image of each train_no
        par.train_file = os.path.join(rgb_depth_root, 'video_label', 'no_'+str(n+1).zfill(2), 'train.csv')
        par.vali_file = os.path.join(rgb_depth_root, 'video_label', 'no_'+str(n+1).zfill(2), 'val.csv')
        par.test_file = os.path.join(rgb_depth_root, 'video_label', 'no_'+str(n+1).zfill(2), 'test.csv')

        train_dataset=GarOFV_Video_Dataset(train_mode='train',transform=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),opt=par.class_model, par=par)
        valid_dataset=GarOFV_Video_Dataset(train_mode='val',transform=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),opt=par.class_model, par=par)
        test_dataset=GarOFV_Video_Dataset(train_mode='test',transform=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),opt=par.class_model, par=par)
        kwargs={'num_workers':1,'pin_memory':True} if cuda else {}
        train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
        valid_loader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,**kwargs)
        test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False,**kwargs)
        
        # format_str = "%Y%m%d-%H%M%S"
        time_str = time.strftime("%Y%m%d-%H%M%S",time.localtime()) #获取当前格式化过的时间字符串
        mylogger.save_log('   Time: {}  '.format(time_str))
        lr=1e-4
        optimizer=optim.Adam(params,lr=lr)
        scheduler=lr_scheduler.StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
        n_epochs=40
        log_interval=100
        loss_fn = torch.nn.CrossEntropyLoss()

        if par.train_mode == 'train':
            mylogger.save_log('#######################   Start training   #############################')
            load_path = os.path.join(rgb_pretrained_model_path, 'best_1.pth')
            mylogger.save_log('load_path = '+load_path)
            model.load_state_dict(torch.load(load_path), strict=False)
            fit(train_loader,valid_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval,ckpt_dir,time_str,par=par, logger=mylogger, writer = writer)
            torch.save(model.state_dict(), os.path.join(ckpt_dir,'last_epoch{}.pth'.format(n_epochs-1)))
            mylogger.save_log('#######################   End Training   #############################')
            
        elif par.train_mode == 'val':
            mylogger.save_log('#######################   Start validating   #############################')
            accuracy_metric=ClassifyAccuracy()
            metrics = []
            load_path = os.path.join(ckpt_dir, 'last_epoch39.pth')
            mylogger.save_log('load_path = '+load_path)
            model.load_state_dict(torch.load(load_path))

            val_loss,metrics,accuracy=test_epoch(valid_loader,model,loss_fn,cuda,metrics,accuracy_metric, logger=mylogger)
            val_loss/=len(valid_loader)
            message='\nEpoch {}/{}. Validation set: Average loss:{:.4f} Accuracy:{:.4f}'.format(30,n_epochs,val_loss,accuracy)
            for metric in metrics:
                message+='\t{}:{}'.format(metric.name(),metric.value())
            mylogger.save_log('#######################   End Validating   #############################')
        
        elif par.train_mode == 'test':
            mylogger.save_log('#######################   Start testing   #############################')
            
            for i in range(1,3+1):# deque的长度，保存的是best_{1,2,3}.pth
                if os.path.exists(os.path.join(ckpt_dir, 'best_{}.pth'.format(i))):
                    load_path = os.path.join(ckpt_dir, 'best_{}.pth'.format(i))
                else:
                    load_path = os.path.join(ckpt_dir, 'last_epoch39.pth')
                mylogger.save_log('load_path = '+load_path)
                print('load_path = '+load_path)
                model.load_state_dict(torch.load(load_path))

                txt_file = "Optical-flow_continuous-perception.txt"
                video_classify(test_loader,  model, False, True, txt_file, n, result_dir=result_root_dir, logger=mylogger)

                mylogger.save_log('finished testing '+load_path)


        mylogger.save_log ('--finished!--No. train '+str(n+1).zfill(2))

if __name__ == '__main__':
    main()