import os
import argparse
from typing import Any,Callable,Dict,IO,Optional,Tuple,Union
import pandas as pd
import cv2
from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from Prepare_tools.HS import HS


from torch.utils.tensorboard import SummaryWriter

cuda=torch.cuda.is_available()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_video_flow(dataloader_OF):
    # video_num = (len(dataloader_OF.dataset)+1)
    # frame_optical = np.zeros((video_len-1)*video_num)# 59 optical flow value * 50 videos
    k=0
    video_i = 0
    flow_save_dir = './flow_256'
    if not os.path.exists(flow_save_dir):
        os.makedirs(flow_save_dir)
    flow_video = []
    # for img_pair_1, img_pair_2,video_label in dataloader_OF:
    for batch_idx,(img_pair_1, img_pair_2,video_label) in enumerate(dataloader_OF):
        if cuda:
            img_pair_1 = img_pair_1.squeeze(0).to(device)
            img_pair_2 = img_pair_2.squeeze(0).to(device)
        list_of_flows = []
        for img_i in range(img_pair_1.shape[0]):
            u, v = HS(img_pair_1[img_i, :, :, :], img_pair_2[img_i, :, :, :])
            u, v = torch.tensor(u.squeeze(0).squeeze(0)), torch.tensor(v.squeeze(0).squeeze(0))
            flow_frame = torch.stack([u, v], dim=0)# [ 2, 64, 64]
            list_of_flows.append(flow_frame)
        
        predicted_flows = torch.stack(list_of_flows)# [59,2, 64, 64]
        torch.save(predicted_flows, os.path.join(flow_save_dir, str(video_label.item())+'.pth') )
        print('video_label = ', video_label)
    
    return flow_video

class OF_Video_Dataset(Dataset):
    def __init__(self,train:bool=True,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None,opt=1,par=None)->None:
        super(OF_Video_Dataset,self).__init__()
        self.train=train
        # self.train_file=par.train_file
        # self.vali_file=par.vali_file
        self.img_path=par.img_path
        # self.csv_path=par.csv_path
        if self.train:
            data_file=par.train_file
            data=pd.read_csv(data_file)
            self.train_data=data.iloc[:,0]
            self.train_labels=data.iloc[:,opt]
            self.video_labels=data.iloc[:,3]
        else:# val
            data_file=par.test_file
            data=pd.read_csv(data_file)
            self.vali_data=data.iloc[:,0]
            self.vali_labels=data.iloc[:,opt]
            self.video_labels=data.iloc[:,3]
        self.transform=transform
        self.target_transform=target_transform
        self.rgb_depth = par.rgb_depth
        
        # self.transform_weights = weights.transforms()

    def __getitem__(self,index:int)->Tuple[Any,Any]:
        if self.train:
            imgs_path=os.path.join(self.img_path, self.train_data[index])
        else:
            imgs_path = os.path.join(self.img_path, self.vali_data[index])
        video_label = self.video_labels.iloc[index]
        
        img_pair_1 = []
        img_pair_2 = []
        # flow = torch.zeros([len(os.listdir(imgs_path)), 2, H, W])
        flow = []
        for img_i in range(len(os.listdir(imgs_path))-1):
            img1_name = os.path.join(imgs_path, str(img_i+1).zfill(4)+'.png')
            img2_name = os.path.join(imgs_path, str(img_i+2).zfill(4)+'.png')
            # flag = 0# gray
            flag = -1# RGB
            if img_i > 0:
                img1 = img2
                img2=cv2.imread(img2_name, flag)#.transpose((2,0,1))
                if self.transform is not None:
                    img2 = self.transform(img2)
            else:
                img1=cv2.imread(img1_name, flag)#.transpose((2,0,1))
                img2=cv2.imread(img2_name, flag)#.transpose((2,0,1))
                if self.transform is not None:
                    img1 = self.transform(img1)
                    img2 = self.transform(img2)

            img_pair_1.append(img1)
            img_pair_2.append(img2)
        
        img_pair_1 = torch.stack(img_pair_1)# [59, 1, 256, 256]
        img_pair_2 = torch.stack(img_pair_2)
        return img_pair_1, img_pair_2, video_label
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.vali_data)

##########################################################################
def main():
    
    pars=argparse.ArgumentParser()
    pars.add_argument('--train_mode',type=str,default='test',help='train modes: train, val, test')
    pars.add_argument('--class_model',type=int,default=1,help='train for shape or weight classification: shape(1), weight(2)')
    pars.add_argument('--rgb_depth',type=str,default='rgb',help='train for shape or weight classification')
    pars.add_argument('--train_split',type=str,default='train_file/',help='train for shape or weight classification')
    pars.add_argument('--vali_split',type=str,default='vali_file/',help='train for shape or weight classification')
    pars.add_argument('--test_split',type=str,default='test_file/',help='train for shape or weight classification')
    pars.add_argument('--img_path',type=str,default='img/',help='train for shape or weight classification')
    pars.add_argument('--csv_path',type=str,default='target/target.csv',help='train for shape or weight classification')
    pars.add_argument('--model_path',type=str,default='Model/',help='train for shape or weight classification')
    pars.add_argument('--exp_name',type=str,default='Ours_scrips/',help='train for shape or weight classification')
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

    par.test_file = os.path.join(rgb_depth_root, 'explore.csv')

    kwargs={'num_workers':1,'pin_memory':True} if cuda else {}


    dataset_OF = OF_Video_Dataset(train=False, transform=transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()]),opt=par.class_model,par=par)
    dataloader_OF=DataLoader(dataset_OF,batch_size=1,shuffle=False,**kwargs)

    save_video_flow(dataloader_OF)


if __name__ == '__main__':
    main()