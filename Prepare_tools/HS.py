import torch
import cv2
import numpy as np
import math
import scipy.signal
import torch.nn.functional as F 
import torchvision.transforms as transforms

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Grayscale()

def my_conv2d(x, y, flag = 'same'):
   # same
    rows, cols = y.shape[-2:]

    res =  F.conv2d(input = x, weight = y, bias = None, stride = 1, padding = [rows // 2, cols // 2])
    if rows % 2== 0 and cols % 2 == 0:
        return res[:, :, 1:, 1:]
    elif rows % 2 == 0:
        return res[:, :, 1:, :]
    elif cols % 2 == 0:
        return res[:, :, :, 1:]
    return res

def computeDerivatives(im1, im2):

    fx = my_conv2d(im1, 0.25 * torch.Tensor([[[[-1, 1], [-1, 1]]]]).to(device), 'same') + my_conv2d(im2, 0.25 * torch.Tensor([[[[-1, 1], [-1, 1]]]]).to(device), 'same')
    fy = my_conv2d(im1, 0.25 * torch.Tensor([[[[-1, -1], [1, 1]]]]).to(device), 'same') + my_conv2d(im2, 0.25 * torch.Tensor([[[[-1, -1], [1, 1]]]]).to(device), 'same')
    ft = my_conv2d(im1, 0.25 * torch.ones([1, 1, 2, 2]).to(device), 'same') + my_conv2d(im2, -0.25 * torch.ones([1, 1, 2, 2]).to(device), 'same')
    return fx, fy, ft


def gaussFilter(segma = 1):
    kSize = 2 * (segma * 3)
    x = [-(kSize/2)]
    while x[-1] + (1+1/kSize) <= (kSize/2):
        x.append(x[-1] + (1+1/kSize))   # -(kSize/2):(1+1/kSize):(kSize/2)
    x = torch.Tensor(x).to(device)
    G = (1 / (torch.sqrt(torch.Tensor([2 * np.pi])).to(device) * segma)) * torch.exp(-(x ** 2) / (2 * segma ** 2))
    G = torch.unsqueeze(G, 0)
    G = torch.unsqueeze(G, 0)
    G = torch.unsqueeze(G, 0)
    return G
    


def smoothedImg(img, segma = 1):
    G = gaussFilter(segma)
    smoothedImg = my_conv2d(img, G, 'same')
    smoothedImg = my_conv2d(smoothedImg, torch.transpose(G, 2, 3), 'same')
    return smoothedImg




def HS(im1, im2, alpha = 1, ite = 90, displayFlow = 0):
    # im1.size : [batchsize, 1, H, W]
    uInitial = torch.zeros([1, 1, im1.shape[-2], im1.shape[-1]]).to(device)
    vInitial = torch.zeros([1, 1, im2.shape[-2], im2.shape[-1]]).to(device)


    # Convert images to grayscale
    if len(im1.shape) >= 3:
        im1 = transform(im1)
    if len(im2.shape) >= 3:
        im2 = transform(im2)
    # if len(im1.shape) >= 3:
    #     im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # if len(im2.shape) >= 3:
    #     im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # im1 = im1.astype(np.float32)
    # im2 = im2.astype(np.float32)

    # im1, im2 = torch.from_numpy(im1).to(device), torch.from_numpy(im2).to(device)
    im1, im2 = im1.to(device), im2.to(device)
    while len(im1.shape) < 4:
        im1, im2 = torch.unsqueeze(im1, 0), torch.unsqueeze(im2, 0)
    # im1, im2 = torch.unsqueeze(im1, 0), torch.unsqueeze(im2, 0)

    im1 = smoothedImg(im1, 1)
    im2 = smoothedImg(im2, 1)

    # Set initial value for the flow vectors
    u = uInitial
    v = vInitial

    fx, fy, ft = computeDerivatives(im1, im2)
    # print(np.sum(fx))
    kernel_1 = torch.Tensor([[[[1/12, 1/6, 1/12], [1/6, 0, 1/6], [1/12, 1/6, 1/12]]]]).to(device)

    for i in range(ite):
        # Compute local averages of the flow vectors
        uAvg=my_conv2d(u,kernel_1,'same')
        vAvg=my_conv2d(v,kernel_1,'same')

        u= uAvg - ( fx * ( ( fx * uAvg ) + ( fy * vAvg ) + ft ) ) / ( alpha ** 2 + fx.pow(2) + fy.pow(2))
        v= vAvg - ( fy * ( ( fx * uAvg ) + ( fy * vAvg ) + ft ) ) / ( alpha ** 2 + fx.pow(2) + fy.pow(2))

    u = np.nan_to_num(u.cpu().numpy())
    v = np.nan_to_num(v.cpu().numpy())
    
    return u, v

