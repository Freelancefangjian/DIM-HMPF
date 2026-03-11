import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from DataSet import DataSet
from config import FLAGES
import torch.utils.data as Data
def l2_penaalty(w):
    return (w**2).sum()/2
import numpy as np
import time
import Model
import math
def PSNR(img1, img2):
    mse_sum  = (img1  - img2 ).pow(2)
    mse_loss = mse_sum.mean(2).mean(2)
    mse = mse_sum.mean()                     #.pow(2).mean()
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    # print(mse)
    return mse_loss, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
now = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
def gradient_diff(GT, Pred):
    R = F.pad(GT, [0, 1, 0, 0])[:, :, :, 1:]
    B = F.pad(GT, [0, 0, 0, 1])[:, :, 1:, :]
    dx1, dy1 = torch.abs(R - GT), torch.abs(B - GT)
    dx1[:, :, :, -1], dy1[:, :, -1, :] = 0, 0
    R = F.pad(Pred, [0, 1, 0, 0])[:, :, :, 1:]
    B = F.pad(Pred, [0, 0, 0, 1])[:, :, 1:, :]
    dx2, dy2 = torch.abs(R - Pred), torch.abs(B - Pred)
    dx2[:, :, :, -1], dy2[:, :, -1, :] = 0, 0
    res = torch.abs(dx2-dx1)+torch.abs(dy2-dy1)
    return res
from torch.optim.lr_scheduler import MultiStepLR
if __name__ == '__main__':
    #freeze_support()
    dataset = DataSet(FLAGES.pan_size, FLAGES.ms_size, FLAGES.hs_size, FLAGES.img_path, FLAGES.data_path, FLAGES.batch_size,
                      FLAGES.stride)
    #HR = np.transpose(dataset.gt, [3, 1, 2])
    PAN = torch.from_numpy(np.transpose(dataset.pan, [0, 3, 1, 2]))
    MSI = torch.from_numpy(np.transpose(dataset.ms, [0,3, 1, 2]))
    HSI = torch.from_numpy(np.transpose(dataset.hs, [0,3, 1, 2]))
    GT = torch.from_numpy(np.transpose(dataset.gt, [0,3, 1, 2]))

    torch_dataset = Data.TensorDataset(PAN, MSI, HSI, GT)
    loader = Data.DataLoader(dataset = torch_dataset, batch_size=8, shuffle=True, num_workers=2)

    device = torch.device("cuda:0")
    net = Model.Net()
    #net.load_state_dict(torch.load("./model/state_dicr_100.pkl"))
    print('# generator parameters:', sum(param.numel() for param in net.parameters()))
    net.to(device)
    import torch.optim as optim

    criterion = nn.L1Loss().to(device)
    WEIGHT_DECAY = 1e-8
    optimizer = optim.Adam(net.parameters(), lr=5*1e-4, weight_decay=WEIGHT_DECAY)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=[25, 50, 100], gamma=0.5)
    min_loss = 1.0
    #lamda_m = 0.01
    #lamda_p = 0.01
    #lamda_m = torch.tensor(lamda_m).float().view([1, 1, 1, 1])
    #lamda_p = torch.tensor(lamda_p).float().view([1, 1, 1, 1])
    #[lamda_m, lamda_p] = [el.to(device) for el in [lamda_m, lamda_p]]
    for epoch in range(0, 301):  # loop over the dataset multiple times

        running_loss = 0.0
        mpsnr = 0.0

        for i, data in enumerate(loader, 0):
            # get the inputs
            PAN1, MSI1, HSI1, GT1 = data
            PAN1 = PAN1.type(torch.FloatTensor)
            MSI1 = MSI1.type(torch.FloatTensor)
            HSI1 = HSI1.type(torch.FloatTensor)
            GT1 = GT1.type(torch.FloatTensor)

            PAN1 = PAN1.cuda(device)
            MSI1 = MSI1.cuda(device)
            HSI1 = HSI1.cuda(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(HSI1, MSI1, PAN1)
            GT1 = GT1.cuda(device)
            loss = criterion(outputs, GT1)
            #loss = criterion(outputs, GT1)
            mse, psnr = PSNR(outputs, GT1)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            mpsnr += psnr
        if epoch % 1 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.7f PSNR:%.3f' %
                    (epoch + 1, i + 1, running_loss/(i + 1), mpsnr/(i + 1)))
            running_loss = 0.0
            if min_loss >= (running_loss/(i+1)):
                min_loss = (running_loss / (i + 1))
                torch.save(net.state_dict(), './model/better_state_dicr.pkl')
        if epoch % 20 == 0:  # print every 2000 mini-batches
            torch.save(net.state_dict(), './model/state_dicr_{}.pkl'.format(epoch))

    print('Finished Training')