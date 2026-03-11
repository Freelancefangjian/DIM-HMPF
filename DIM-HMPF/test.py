import scipy.io as sio
import numpy as np
import Model
import os
import cv2
import torch
import metrics
if __name__ == '__main__':
    net = Model.Net().cuda()
    net.load_state_dict(torch.load("./model/state_dicr_220.pkl",weights_only=True))
    test_path = 'D:\\fangjian\\HSI-MSI-PAN\\chikusei\\dataprocess\\chikusei_test'
    psnr, rmse, ergas, sam, uiqi, ssim = np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8), np.zeros(8)
    lamda_m, lamda_p = 0.01, 0.01
    lamda_m = torch.tensor(lamda_m).float().view([1, 1, 1, 1])
    lamda_p = torch.tensor(lamda_p).float().view([1, 1, 1, 1])
    device = torch.device("cuda:0")
    [lamda_m, lamda_p] = [el.to(device) for el in [lamda_m, lamda_p]]
    for i in range(8):
        ind = i + 1
        path = str(i + 1) + '.mat'
        print('processing for %d' % ind)
        source_hs_path = os.path.join(test_path, 'hs', path)
        HS = sio.loadmat(source_hs_path)
        HSI = torch.FloatTensor(HS['I']).permute(2, 0, 1).unsqueeze(0).cuda()

        source_ms_path = os.path.join(test_path, 'ms', path)
        MS = sio.loadmat(source_ms_path)
        MSI = torch.FloatTensor(MS['I']).permute(2, 0, 1).unsqueeze(0).cuda()

        source_pan_path = os.path.join(test_path, 'pan', path)
        PA = sio.loadmat(source_pan_path)
        PAN = torch.FloatTensor(np.expand_dims(PA['I'], axis=2)).permute(2, 0, 1).unsqueeze(0).cuda()

        source_ms_path = os.path.join(test_path, 'gt', path)
        GT = sio.loadmat(source_ms_path)
        GT = torch.FloatTensor(GT['I']).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = net(HSI, MSI, PAN)

        psnr[i], rmse[i], ergas[i], sam[i], uiqi[i], ssim[i], data_get = metrics.metric(outputs, GT)
        #data_get = np.array(data_get, dtype=np.float64)
        sio.savemat('./get/eval_%d.mat' % ind, {'b': data_get})
        torch.cuda.empty_cache()
    print('PSNR is: %.4f, RMSE is: %.4f, ERGAS is: %.4f, SAM is: %.4f, UIQI is: %.4f, SSIM is: %.4f' %
          (np.mean(psnr), np.mean(rmse), np.mean(ergas), np.mean(sam), np.mean(uiqi), np.mean(ssim)))