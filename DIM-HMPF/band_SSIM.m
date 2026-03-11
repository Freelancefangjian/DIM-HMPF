clc;
clear;
sf = 8;
addpath('D:\fangjian\HSI-MSI-PAN\function');
test_path = 'get\';
test_EXP_path = 'D:\fangjian\HSI-MSI-PAN\chikusei\HSMSFusionToolbox\EXP\';
test_HySure_path = 'D:\fangjian\HSI-MSI-PAN\chikusei\HSMSFusionToolbox\HySure\';
test_FUSE_path = 'D:\fangjian\HSI-MSI-PAN\chikusei\HSMSFusionToolbox\FUSE\';

test_DUNet_path = 'D:\fangjian\HSI-MSI-PAN\chikusei\D-UNet\get\';
test_PSRT_path = 'D:\fangjian\HSI-MSI-PAN\chikusei\PSRT-main\get\';

test_HYPERPNN_path = 'D:\fangjian\HSI-MSI-PAN\chikusei\HyperPNN\get\';
test_DRCNN_path = 'D:\fangjian\HSI-MSI-PAN\chikusei\IEEE_GRSL_DRCNN-main\get_10Hfus\';

test_HMPNet_path = 'D:\fangjian\HSI-MSI-PAN\chikusei\HMPNet-master\get\';

gt_path = 'D:\fangjian\HSI-MSI-PAN\chikusei\dataprocess\chikusei_test\gt\';

i=2;
test_our = load([test_path 'eval_' int2str(i) '.mat']).b;
test_EXP = load([test_EXP_path int2str(i) '.mat']).Out;
test_FUSE = load([test_FUSE_path int2str(i) '.mat']).Out;
test_DUNet = load([test_DUNet_path 'eval_' int2str(i) '.mat']).b;
test_HYPERPNN = load([test_HYPERPNN_path 'eval_' int2str(i) '.mat']).b;
test_DRCNN = load([test_DRCNN_path int2str(i) '.mat']).Hfus;
test_HySure = load([test_HySure_path int2str(i) '.mat']).Out;
test_PSRT = load([test_PSRT_path 'eval_' int2str(i) '.mat']).b;
test_HMPNet = load([test_HMPNet_path 'eval_' int2str(i) '.mat']).b;
%% ground truth
gt = load([gt_path int2str(i) '.mat']).I;
x=[1:128];
%[ssim1(i), rmse1(i), ergas1(i), sam1(i), uiqi1(i), ssim1(i)] = quality_assessment(double(im2uint8(gt)),double(im2uint8(test_our)), 0, 1.0/sf);
for i =1:128
    ssim_our(i)=cal_ssim(double(im2uint8(gt(:,:,i))), double(im2uint8(test_our(:,:,i))),0,0);
    ssim_EXP(i)=cal_ssim(double(im2uint8(gt(:,:,i))), double(im2uint8(test_EXP(:,:,i))),0,0);
    ssim_FUSE(i)=cal_ssim(double(im2uint8(gt(:,:,i))), double(im2uint8(test_FUSE(:,:,i))),0,0);
    ssim_DUNet(i)=cal_ssim(double(im2uint8(gt(:,:,i))), double(im2uint8(test_DUNet(:,:,i))),0,0);
    ssim_HySure(i)=cal_ssim(double(im2uint8(gt(:,:,i))), double(im2uint8(test_HySure(:,:,i))),0,0);
    ssim_PSRT(i)=cal_ssim(double(im2uint8(gt(:,:,i))), double(im2uint8(test_PSRT(:,:,i))),0,0);
    %ssim_LTTR(i)=cal_ssim(double(im2uint8(gt(:,:,i))), double(im2uint8(test_LTTR(:,:,i))),0,0);
    ssim_HYPERPNN(i)=cal_ssim(double(im2uint8(gt(:,:,i))), double(im2uint8(test_HYPERPNN(:,:,i))),0,0);
    ssim_DRCNN(i)=cal_ssim(double(im2uint8(gt(:,:,i))), double(im2uint8(test_DRCNN(:,:,i))),0,0);
    %ssim_HYPERNET(i)=cal_ssim(double(im2uint8(gt(:,:,i))), double(im2uint8(test_HYPERNET(:,:,i))),0,0);
    ssim_HMPNet(i)=cal_ssim(double(im2uint8(gt(:,:,i))), double(im2uint8(test_HMPNet(:,:,i))),0,0);
end
plot(x,ssim_EXP,'color','#458A74','linewidth',1.5)
hold on
plot(x,ssim_HySure,'color','#D9A421','linewidth',1.5)
hold on
plot(x,ssim_FUSE,'color','#41B9C1','linewidth',1.5)
hold on
plot(x,ssim_DUNet,'color','#4E5689','linewidth',1.5)
hold on
plot(x,ssim_PSRT,'color','#6A8EC9','linewidth',1.5)
hold on
plot(x,ssim_HYPERPNN,'color','#652884','linewidth',1.5)
hold on
plot(x,ssim_DRCNN,'color','#8A7355','linewidth',1.5)
hold on
plot(x,ssim_HMPNet,'color','#B46DA9','linewidth',1.5)
hold on
plot(x,ssim_our,'color','#FF0000', 'linewidth',1.5)
axis([1,128,0,1])
%x,ssim2,x,ssim3,x,ssim4)
legend( 'EXP', 'HySure','FUSE','DUNet', 'PSRT','HyperPNN','DRCNN','HMPNet', 'DIM-HMPF')
xlabel('Band Number')
ylabel('MSSIM')


