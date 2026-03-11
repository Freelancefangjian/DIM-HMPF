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
i = 2;
test_our = double(load([test_path 'eval_' int2str(i) '.mat']).b);
test_EXP = double(load([test_EXP_path int2str(i) '.mat']).Out);
test_FUSE = double(load([test_FUSE_path int2str(i) '.mat']).Out);
test_DUNet = double(load([test_DUNet_path 'eval_' int2str(i) '.mat']).b);
test_HYPERPNN = double(load([test_HYPERPNN_path 'eval_' int2str(i) '.mat']).b);
test_DRCNN = double(load([test_DRCNN_path int2str(i) '.mat']).Hfus);
test_HySure = double(load([test_HySure_path int2str(i) '.mat']).Out);
test_PSRT = double(load([test_PSRT_path 'eval_' int2str(i) '.mat']).b);
test_HMPNet = double(load([test_HMPNet_path 'eval_' int2str(i) '.mat']).b);
%% ground truth
x = 1:262144;

gt = load([gt_path int2str(i) '.mat']).I;
gt = double(im2uint8(gt));
%[psnr1(i), rmse1(i), ergas1(i), sam1(i), uiqi1(i), ssim1(i)] = quality_assessment(double(im2uint8(gt)),double(im2uint8(test_our)), 0, 1.0/sf);
SAM_our = angle_map1(test_our, gt);
SAM_our = sort(SAM_our(:));

SAM_EXP = angle_map1(test_EXP, gt);
SAM_EXP = sort(SAM_EXP(:));

SAM_FUSE = angle_map1(test_FUSE, gt);
SAM_FUSE = sort(SAM_FUSE(:));

SAM_DUNet = angle_map1(test_DUNet, gt);
SAM_DUNet = sort(SAM_DUNet(:));

SAM_HYPERPNN = angle_map1(test_HYPERPNN, gt);
SAM_HYPERPNN = sort(SAM_HYPERPNN(:));

SAM_DRCNN = angle_map1(test_DRCNN, gt);
SAM_DRCNN = sort(SAM_DRCNN(:));

SAM_HMPNet = angle_map1(test_HMPNet, gt);
SAM_HMPNet = sort(SAM_HMPNet(:));

SAM_DUNet = angle_map1(test_DUNet, gt);
SAM_DUNet = sort(SAM_DUNet(:));


SAM_HySure = angle_map1(test_HySure, gt);
SAM_HySure = sort(SAM_HySure(:));

SAM_PSRT = angle_map1(test_PSRT, gt);
SAM_PSRT = sort(SAM_PSRT(:));

plot(x,SAM_EXP,'color','#458A74','linewidth',1.5)
hold on
plot(x,SAM_HySure,'color','#D9A421','linewidth',1.5)
hold on
plot(x,SAM_FUSE,'color','#41B9C1','linewidth',1.5)
hold on
plot(x,SAM_DUNet,'color','#4E5689','linewidth',1.5)
hold on
plot(x,SAM_PSRT,'color','#6A8EC9','linewidth',1.5)
hold on
plot(x,SAM_HYPERPNN,'color','#652884','linewidth',1.5)
hold on
plot(x,SAM_DRCNN,'color','#8A7355','linewidth',1.5)
hold on
plot(x,SAM_HMPNet,'color','#B46DA9','linewidth',1.5)
hold on
plot(x,SAM_our,'color','#FF0000', 'linewidth',1.5)

axis([1,262144,0,2])
%x,psnr2,x,psnr3,x,psnr4)
legend( 'EXP', 'HySure','FUSE','DUNet', 'PSRT','HyperPNN','DRCNN','HMPNet', 'DIM-HMPF')
xlabel('Pixel(sorted)')
ylabel('SAM[log_{10}(.)]')



