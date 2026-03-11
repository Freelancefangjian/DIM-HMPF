clear;
clc;
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
ssim1 = zeros(12,1);
uiqi1=ssim1;
sam1=ssim1;
ergas1=ssim1;
rmse1=ssim1;
psnr1=ssim1;
 
for i=2
    i
    %% test
    for j=3
        %% ????????
        test_our = load([test_path 'eval_' int2str(i) '.mat']).b;
        test_EXP = load([test_EXP_path int2str(i) '.mat']).Out;
        test_FUSE = load([test_FUSE_path int2str(i) '.mat']).Out;
        test_DUNet = load([test_DUNet_path 'eval_' int2str(i) '.mat']).b;
        test_HYPERPNN = load([test_HYPERPNN_path 'eval_' int2str(i) '.mat']).b;
        test_DRCNN = load([test_DRCNN_path int2str(i) '.mat']).Hfus;
        test_HySure = load([test_HySure_path int2str(i) '.mat']).Out;
        test_PSRT = load([test_PSRT_path 'eval_' int2str(i) '.mat']).b;
        %test_HYPERNET = load([test_HYPERNET_path int2str(i) '.mat']).out;
        test_HMPNet = load([test_HMPNet_path 'eval_' int2str(i) '.mat']).b;
        
        %% ground truth
        gt = load([gt_path int2str(i) '.mat']).I;
        gt = im2uint8(gt);
        test_our = im2uint8(test_our);
        test_EXP = im2uint8(test_EXP);
        test_FUSE = im2uint8(test_FUSE);
        test_DUNet = im2uint8(test_DUNet);
        test_HYPERPNN = im2uint8(test_HYPERPNN);
        test_DRCNN = im2uint8(test_DRCNN);
        test_HySure = im2uint8(test_HySure);
        test_PSRT = im2uint8(test_PSRT);
        %test_HYPERNET = im2uint8(test_HYPERNET);
        test_HMPNet = im2uint8(test_HMPNet);
        
        %mean(image_our, 3)
        image_our = power(double(test_our) - double(gt),2);
        image_EXP = power(double(test_EXP) - double(gt),2);
        image_FUSE = power(double(test_FUSE) - double(gt),2);
        image_DUNet = power(double(test_DUNet) - double(gt),2);
        image_HYPERPNN = power(double(test_HYPERPNN) - double(gt),2);
        image_DRCNN = power(double(test_DRCNN) - double(gt),2);
        image_HySure = power(double(test_HySure) - double(gt),2);
        image_PSRT = power(double(test_PSRT) - double(gt),2);
        %image_HYPERNET = power(test_HYPERNET - gt,2);
        image_HMPNet = power(double(test_HMPNet) - double(gt),2);
        
        
        I_our = mean(image_our, 3)/255;
        I_EXP = mean(image_EXP, 3)/255;
        I_FUSE = mean(image_FUSE,3)/255;
        I_DUNet = mean(image_DUNet,3)/255;
        I_HYPERPNN = mean(image_HYPERPNN,3)/255;
        I_DRCNN = mean(image_DRCNN,3)/255;
        I_HySure = mean(image_HySure,3)/255;
        I_PSRT = mean(image_PSRT,3)/255;
        %I_HYPERNET = mean(image_HYPERNET,3)/255;
        I_HMPNet = 1.3*mean(image_HMPNet,3)/255;
        
        value = 600;
        num = 255;
        rgb_our = ind2rgb(gray2ind(I_our,num),jet(value));
        rgb_EXP = ind2rgb(gray2ind(I_EXP,num),jet(value));
        rgb_FUSE = ind2rgb(gray2ind(I_FUSE,num),jet(value));
        rgb_DUNet = ind2rgb(gray2ind(I_DUNet,num),jet(value));
        rgb_HYPERPNN = ind2rgb(gray2ind(I_HYPERPNN,num),jet(value));
        rgb_DRCNN = ind2rgb(gray2ind(I_DRCNN,num),jet(value));
        rgb_HySure = ind2rgb(gray2ind(I_HySure,num),jet(value));
        rgb_PSRT = ind2rgb(gray2ind(I_PSRT,num),jet(value));
        %rgb_HYPERNET = ind2rgb(gray2ind(I_HYPERNET,num),jet(value));
        rgb_HMPNet = ind2rgb(gray2ind(I_HMPNet,num),jet(value));
       
       
        test_HMPNet1= test_HMPNet(:,:,[70 100 36]);
        test_HMPNet1 = test_HMPNet1*3;
        test_HMPNet_1 = Rectangle_image(test_HMPNet1, 0);
        imwrite(test_HMPNet_1,['.\picture\' int2str(i) 'HMPNet_test.bmp']);
        rgb_HMPNet_1 = Rectangle_image(rgb_HMPNet, 0);
        imwrite(rgb_HMPNet_1,['.\picture\' int2str(i) 'HMPNet_rgb.bmp']);
        
        test_our1= test_our(:,:,[70 100 36]);
        test_our1 = test_our1*3;
        test_our_1 = Rectangle_image(test_our1, 0);
        imwrite(test_our_1,['.\picture\' int2str(i) 'our_test.bmp']);
        rgb_our_1 = Rectangle_image(rgb_our, 0);
        imwrite(rgb_our_1,['.\picture\' int2str(i) 'our_rgb.bmp']);
        
        test_EXP1= test_EXP(:,:,[70 100 36]);
        test_EXP_1 = Rectangle_image(test_EXP1, 0);
        imwrite(test_EXP_1,['.\picture\' int2str(i) 'EXP_test.bmp']);
        rgb_EXP_1 = Rectangle_image(rgb_EXP, 0);
        imwrite(rgb_EXP_1,['.\picture\' int2str(i) 'EXP_rgb.bmp']);
        
        test_FUSE1= test_FUSE(:,:,[70 100 36]);
        test_FUSE_1 = Rectangle_image(test_FUSE1, 0);
        imwrite(test_FUSE_1,['.\picture\' int2str(i) 'FUSE_test.bmp']);
        rgb_FUSE_1 = Rectangle_image(rgb_FUSE, 0);
        imwrite(rgb_FUSE_1,['.\picture\' int2str(i) 'FUSE_rgb.bmp']);
        
        test_DUNet1= test_DUNet(:,:,[70 100 36]);
        test_DUNet1 = test_DUNet1*3;
        test_DUNet_1 = Rectangle_image(test_DUNet1, 0);
        imwrite(test_DUNet_1,['.\picture\' int2str(i) 'DUNet_test.bmp']);
        rgb_DUNet_1 = Rectangle_image(rgb_DUNet, 0);
        imwrite(rgb_DUNet_1,['.\picture\' int2str(i) 'DUNet_rgb.bmp']);
        
        test_HYPERPNN1= test_HYPERPNN(:,:,[70 100 36]);
        test_HYPERPNN1 = test_HYPERPNN1*3;
        test_HYPERPNN_1 = Rectangle_image(test_HYPERPNN1, 0);
        imwrite(test_HYPERPNN_1,['.\picture\' int2str(i) 'HYPERPNN_test.bmp']);
        rgb_HYPERPNN_1 = Rectangle_image(rgb_HYPERPNN, 0);
        imwrite(rgb_HYPERPNN_1,['.\picture\' int2str(i) 'HYPERPNN_rgb.bmp']);
        
        test_DRCNN1= test_DRCNN(:,:,[70 100 36]);
        test_DRCNN1 = test_DRCNN1*3;
        test_DRCNN_1 = Rectangle_image(test_DRCNN1, 0);
        imwrite(test_DRCNN_1,['.\picture\' int2str(i) 'DRCNN_test.bmp']);
        rgb_DRCNN_1 = Rectangle_image(rgb_DRCNN, 0);
        imwrite(rgb_DRCNN_1,['.\picture\' int2str(i) 'DRCNN_rgb.bmp']);
        
        test_HySure1= test_HySure(:,:,[70 100 36]);
        test_HySure_1 = Rectangle_image(test_HySure1, 0);
        imwrite(test_HySure_1,['.\picture\' int2str(i) 'HySure_test.bmp']);
        rgb_HySure_1 = Rectangle_image(rgb_HySure, 0);
        imwrite(rgb_HySure_1,['.\picture\' int2str(i) 'HySure_rgb.bmp']);
        
        test_PSRT1= test_PSRT(:,:,[70 100 36]);
        test_PSRT1 = test_PSRT1*3;
        test_PSRT_1 = Rectangle_image(test_PSRT1, 0);
        imwrite(test_PSRT_1,['.\picture\' int2str(i) 'PSRT_test.bmp']);
        rgb_PSRT_1 = Rectangle_image(rgb_PSRT, 0);
        imwrite(rgb_PSRT_1,['.\picture\' int2str(i) 'PSRT_rgb.bmp']);
        
        
        imwrite(gt(:,:,j),['.\picture\' int2str(i) 'gt.bmp']);
        colormap jet
        rgb_HySure = ind2rgb(gray2ind(test_HySure(:,:,j),255),jet(255));
        imshow(rgb_HySure);colorbar('southoutside','FontSize',30,  'Direction','normal', 'Location', 'eastoutside')
        caxis([0,10])
        %% imshow
        
    end
end
 
 


