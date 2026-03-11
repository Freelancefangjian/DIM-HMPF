clear;
clc;
sf = 8;
addpath('G:\OneDrive - njust.edu.cn\Hyperspectral_Image_Fusion_Benchmarkx8\CAVE\MW-DAN\function');
test_path = 'get\';
gt_path = 'G:\OneDrive - njust.edu.cn\HSI-MSI-PAN\dataset\test\gt\';
ssim1 = zeros(12,1);
uiqi1=ssim1;
sam1=ssim1;
ergas1=ssim1;
rmse1=ssim1;
psnr1=ssim1;
for i=1:12
    i
    %% test 
    test1 = load([test_path 'eval_' int2str(i) '.mat']).b;
    
    %% ground truth
    gt = load([gt_path int2str(i) '.mat']).I;
    
    %% imshow
    subplot(1,2,1);imshow(test1(:,:,[30 20 10]),[]);
    title('Our result','fontname','Times New Roman','Color','k','FontSize',12);
    hold on;
    subplot(1,2,2);imshow(gt(:,:,[30 20 10]),[]);
    title('ground truth','fontname','Times New Roman','Color','k','FontSize',12);
    pause(0.1);
    [psnr1(i), rmse1(i), ergas1(i), sam1(i), uiqi1(i), ssim1(i)] = quality_assessment(double(im2uint8(gt)),double(im2uint8(test1)), 0, 1.0/sf);
end
mean(psnr1)
string = ["psnr","rmse","ergas","sam","uiqi","ssim"];
xlswrite('data1.xlsx', string, 'sheet1', 'A1:F1');
da = [psnr1 rmse1 ergas1 sam1 uiqi1 ssim1];
xlswrite('data1.xlsx', da, 'sheet1', 'A2:F13');