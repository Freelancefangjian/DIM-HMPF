function [drawRectangleImage] = drawRectangleFrame(image,windowLocation,windowSize, value)
[row,col] = size(image); % 输入图像尺寸
x = windowLocation(1);%矩形框位置坐标，其格式为[x,y]
y = windowLocation(2);
height = windowSize(1);%矩形框尺寸，其格式为[height,width]，即[高度,宽度]
width = windowSize(2);
if((x<=row && y<=col)&&(height<=row && width<=col))
    %disp('矩形框合法！');
    LabelLineColor_1 = 255;          % 标记线颜色
    LabelLineColor_2 = 0;          % 标记线颜色
    LabelLineColor_3 = 0;          % 标记线颜色
    drawRectangleImage = image;
    topMost = x-height;                  % 矩形框上边缘
    botMost = x;        % 矩形框下边缘
    lefMost = y-width;                  % 矩形框左边缘
    rigMost = y;        % 矩形框右边缘
    %value = 2;
    drawRectangleImage(topMost:botMost,lefMost-value:lefMost+value,1) = LabelLineColor_1; % 左边框
    drawRectangleImage(topMost:botMost,lefMost-value:lefMost+value,2) = LabelLineColor_2; % 左边框
    %drawRectangleImage(topMost:botMost,lefMost-value:lefMost+value,3) = LabelLineColor_3; % 左边框
    
    drawRectangleImage(topMost:botMost,rigMost-value:rigMost+value,1) = LabelLineColor_1; % 右边框
    drawRectangleImage(topMost:botMost,rigMost-value:rigMost+value,2) = LabelLineColor_2; % 右边框
    %drawRectangleImage(topMost:botMost,rigMost-value:rigMost+value,3) = LabelLineColor_3; % 右边框
    
    drawRectangleImage(topMost-value:topMost+value,lefMost:rigMost,1) = LabelLineColor_1; % 上边框
    drawRectangleImage(topMost-value:topMost+value,lefMost:rigMost,2) = LabelLineColor_2; % 上边框
    %drawRectangleImage(topMost-value:topMost+value,lefMost:rigMost,3) = LabelLineColor_3; % 上边框
    
    drawRectangleImage(botMost-value:botMost+value,lefMost:rigMost,1) = LabelLineColor_1; % 下边框
    drawRectangleImage(botMost-value:botMost+value,lefMost:rigMost,2) = LabelLineColor_2; % 下边框
    %drawRectangleImage(botMost-value:botMost+value,lefMost:rigMost,3) = LabelLineColor_3; % 下边框
    
else
    disp('矩形框不合法！');
end