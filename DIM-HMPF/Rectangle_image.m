function [III] = Rectangle_image(test_our,j)
    if j ~= 0
        data(:,:,1) = test_our(:,:,j);
        data(:,:,2) = test_our(:,:,j);
        data(:,:,3) = test_our(:,:,j);
    else
        data = test_our;
    end
    III = drawRectangleFrame(data,[330,400],[100,100],1);
    if j~= 0 
         image = test_our(230:330,300:400,:);
    else
        image = test_our(230:330,300:400,:);
    end
    J = imresize(image, 1.5);
    width = size(J,1);
    height = size(J,2);
    if j~=0
        III(end-width+1:end,end-height+1:end,1) = J;
        III(end-width+1:end,end-height+1:end,2) = J;
        III(end-width+1:end,end-height+1:end,3) = J;
    else
        III(end-width+1:end,end-height+1:end,:) = J;
    end
    III = drawRectangleFrame(III,[510,510],[100*1.5,100*1.5],1);
end