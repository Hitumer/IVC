function [Ivdsr,calcpsnr_new]= Image_repair(Ireference,Ilowres,net)

    %Ireference is bmp
    %Ilowers is rgb
    [nrows,ncols,~] = size(Ireference);
    
    Iycbcr = rgb2ycbcr(Ilowres);
    Iy = Iycbcr(:,:,1);
    Icb = Iycbcr(:,:,2);
    Icr = Iycbcr(:,:,3);
    
    Iy_bicubic = imresize(Iy,[nrows ncols],"bicubic");
    Icb_bicubic = imresize(Icb,[nrows ncols],"bicubic");
    Icr_bicubic = imresize(Icr,[nrows ncols],"bicubic");
    
    Iresidual = activations(net,Iy_bicubic,41);
    Iresidual = double(Iresidual);
    
    Isr = Iy_bicubic + Iresidual;
    Ivdsr = ycbcr2rgb(cat(3,Isr,Icb_bicubic,Icr_bicubic));
    calcpsnr_new = calcPSNR(Ivdsr,Ireference);
    function MSE = calcMSE(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
% Output        : MSE      (Mean Squared Error)
% YOUR CODE HERE
    [m, n, c] = size(Image);
    Image = double(Image);
    recImage = double(recImage);
    MSE = 1/(m * n * c) * sum((Image - recImage).^2, 'all');
    end

    function PSNR = calcPSNR(Image, recImage)
    % Input         : Image    (Original Image)
    %                 recImage (Reconstructed Image)
    %
    % Output        : PSNR     (Peak Signal to Noise Ratio)
    % YOUR CODE HERE
    % call calcMSE to calculate MSE
    PSNR=10*log10((2^8-1).^2/calcMSE(Image, recImage));
    end
end    