classdef videocom < imagecom

    properties
        current_image
        previous_image
        rec_image
        motion_vector
        error_image
        rec_frame
        rec_frame_rgb
    end

methods
    function obj = videocom(image1,image2)
        obj = obj@imagecom(image2);
        obj.current_image = obj.ictRGB2YCbCr(image2);
        obj.previous_image = obj.ictRGB2YCbCr(image1);
        
        l1 = [size(image2,1)-size(image1,1)]/2;
        ref_image = padarray(image1, [4, 4], 0, 'both');
        [W, H, ~] = size(image1);
        w1 = W / 8;
        h1 = H / 8;
        image = image2;
        %%SSD function
        image_ssd = image(:,:,1);
        ref_image_ssd = ref_image(:,:,1);
        motion_vectors_indices = zeros(w1, h1);
        error = zeros(9,9);
        for w = 1 : 1: w1
            for h = 1 : 1 : h1
                %8*8 block
               current_image = image_ssd((w-1)*8+1:8*w,(h-1)*8+1:8*h);
               compare_part = ref_image_ssd((w-1)*8+1:8*w+8,(h-1)*8+1:8*h+8);
               for x = 1:9
                   for y = 1:9
                       part = compare_part(x:x+7,y:y+7);
                       error(y,x) = sum((current_image - part).^2,"all");
                   end
               end
              index = find(error == min(min(error)));
              if numel(index) ~=1
                  index = index(1);
              end
              motion_vectors_indices(w,h) = index;
                
            end
        end

        obj.motion_vector = motion_vectors_indices;
        %%SSD_rec
        
            %  Input         : ref_image(Reference Image, YCbCr image)
            %                  motion_vectors
            %
            %  Output        : rec_image (Reconstructed current image, YCbCr image)
            motion_vectors = motion_vectors_indices;
            [M,N,C] =size(obj.previous_image);
            [l1,l2] = size(motion_vectors);
            m1 = M/l1;
            n1 = N/l2;
                rec_image = zeros(M,N,C);
                ref_image = padarray(obj.previous_image, [4, 4], 0, 'both');
            for c= 1:C
                for m = 1:1:l1
                    for n = 1:1:l2
                        index = motion_vectors(m,n);
                       [y,x] = ind2sub([9,9],index);
                        x_ref = (m-1)*m1+x:m*m1+x-1;
                        y_ref = (n-1)*n1+y:n*n1+y-1;
                        temp = ref_image(x_ref,y_ref,c);
                        x_rec = (m-1)*m1+1:m*m1;
                        y_rec = (n-1)*n1+1:n*n1;
                        rec_image(x_rec,y_rec,c) = temp;
            
                    end
                end
            end
            obj.rec_image = rec_image;
            obj.error_image = obj.current_image - rec_image;
            end

            
        function rec_image = SSD_rec(obj, motion_vectors)
        %  Input         : ref_image(Reference Image, YCbCr image)
        %                  motion_vectors
        %
        %  Output        : rec_image (Reconstructed current image, YCbCr image)
        ref_image = obj.previous_image;
        [M,N,C] =size(ref_image);
        [l1,l2] = size(motion_vectors);
        m1 = M/l1;
        n1 = N/l2;
            rec_image = zeros(M,N,C);
            ref_image = padarray(ref_image, [4, 4], 0, 'both');
            for c= 1:C
                for m = 1:1:l1
                    for n = 1:1:l2
                        index = motion_vectors(m,n);
                       [y,x] = ind2sub([9,9],index);
                        x_ref = (m-1)*m1+x:m*m1+x-1;
                        y_ref = (n-1)*n1+y:n*n1+y-1;
                        temp = ref_image(x_ref,y_ref,c);
                        x_rec = (m-1)*m1+1:m*m1;
                        y_rec = (n-1)*n1+1:n*n1;
                        rec_image(x_rec,y_rec,c) = temp;
            
                    end
                end
            end
        end
function yuv = ictRGB2YCbCr(obj,rgb)
% Input         : rgb (Original RGB Image)
% Output        : yuv (YCbCr image after transformation)
% YOUR CODE HERE
    r = rgb(:, :, 1);
    g = rgb(:, :, 2);
    b = rgb(:, :, 3);
    yuv(:, :, 1) = 0.299 * r + 0.587 * g + 0.114 * b;
    yuv(:, :, 2) = -0.169 * r - 0.331 * g + 0.5 * b;
    yuv(:, :, 3) = 0.5 * r - 0.419 * g - 0.081 * b;
end

function rgb = ictYCbCr2RGB(yuv)
% Input         : yuv (Original YCbCr image)
% Output        : rgb (RGB Image after transformation)
% YOUR CODE HERE
    y = yuv(:, :, 1);
    Cb = yuv(:, :, 2);
    Cr = yuv(:, :, 3);
    rgb(:, :, 1) = y + 1.402 * Cr;
    rgb(:, :, 2) = y - 0.344 * Cb - 0.714 * Cr;
    rgb(:, :, 3) = y + 1.772 * Cb;
end


end


end