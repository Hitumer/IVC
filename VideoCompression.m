function foreman_video = VideoCompression(frames,j,scales_video,foreman_video,foreman_image_still,EOB,net)

     
    for p = 1:length(scales_video)
%         PSNR_each_frame = zeros(1, num_frames);
%         BPP_each_frame = zeros(1, num_frames);
        qScale = scales_video(p);
    % Intra-Encode the first frame
        if j == 1
            foreman_video{p,j} = foreman_image_still{1,p,j};
%             foreman_video{p,j}.PSNR_sum = foreman_video{p,j}.PSNR_sum +foreman_video{p,j}.PSNR;
%             
%             foreman_video{p,j}.BPP_sum = foreman_video{p,j}.BPP_sum +foreman_video{p,j}.BPP;
    % Motion compensation
        else
            foreman_video{p,j} = videocom(foreman_video{p,j - 1}.rec_image_rgb,frames{j});
            % Get motion vectors and error image
            % Intra-Encode error_image
            %put the error image into  and calculate the k_array of
            %error imgae
            foreman_video{p,j}.k_array = IntraEncode(foreman_video{p,j}.error_image, qScale, EOB, false);
             
             % Build Huffmann code table for motion vector and error image
           
            mv_video = imagecom(foreman_video{p,j}.motion_vector);
            mv_video.pmf_ref = stats_marg(mv_video.image...
                , 1:81);

            mv_video.buildHuffman();
            foreman_video{p,j}.pmf_ref = stats_marg(foreman_video{p,j}.k_array, -2000:4000);
            foreman_video{p,j}.buildHuffman();
            
            % Huffmann encoding motion vector and error image
            
            mv_video.offset = 0;
            mv_video.k_array =  foreman_video{p,j}.motion_vector;
            mv_video.k_array_from_1 = mv_video.k_array + mv_video.offset;
            mv_video.enc_huffman_new(mv_video.BinCode,mv_video.Codelength);
            mv_video.dec_huffman_new(mv_video.BinaryTree);
            mv_video.BPP = mv_video.BPP/192;
            foreman_video{p,j}.offset= 2001;
            foreman_video{p,j}.k_array_from_1 =  foreman_video{p,j}.k_array +    foreman_video{p,j}.offset;
            foreman_video{p,j}.enc_huffman_new(foreman_video{p,j}.BinCode,foreman_video{p,j}.Codelength);
            % Huffmann decoding motion vector and error
            foreman_video{p,j}.BPP = foreman_video{p,j}.BPP + mv_video.BPP;
            % calculate the k_rec_array
            foreman_video{p,j}.dec_huffman_new(foreman_video{p,j}.BinaryTree);
            foreman_video{p,j}.error_image = IntraDecode(foreman_video{p,j}.k_rec_array,size(frames{j}), qScale, EOB, false);
          
            % Reconstruct frame
            temp = foreman_video{p,j}.SSD_rec(mv_video.k_rec_array);
            foreman_video{p,j}.rec_frame = foreman_video{p,j}.error_image+ temp;
            
            foreman_video{p,j}.rec_image_rgb = ictYCbCr2RGB(foreman_video{p,j}.rec_frame);
           [foreman_video{p,j}.image_repair, foreman_video{p,j}.PSNR_after_repair]= Image_repair(foreman_video{p,j}.image_bpm,foreman_video{p,j}.rec_image_rgb,net);


            foreman_video{p,j}.calcPSNR;
            %foreman_video{p,j}.PSNR_sum = foreman_video{p,j-1}.PSNR_sum +foreman_video{p,j}.PSNR;
            
            %foreman_video{p,j}.BPP_sum = foreman_video{p,j-1}.BPP_sum +foreman_video{p,j}.BPP;
%             foreman_video{p,j}.PSNR_mean = foreman_video{p,j}.PSNR_sum/j;
%             foreman_video{p,j}.BPP_mean = foreman_video{p,j}.BPP_sum/j;
        end
        fprintf('frame: %.0f Scale: %.2f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', j, qScale, foreman_video{p,j}.BPP, foreman_video{p,j}.PSNR);
        fprintf('frame: %.0f Scale: %.2f bit-rate: %.2f bits/pixel PSNR After Compensation: %.2fdB\n', j, qScale, foreman_video{p,j}.BPP, foreman_video{p,j}.PSNR_after_repair);
 
    end
%     foreman_video{p,j}.PSNR_mean = foreman_video{p,j}.PSNR_sum/j;
%     foreman_video{p,j}.BPP_mean = foreman_video{p,j}.BPP_sum/j;
    fprintf('--------------------------------------------------------------------------------\n');
%%
% subfunction
    function dst = IntraDecode(image, img_size , qScale, EOB, ict)
%  Function Name : IntraDecode.m
%  Input         : image (zero-run encoded image, 1xN)
%                  img_size (original image size)
%                  qScale(quantization scale)
%  Output        : dst   (decoded image)
    image_zzd = ZeroRunDec_EoB(image, EOB);
    num_rows = img_size(1) / 8 * 64;
    num_columns = img_size(2) / 8;
    image_zzd = reshape(image_zzd(:), [num_rows, num_columns * img_size(3)]);   %correct
    [M, N] = size(image_zzd);
    image_dezig = zeros(img_size);  %Correct
    for s = 1:N
        temp1 = blockproc(image_zzd(:, s), [64, 1], @(block_struct) DeZigZag8x8(block_struct.data));
        current_dim = mod(s, 3);
        if current_dim == 0
            current_dim = 3;
        end
        current_index = floor((s - 1)/3);
        image_dezig(:, current_index*8 + 1: (current_index + 1)*8, current_dim) = temp1;
    end
   
    
    image_dequant = blockproc(image_dezig, [8, 8], @(block_struct) DeQuant8x8(block_struct.data, qScale));
    image_IDCT = blockproc(image_dequant, [8, 8], @(block_struct) IDCT8x8(block_struct.data));
    if ict == true
        dst = ictYCbCr2RGB(image_IDCT);
    else
        dst = image_IDCT;
    end
end

        function dst = IntraEncode(image, qScale, EOB, ict)
        %  Function Name : IntraEncode.m
        %  Input         : image (Original RGB Image)
        %                  qScale(quantization scale)
        %  Output        : dst   (sequences after zero-run encoding, 1xN)
            if ict == true
                imageYUV = ictRGB2YCbCr(image);
            else
                imageYUV = image;
            end
            
            %DCT Transform
            imageYUV_DCT = blockproc(imageYUV, [8, 8], @(block_struct) DCT8x8(block_struct.data));
            imageYUV_quant = blockproc(imageYUV_DCT, [8, 8], @(block_struct) Quant8x8(block_struct.data, qScale));
            imageYUV_zz = blockproc(imageYUV_quant, [8, 8], @(block_struct) ZigZag8x8(block_struct.data));
            dst = ZeroRunEnc_EoB(imageYUV_zz(:), EOB);
        end
        %% and many more functions
        function coeff = DCT8x8(block)
        %  Input         : block    (Original Image block, 8x8x3)
        %
        %  Output        : coeff    (DCT coefficients after transformation, 8x8x3)
            coeff = zeros(size(block));
            [M, N, C] = size(block);
            % Y = AXA'
            for c = 1:C
                coeff(:,:,c)=dct2(block(:,:,c));
            end
        end
        
        function block = IDCT8x8(coeff)
        %  Function Name : IDCT8x8.m
        %  Input         : coeff (DCT Coefficients) 8*8*N
        %  Output        : block (original image block) 8*8*N
            block = zeros(size(coeff));
            [~, ~, C] = size(coeff);
            for c = 1:C
                block(:,:,c)=idct2(coeff(:,:,c));
            end
        end
        
        function yuv = ictRGB2YCbCr(rgb)
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
        
        function pmf = stats_marg(image, range)
        %UNTITLED Summary of this function goes here
        %   Detailed explanation goes here
        [counts,~]=hist(image(:),range);
        pmf=counts/sum(counts);
        end
        
        function quant = Quant8x8(dct_block, qScale)
        %  Input         : dct_block (Original Coefficients, 8x8x3)
        %                  qScale (Quantization Parameter, scalar)
        %
        %  Output        : quant (Quantized Coefficients, 8x8x3)
           L = qScale * [16, 11, 10, 16, 24, 40, 51, 61;
                                12, 12, 14, 19, 26, 58, 60, 55;
                                14, 13, 16, 24, 40, 57, 69, 56;
                                14, 17, 22, 29, 51, 87, 80, 62;
                                18, 55, 37, 56, 68, 109, 103, 77;
                                24, 35, 55, 64, 81, 104, 113, 92;
                                49, 64, 78, 87, 103, 121, 120, 101;
                                72, 92, 95, 98, 112, 100, 103, 99];
            
                  C =  qScale * [17, 18, 24, 47, 99, 99, 99, 99;
                                 18, 21, 26, 66, 99, 99, 99, 99;
                                 24, 13, 56, 99, 99, 99, 99, 99;
                                 47, 66, 99, 99, 99, 99, 99, 99;
                                 99, 99, 99, 99, 99, 99, 99, 99;
                                 99, 99, 99, 99, 99, 99, 99, 99;
                                 99, 99, 99, 99, 99, 99, 99, 99;
                                 99, 99, 99, 99, 99, 99, 99, 99;];
             quant(:, :, 1) = round(dct_block(:, :, 1) ./ L);
             quant(:, :, 2) = round(dct_block(:, :, 2) ./ C);
             quant(:, :, 3) = round(dct_block(:, :, 3) ./ C);
        end
        function dct_block = DeQuant8x8(quant_block, qScale)
        %  Function Name : DeQuant8x8.m
        %  Input         : quant_block  (Quantized Block, 8x8x3)
        %                  qScale       (Quantization Parameter, scalar)
        %
        %  Output        : dct_block    (Dequantized DCT coefficients, 8x8x3)
         L = qScale * [16, 11, 10, 16, 24, 40, 51, 61;
                                12, 12, 14, 19, 26, 58, 60, 55;
                                14, 13, 16, 24, 40, 57, 69, 56;
                                14, 17, 22, 29, 51, 87, 80, 62;
                                18, 55, 37, 56, 68, 109, 103, 77;
                                24, 35, 55, 64, 81, 104, 113, 92;
                                49, 64, 78, 87, 103, 121, 120, 101;
                                72, 92, 95, 98, 112, 100, 103, 99];
            
            C =  qScale * [17, 18, 24, 47, 99, 99, 99, 99;
                                 18, 21, 26, 66, 99, 99, 99, 99;
                                 24, 13, 56, 99, 99, 99, 99, 99;
                                 47, 66, 99, 99, 99, 99, 99, 99;
                                 99, 99, 99, 99, 99, 99, 99, 99;
                                 99, 99, 99, 99, 99, 99, 99, 99;
                                 99, 99, 99, 99, 99, 99, 99, 99;
                                 99, 99, 99, 99, 99, 99, 99, 99;];
        
             dct_block(:, :, 1) = quant_block(:, :, 1) .* L;
             dct_block(:, :, 2) = quant_block(:, :, 2) .* C;
             dct_block(:, :, 3) = quant_block(:, :, 3) .* C;
        end
        
        function zz = ZigZag8x8(quant)
        %  Input         : quant (Quantized Coefficients, 8x8xN)
        %
        %  Output        : zz (zig-zag scaned Coefficients, 64xN)
            ZigZag =    [1     2    6    7    15   16   28   29;
                             3     5    8    14   17   27   30   43;
                             4     9    13   18   26   31   42   44;
                             10    12   19   25   32   41   45   54;
                             11    20   24   33   40   46   53   55;
                             21    23   34   39   47   52   56   61;
                             22    35   38   48   51   57   60   62;
                             36    37   49   50   58   59   63   64];
            [M, N, C] = size(quant);
            zz = zeros(M * N, C);
            for c = 1:C
                temp2 = quant(:, :, c);
                zz(ZigZag(:), c) = temp2(:);
            end
        end

        function coeffs = DeZigZag8x8(zz)
        %  Function Name : DeZigZag8x8.m
        %  Input         : zz    (Coefficients in zig-zag order)
        %
        %  Output        : coeffs(DCT coefficients in original order)
            [~, N] = size(zz);
            coeffs = zeros([8, 8, N]);
            ZigZag =    [1     2    6    7    15   16   28   29;
                         3     5    8    14   17   27   30   43;
                         4     9    13   18   26   31   42   44;
                         10    12   19   25   32   41   45   54;
                         11    20   24   33   40   46   53   55;
                         21    23   34   39   47   52   56   61;
                         22    35   38   48   51   57   60   62;
                         36    37   49   50   58   59   63   64];
            for l = 1:N
                ith_zz = zz(:, l);
                temp3 = ith_zz(ZigZag(:));
                temp3 = reshape(temp3, 8, 8);
                coeffs(:, :, l) = temp3;
            end
        end
        
        function zze = ZeroRunEnc_EoB(zz, EOB)
        %  Input         : zz (Zig-zag scanned sequence, 1xN)
        %                  EOB (End Of Block symbol, scalar)
        %
        %  Output        : zze (zero-run-level encoded sequence, 1xM)
            zze = zeros(size(zz));  %pre-allocate memory
            count_zze = 1;    %Using indexing, which is much faster
            len_zz = length(zz);
            zeros_num = 0;  %Number of repetitions of zeros
            for k = 1:len_zz
                temp4 = zz(k);
                % When current symbol is 0
                switch temp4 
                    case 0
                    % When this 0 is the first 0 in a string
                    switch zeros_num
                        case 0
                        zze(count_zze:count_zze + 1) = [0, 0];
                        zeros_num = 1;
                        count_zze = count_zze + 2;
                    % When this zeros is the following 0 in a string, rep += 1
                        otherwise
                        zze(count_zze - 1) = zeros_num;
                        zeros_num = zeros_num + 1;
                    end
                % When the current symbol is not 0
                    otherwise 
                    zeros_num = 0;
                    zze(count_zze) = temp4;
                    count_zze = count_zze + 1;
                end
                % Check if the last symbol of 8x8 block is 0
                if (zz(k) == 0)
                    if mod(k,64) == 0
                    zze(count_zze - 2) = EOB;
                    count_zze = count_zze - 1;
                    zeros_num = 0;
                    end
                end
            end
            
            zze(count_zze:end) = [];
            % Process end of the sequence
            if zze(end) == 0 | zze(end - 1) == 0
                zze(end - 1:end) = [];
                zze(end + 1) = EOB;
            end
        end
        
        function dst = ZeroRunDec_EoB(src, EoB)
        %  Function Name : ZeroRunDec1.m zero run level decoder
        %  Input         : src (zero run encoded sequence 1xM with EoB signs)
        %                  EoB (end of block sign)
        %
        %  Output        : dst (reconstructed zig-zag scanned sequence 1xN)
            flag_is_zero = 0;   
            [M, N] = size(src);
            dst = zeros(1, 100 * N);
            pointer = 1;
            for x = 1:length(src)
                switch src(x) 
                    case EoB
                        switch mod(pointer, 64) 
                        case 0 %at the last position of one block
                            dst(pointer) = 0;
                            pointer = pointer + 1;
                        otherwise
                            num_zeros = 64 - mod(pointer, 64) + 1;
                            dst(pointer : pointer + num_zeros - 1) = zeros(1, num_zeros);
                            pointer = pointer + num_zeros;
                        end
                       
                    otherwise
                        switch  flag_is_zero
                        case 1
                            dst(pointer:pointer + src(x) - 1) = zeros(1, src(x));
                            flag_is_zero = 0;
                            pointer = pointer + src(x);  
                        otherwise
                            if  (src(x) == 0) && (~flag_is_zero)
                                flag_is_zero = 1;
                                dst(pointer) = 0;
                                pointer = pointer + 1;
                            else
                                dst(pointer) = src(x);
                                pointer = pointer + 1;
                        end
                end
                
            end
            %Process the end of dst
            dst(pointer : end) = [];
        
            end
        end




 end