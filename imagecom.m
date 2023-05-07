classdef imagecom < handle
    properties
        height
        width
        layer
        image
        image_bpm
        image_rec
        rec_image_rgb
        image_repair
        k_array
        k_array_from_1
        k_rec_array
        pmf
        pmf_ref
        BinCode
        Codelength
        BinaryTree
        HuffCode
        BPP
        BPP_mean
        BPP_sum
        PSNR
        PSNR_mean
        PSNR_sum
        PSNR_after_repair
        Range
        Range_rec
        bytestream
        offset
        qScale
    end
    methods
        function obj = imagecom(image)
            obj.image = double(image);
            [obj.height,obj.width,obj.layer] = size(image);
            obj.PSNR_sum = 0;
            obj.BPP_sum = 0;
            obj.image_bpm = im2double(image);
        end
        function range(obj)
                obj.Range = min(obj.k_array) : max(obj.k_array);
                obj.k_array_from_1 = obj.k_array - min(obj.k_array)+1;
                obj.offset = -min(obj.k_array) +1;
        end
        function calcPSNR(obj)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
%
% Output        : PSNR     (Peak Signal to Noise Ratio)
% YOUR CODE HERE
            recImage = double(obj.rec_image_rgb);
            MSE = 1/(obj.height * obj.width * obj.layer) * sum((obj.image - recImage).^2, 'all');
            obj.PSNR=10*log10((2^8-1).^2/MSE);
            
        end

        function motion_vectors_indices = SSD(ref_image, obj)
        %  Input         : ref_image(Reference Image, size: height x width)
        %                  image (Current Image, size: height x width)
        %
        %  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
            ref_image = padarray(ref_image, [4, 4], 0, 'both');
            [W, H, ~] = size(obj.image);
            w1 = W / 8;
            h1 = H / 8;
            motion_vectors_indices = zeros(w1, h1);
            error = zeros(9,9);
            for w = 1 : 1: w1
                for h = 1 : 1 : h1
                    %8*8 block
                   current_image = obj.image((w-1)*8+1:8*w,(h-1)*8+1:8*h);
                   compare_part = ref_image((w-1)*8+1:8*w+8,(h-1)*8+1:8*h+8);
                   for x = 1:9
                       for y = 1:9
                           part = compare_part(x:x+7,y:y+7);
                           error(y,x) = sum((current_image - part).^2,"all");
                       end
                   end
                  index = find(error == min(min(error)));
                    motion_vectors_indices(w,h) = index;
                    
                end
            end
         end
        

        function rec_image = SSD_rec(ref_image, motion_vectors)
        %  Input         : ref_image(Reference Image, YCbCr image)
        %                  motion_vectors
        %
        %  Output        : rec_image (Reconstructed current image, YCbCr image)
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

%% huffman code
        %%
        function  buildHuffman( obj )
        global y
        p = obj.pmf_ref;
        p=p(:)/sum(p)+eps;              % normalize histogram
        p1=p;                           % working copy
        c=cell(length(p1),1);			% generate cell structure 
        for i=1:length(p1)				% initialize structure
           c{i}=i;						
        end
        while size(c)-2					% build Huffman tree
	        [p1,i]=sort(p1);			% Sort probabilities
	        c=c(i);						% Reorder tree.
	        c{2}={c{1},c{2}};           % merge branch 1 to 2
            c(1)=[];	                % omit 1
	        p1(2)=p1(1)+p1(2);          % merge Probabilities 1 and 2 
            p1(1)=[];	                % remove 1
        end
        getcodes(c,[]);                  % recurse to find codes
        code=char(y);
        [numCodes maxlength] = size(code); % get maximum codeword length
        length_b=0;
        HuffCode=zeros(1,numCodes);
        for symbol=1:numCodes
            for bit=1:maxlength
                length_b=bit;
                if(code(symbol,bit)==char(49)) HuffCode(symbol) = HuffCode(symbol)+2^(bit-1)*(double(code(symbol,bit))-48);
                elseif(code(symbol,bit)==char(48))
                else 
                    length_b=bit-1;
                    break;
                end;
            end;
            Codelengths(symbol)=length_b;
        end;
        obj.Codelength = Codelengths;
        obj.BinaryTree = c;
        obj.BinCode = code;
        obj.HuffCode = HuffCode;
        clear global y;
        return

        function getcodes(a,dum)                                % in every level: use the same y
        if isa(a,'cell')                    % if there are more branches...go on
                 getcodes(a{1},[dum 0]);    % 
                 getcodes(a{2},[dum 1]);
        else   
           y{a}=char(48+dum);   
        end
        end
        end
        %----------------------------------------------------------------
        
        %% 
        %--------------------------------------------------------------
        %
        %
        %
        %           %%%    %%%       %%%      %%%%%%%%
        %           %%%    %%%      %%%     %%%%%%%%%
        %           %%%    %%%     %%%    %%%%
        %           %%%    %%%    %%%    %%%
        %           %%%    %%%   %%%    %%%
        %           %%%    %%%  %%%    %%%
        %           %%%    %%% %%%    %%%
        %           %%%    %%%%%%    %%%
        %           %%%    %%%%%     %%%
        %           %%%    %%%%       %%%%%%%%%%%%
        %           %%%    %%%          %%%%%%%%%   BUILDHUFFMAN.M
        %
        %
        % description:  creatre a huffman table from a given distribution
        %
        % input:        data              - Data to be encoded (indices to codewords!!!!
        %               BinCode           - Binary version of the Code created by buildHuffman
        %               Codelengths       - Array of Codelengthes created by buildHuffman
        %
        % returnvalue:  bytestream        - the encoded bytestream
        %
        % Course:       Image and Video Compression
        %               Prof. Eckehard Steinbach
        %
        %-----------------------------------------------------------------------------------
        
        function  enc_huffman_new( obj,BinCode, Codelengths)
        
       data = obj.k_array_from_1;
        a = BinCode(data(:),:)';
        b = a(:);
        mat = zeros(ceil(length(b)/8)*8,1);
        p  = 1;
        for i = 1:length(b)
            if b(i)~=' '
                mat(p,1) = b(i)-48;
                p = p+1;
            end
        end
        p = p-1;
        mat = mat(1:ceil(p/8)*8);
        d = reshape(mat,8,ceil(p/8))';
        multi = [1 2 4 8 16 32 64 128];
        bytestreams = sum(d.*repmat(multi,size(d,1),1),2);
        BPPs = (numel(bytestreams) * 8) / (numel(obj.image) / 3);
        obj.bytestream = bytestreams;
        obj.BPP = BPPs;
        end
        
        
        
        %% 
        %--------------------------------------------------------------
        %
        %
        %
        %           %%%    %%%       %%%      %%%%%%%%
        %           %%%    %%%      %%%     %%%%%%%%%            
        %           %%%    %%%     %%%    %%%%
        %           %%%    %%%    %%%    %%%
        %           %%%    %%%   %%%    %%%
        %           %%%    %%%  %%%    %%%
        %           %%%    %%% %%%    %%%
        %           %%%    %%%%%%    %%%
        %           %%%    %%%%%     %%% 
        %           %%%    %%%%       %%%%%%%%%%%%
        %           %%%    %%%          %%%%%%%%%   BUILDHUFFMAN.M
        %
        %
        % description:  creatre a huffman table from a given distribution
        %
        % input:        bytestream        - Encoded bitstream
        %               BinaryTree        - Binary Tree of the Code created by buildHuffma
        %               nr_symbols        - Number of symbols to decode
        %
        % returnvalue:  output            - decoded data
        %
        % Course:       Image and Video Compression
        %               Prof. Eckehard Steinbach
        %
        %
        %-----------------------------------------------------------------------------------
        
        function dec_huffman_new (obj,BinaryTree)
        

            bytestream = obj.bytestream;
            
            nr_symbols = numel(obj.k_array);
        output = zeros(1,nr_symbols);
        ctemp = BinaryTree;
        
        dec = zeros(size(bytestream,1),8);
        for i = 8:-1:1
            dec(:,i) = rem(bytestream,2);
            bytestream = floor(bytestream/2);
        end
        
        dec = dec(:,end:-1:1)';
        a = dec(:);
        
        i = 1;
        p = 1;
        while(i <= nr_symbols)&&p<=max(size(a))
            while(isa(ctemp,'cell'))
                next = a(p)+1;
                p = p+1;
                ctemp = ctemp{next};
            end;
            output(i) = ctemp;
            ctemp = BinaryTree;
            i=i+1;
        end;
        k_rec_array = double(output - obj.offset);
        obj.k_rec_array = reshape(k_rec_array,size(obj.k_array));
        end
        
        
        
        
        
        
        % ctemp = BinaryTree;
        % i = 1;
        % p = 1;
        % while(i <= nr_symbols)
        %     while(isa(ctemp,'cell'))
        %         next = a(p)+1;
        %         next
        % p = p+1;
        %         ctemp = ctemp{next};
        %     end;
        %     output2(i) = ctemp;
        %     ctemp = BinaryTree;
        %     i=i+1;
        % end
        
        
        
        %return
        
%% 


end

   
end