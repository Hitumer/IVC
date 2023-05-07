function [PSNR_after_repair_still,PSNR_after_repair_video,BPP_still_averge,PSNR_still_averge,BPP_video_averge,PSNR_video_averge] = PSNRCal(scales_still,scales_video,num_frames,foreman_image_still,foreman_video)
%   PSNR_after_repair_still =zeros(length(scales_still),1);
%   for i = 1:length(scales_still)
%         for j =1:num_frames
%         PSNR_after_repair_still(i) = foreman_image_still{1,i,j}.PSNR_after_repair + PSNR_after_repair_still(i);
%         end  
%   end
%   PSNR_after_repair_still = PSNR_after_repair_still/num_frames;
% 
%   PSNR_after_repair_video =zeros(length(scales_video),1);
%   for i = 1:length(scales_video)
%         for j =1:num_frames
%         PSNR_after_repair_video(i) = foreman_video{i,j}.PSNR_after_repair + PSNR_after_repair_video(i);
%         end
%         
%   end
%   PSNR_after_repair_video = PSNR_after_repair_video/num_frames;
%   
%   BPP_still_averge = zeros(length(scales_still),1);
%   PSNR_still_averge = zeros(length(scales_still),1);
%     for i = 1 : length(scales_still)
%        
%         for j = 1:num_frames
%             BPP_still_averge(i) = BPP_still_averge(i) + foreman_image_still{2,i,j}.BPP_mean;
%             PSNR_still_averge(i) = foreman_image_still{2,i,j}.PSNR_mean+ PSNR_still_averge(i);
%         end
% 
%     end
%     BPP_still_averge = BPP_still_averge/num_frames;
%     PSNR_still_averge = PSNR_still_averge/num_frames;
% 
%    
%         for i = 1 : length(scales_video)
% 
%             BPP_video(i) = foreman_video{i,end}.BPP_mean;
%             PSNR_video(i) = foreman_video{i,end}.PSNR_mean;
% 
%         end
%%
  PSNR_after_repair_still =zeros(length(scales_still),1);
  for i = 1:length(scales_still)
        for j =1:num_frames
        PSNR_after_repair_still(i) = foreman_image_still{1,i,j}.PSNR_after_repair + PSNR_after_repair_still(i);
        end  
  end
  PSNR_after_repair_still = PSNR_after_repair_still/num_frames;

  PSNR_after_repair_video =zeros(length(scales_video),1);
  for i = 1:length(scales_video)
        for j =1:num_frames
        PSNR_after_repair_video(i) = foreman_video{i,j}.PSNR_after_repair + PSNR_after_repair_video(i);
        end
        
  end
  PSNR_after_repair_video = PSNR_after_repair_video/num_frames;
 %% 
  BPP_still_averge = zeros(length(scales_still),1);
  PSNR_still_averge = zeros(length(scales_still),1);
    for i = 1 : length(scales_still)
       
        for j = 1:num_frames
            BPP_still_averge(i) = BPP_still_averge(i) + foreman_image_still{2,i,j}.BPP;
            PSNR_still_averge(i) = foreman_image_still{2,i,j}.PSNR+ PSNR_still_averge(i);
        end

    end
    BPP_still_averge = BPP_still_averge/num_frames;
    PSNR_still_averge = PSNR_still_averge/num_frames;

   
  BPP_video_averge = zeros(length(scales_video),1);
  PSNR_video_averge = zeros(length(scales_video),1);
    for i = 1 : length(scales_video)
       
        for j = 1:num_frames
            BPP_video_averge(i) = BPP_video_averge(i) + foreman_video{i,j}.BPP;
            PSNR_video_averge(i) = PSNR_video_averge(i) + foreman_video{i,j}.PSNR ;
        end

    end
    BPP_video_averge = BPP_video_averge/num_frames;
    PSNR_video_averge = PSNR_video_averge/num_frames;



       
end