%% IVC Lab WS22/23 Chapter 6 Final Optimization Submission
% Student 1: Peng Xie

%% Clean the workspace
clc
clear all
close all

% Load 'bpp_solution_ch4' and 'psnr_solution_ch4' variables
load('solution_values/solution_values4.mat')

% Load 'bpp_solution' and 'psnr_solution' variables
load('solution_values/solution_values5.mat')

%% Your implementation
%% Define the parameters of codec
    directory = 'foreman20_40_RGB';    %Current folder path
    flage = 0;
    %flag to control the network; use the attached net 0; trian the net 1;
%      scales_video = [0.07,0.2,0.4,0.8,1.0,1.5,2,3,4,4.5,10,20];
%      scales_still = [0.15,0.3,0.7,1.0,1.5,3,5,7,10,30];
scales_video = [0.07];
     scales_still = [0.15];

    EOB = 4000;
    range = -1000 : 4000;
    lena_small = double(imread('lena_small.tif'));
    lena_small_obj = imagecom(lena_small);
   
    frames_dir = dir(fullfile(directory,'*.bmp'));  
    num_frames = length(frames_dir); 
    foreman_image_still = cell(2,length(scales_video), num_frames);
    frames = cell(num_frames,1);
    foreman_video = cell(length(scales_video),num_frames);
%% train the nn net
   
    net = VDSRNetGeneration(directory,flage);% generate the nn for the compesation 
%% Still image codec
    fprintf('Process foreman based on lena_small with all different scales\n');
    for k = 1:2
        %Calculate the PSNR of the all frames under scales_still and
        %scales_video; So that don't need calculate the still_image PSNR in
        %video comerssion
        if k == 1
          scales_array = scales_video;
        else
           scales_array = scales_still;
        end
        for j = 1:num_frames
            % compress the image and compensate
                foreman_image_still = ImageCompression(directory,j,scales_array,frames_dir,k,foreman_image_still,lena_small_obj,EOB,range,net);
           fprintf('--------------------------------------------------------------------------------\n');
         end
    end

%% Video codec
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Video Compression......\n');
 for j = 1:num_frames
     % compress the video and the compensate 
     frames{j}=imread(fullfile(directory, frames_dir(j).name));
     foreman_video_temp = VideoCompression(frames,j,scales_video,foreman_video,foreman_image_still,EOB,net);
     foreman_video = foreman_video_temp;
 end
%% calculate the PSNR 
[PSNR_after_repair_still,PSNR_after_repair_video,BPP_still_averge,PSNR_still_averge,BPP_video,PSNR_video]...
    = PSNRCal(scales_still,scales_video,num_frames,foreman_image_still,foreman_video);
%% plot
PLOTPSNRBPP(PSNR_after_repair_still,PSNR_after_repair_video,BPP_still_averge,BPP_video,bpp_solution,psnr_solution,bpp_solution_ch4, psnr_solution_ch4)









