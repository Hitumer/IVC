function PLOTPSNRBPP(PSNR_after_repair_still,PSNR_after_repair_video,BPP_still_averge,BPP_video,bpp_solution,psnr_solution,bpp_solution_ch4, psnr_solution_ch4)

 figure;
    hold on
    plot(bpp_solution, psnr_solution, '--xb', 'LineWidth' , 2, 'MarkerSize', 8);
    plot(bpp_solution_ch4, psnr_solution_ch4, '--o', 'LineWidth' , 2, 'MarkerSize', 8);
    plot(BPP_still_averge, PSNR_after_repair_still, '--sblack', 'LineWidth' , 2, 'MarkerSize', 8);
    plot(BPP_video, PSNR_after_repair_video, '--o', 'LineWidth' , 2, 'MarkerSize', 8);
    legend('Still-Image Codec with Corresponding qScale','Video Codec with Corresponding qScale','Still-Image  After Compensation','Video Codec After Compensation');
    title('Rate-Distortion Plot');
    xlabel('Bit Rate [bits/pixel]');
    ylabel('PSNR [dB]');
end