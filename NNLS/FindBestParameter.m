
clear
dataset = 'AR_DAT';
% AR_DAT
% YaleBCrop025
%% directory to save the results
Original_image_dir  = ['C:/Users/csjunxu/Desktop/Classification/Results/' dataset '/'];
Sdir = regexp(Original_image_dir, '/', 'split');
fpath = fullfile(Original_image_dir, 'AR_DAT_DANNLSR_DR54*.mat');
im_dir  = dir(fpath);
im_num = length(im_dir);

Acc = [];
for i = 1:im_num
    result = fullfile(Original_image_dir, im_dir(i).name);
    eval(['load ' num2str(result)]);
    Acc = [Acc accuracy];
end
maxAcc = max(Acc);
fprintf('%s : maxAcc = %2.4f: ', im_dir(i).name, maxAcc);
