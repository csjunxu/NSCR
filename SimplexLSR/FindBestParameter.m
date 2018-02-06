clear
dataset = 'AR_DAT';
% AR_DAT
% YaleBCrop025
%% directory to save the results
Original_image_dir  = ['C:/Users/csjunxu/Desktop/Classification/Results/' dataset '/'];
Sdir = regexp(Original_image_dir, '/', 'split');
fpath = fullfile(Original_image_dir, '*.mat');
im_dir  = dir(fpath);
im_num = length(im_dir);

Acc = [];
for i = 1:im_num
    result = fullfile(Original_image_dir, im_dir(i).name);
    eval(['load ' num2str(result)]);
    Acc = [Acc avgacc];
end
[maxAcc, index] = max(Acc);
 fprintf('%s : \n maxAcc = %2.4f: ', im_dir(index).name, maxAcc);
                             