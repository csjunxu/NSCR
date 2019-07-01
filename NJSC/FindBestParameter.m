
clear
dataset = 'YaleBCrop025';
% AR_DAT
% YaleBCrop025
%% directory to save the results
Original_image_dir  = ['C:/Users/csjunxu/Desktop/DANNLSR/Results/' dataset '/'];
Sdir = regexp(Original_image_dir, '/', 'split');
fpath = fullfile(Original_image_dir, '*.mat');
im_dir  = dir(fpath);
im_num = length(im_dir);

Acc = [];
for i = 1:im_num
    result = fullfile(Original_image_dir, im_dir(i).name);
    eval(['load ' num2str(result)]);
    if avgacc >= 0.984
        fprintf('%s;\n ', im_dir(i).name);
    end
    Acc = [Acc avgacc];
end
[maxAcc, id] = max(Acc);
fprintf('%s : maxAcc = %2.4f: ', im_dir(id).name, maxAcc);
