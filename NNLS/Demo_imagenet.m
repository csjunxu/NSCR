clear all; clc;
warning off;

addpath('.\data');
addpath('.\liblinear-1.96');
addpath('.\invChol');

% dataDir = fullfile('data','imagenet12-feat-caffe-alex') ; 
%dataDir = fullfile('data','imagenet12-sbow-split') ;

dirTrain=dir(fullfile(dataDir,'train_category_*.mat'));
TrainfileNames={dirTrain.name}';
dirTest=dir(fullfile(dataDir,'valid_category_*.mat'));
TestfileNames={dirTest.name}';

dname = 'ImageNet Dataset';
normalization.flag = 1;
normalization.type = 1;
gamma = [1e-2];
lambda = [1e-2];
regularizer = 'l2';

method = 'ProCRC';
    
%% pre-processing

num_class = length(TrainfileNames);
num_atom_per_class = 50;
D = [];
labelD = [];
tt_dat = [];
ttls   = [];
num_atom_ci = num_atom_per_class;
eta = 1;
%fprintf('class:');
parfor ci = 1:num_class
    tic;
    tr_dat_ci = load(fullfile(dataDir,TrainfileNames{ci}));
    %tr_dat_ci.descrs = tr_dat_ci.descrs(:,1:300);
    tr_dat_ci_descrs = double(tr_dat_ci.descrs');
    tr_dat_ci_labels = tr_dat_ci.labels';
    
    tt_dat_ci = load(fullfile(dataDir,TestfileNames{ci}));
    %tt_dat_ci.descrs = tt_dat_ci.descrs(:,1:300);
    tt_dat_ci_descrs = double(tt_dat_ci.descrs');
    tt_dat_ci_labels = tt_dat_ci.labels';
    
    if normalization.flag
        if normalization.type == 1
            tr_dat_ci_descrs = tr_dat_ci_descrs./( repmat(sqrt(sum(tr_dat_ci_descrs.*tr_dat_ci_descrs)+eps), [size(tr_dat_ci_descrs,1),1]) );
            tt_dat_ci_descrs = tt_dat_ci_descrs./( repmat(sqrt(sum(tt_dat_ci_descrs.*tt_dat_ci_descrs)+eps), [size(tt_dat_ci_descrs,1),1]) );
        end
    end
    iind = 1-isnan(tr_dat_ci_descrs);
    tr_dat_ci_descrs = tr_dat_ci_descrs(:,logical(iind(1,:)));
    iind2 = 1-isnan(tt_dat_ci_descrs);
    tt_dat_ci_labels = tt_dat_ci_labels(logical(iind2(1,:)));
    tt_dat_ci_descrs = tt_dat_ci_descrs(:,logical(iind2(1,:)));
    
%     %tr_dat_ci           =    tr_dat(:,trls==ci);
%     [Dini_ci,~,mean_ci] =    Eigenface_f(tr_dat_ci_descrs,num_atom_ci-1);
%     Dini_ci             =    [Dini_ci mean_ci./norm(mean_ci)];
     Dini_ci             =    normcol_equal(randn(size(tr_dat_ci_descrs,1),num_atom_ci));
    D_ci                =    dict_learner2(tr_dat_ci_descrs, Dini_ci, eta);
%D_ci = tr_dat_ci_descrs;
    %D_ci                =   Dini_ci;
   %D_ci                =     tr_dat_ci_descrs;
    D                   =    [D D_ci];
    labelD              =    [labelD ci*ones(1,size(D_ci,2))];
    
    tt_dat              =    [tt_dat tt_dat_ci_descrs];
    ttls                =    [ttls tt_dat_ci_labels];
    elapsed_time        =    toc;
    fprintf(['Sub-dictionary for category_',num2str(ci),' is finished in ', num2str(elapsed_time),'s','\n'])
end

tr_dat = D;
trls = labelD;

clear D labelD;

% load imagenet_trainP;
% tr_dat = double(tr_dat);
% load imagenet_test;

switch method
    case 'ProCRC'
        [x, y] = meshgrid(gamma, lambda);
        para_table = [x(:) y(:)];
        
        for i = 1:size(para_table,1)
            %% run ProCRC
            fprintf('\n------------------------Coding Process------------------------\n\n');
            tic;
            Alpha = ProCRC(tt_dat, tr_dat, trls, para_table(i,1), para_table(i,2), regularizer);
            toc
            
            tic;
            fprintf('\n------------------------Labeling Process------------------------\n\n');
            [pred_ttls, pre_matrix] = ProMax(tt_dat, Alpha, tr_dat, trls, para_table(i,1), para_table(i,2), regularizer);
            toc
            accuracy  =  (sum(pred_ttls==ttls))/length(ttls);
            %mAccuracy = [mAccuracy, accuracy];
            %lname = ['RandomProCRC_D',num2str(l+20)];
            %save(lname,'pre_matrix','pred_ttls','accuracy');
            fprintf(['\nThe accuracy on the ', dname, ' with gamma=',num2str(para_table(i,1)), ' and lambda=',num2str(para_table(i,2)), ' is ', num2str(roundn(accuracy,-3))])
        end
        
    case 'Liblinear'
        predictmatrix = [];
        mapmatrix = [];
        for ci = 1:num_class
            label_svm = -1*ones(1,size(trls,2));
            label_svm(1,trls==ci) = 1;
            model = train(label_svm',sparse(tr_dat)', '-s 1 -c 1');
            tsprediction = model.w*tt_dat+model.bias;
            predictmatrix = [predictmatrix; tsprediction];
            mapmatrix = [mapmatrix, tsprediction'];
        end
        [~, pred_ttls] = max(predictmatrix,[],1);
        accuracy  =  (sum(pred_ttls==ttls))/length(ttls);
        fprintf(['\nThe accuracy on the ', dname, ' is ', num2str(roundn(accuracy,-3))])
end
