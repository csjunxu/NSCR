clear
maxNumCompThreads(1);
warning off;

addpath('C:\Users\csjunxu\Desktop\CVPR2018 Classification\Dataset');
addpath('.\invChol');

dataDir = fullfile('C:\Users\csjunxu\Desktop\CVPR2018 Classification\Dataset','imagenet12-feat-caffe-alex') ;
% dataDir = fullfile('C:\Users\csjunxu\Desktop\Classification\Dataset','imagenet12-sbow-split') ;

dirTrain=dir(fullfile(dataDir,'train_category_*.mat'));
TrainfileNames={dirTrain.name}';
dirTest=dir(fullfile(dataDir,'valid_category_*.mat'));
TestfileNames={dirTest.name}';

dataset = 'ImageNet';
nExperiment = 1;
nDimArray = [4096];
SampleArray = 0;
normalization.flag = 1;
normalization.type = 1;
% -------------------------------------------------------------------------
%% directory to save the results
writefilepath  = ['C:/Users/csjunxu/Desktop/CVPR2018 Classification/Results/' dataset '/'];
if ~isdir(writefilepath)
    mkdir(writefilepath);
end
% -------------------------------------------------------------------------
%% choosing classification methods
% ClassificationMethod = 'NSC';
% ClassificationMethod = 'SRC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\l1_ls_matlab'));
% ClassificationMethod = 'CRC';
% ClassificationMethod = 'CROC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\CROC CVPR2012'));
% ClassificationMethod = 'ProCRC'; addpath(genpath('C:\Users\csjunxu\Desktop\CVPR2018 Classification\ProCRC'));
% ClassificationMethod = 'NNLSR' ; % non-negative LSR
% ClassificationMethod = 'NPLSR' ; % non-positive LSR
% ClassificationMethod = 'ANNLSR' ; % affine and non-negative LSR
% ClassificationMethod = 'ANPLSR' ; % affine and non-positive LSR
ClassificationMethod = 'DANNLSR' ; % deformable, affine and non-negative LSR
% ClassificationMethod = 'DANPLSR' ; % deformable, affine and non-positive LSR
% ClassificationMethod = 'ADANNLSR' ; % deformable, affine and non-negative LSR
% ClassificationMethod = 'ADANPLSR' ; % deformable, affine and non-positive LSR
% %% pre-processing
% num_class = length(TrainfileNames);
% num_atom_per_class = 50;
% D = [];
% labelD = [];
% tt_dat = [];
% ttls   = [];
% num_atom_ci = num_atom_per_class;
% eta = 1;
% %fprintf('class:');
% for ci = 1:num_class
%     tic;
%     tr_dat_ci = load(fullfile(dataDir,TrainfileNames{ci}));
%     %tr_dat_ci.descrs = tr_dat_ci.descrs(:,1:300);
%     tr_dat_ci_descrs = double(tr_dat_ci.descrs');
%     tr_dat_ci_labels = tr_dat_ci.labels';
%
%     tt_dat_ci = load(fullfile(dataDir,TestfileNames{ci}));
%     %tt_dat_ci.descrs = tt_dat_ci.descrs(:,1:300);
%     tt_dat_ci_descrs = double(tt_dat_ci.descrs');
%     tt_dat_ci_labels = tt_dat_ci.labels';
%
%     if normalization.flag
%         if normalization.type == 1
%             tr_dat_ci_descrs = tr_dat_ci_descrs./( repmat(sqrt(sum(tr_dat_ci_descrs.*tr_dat_ci_descrs)+eps), [size(tr_dat_ci_descrs,1),1]) );
%             tt_dat_ci_descrs = tt_dat_ci_descrs./( repmat(sqrt(sum(tt_dat_ci_descrs.*tt_dat_ci_descrs)+eps), [size(tt_dat_ci_descrs,1),1]) );
%         end
%     end
%     iind = 1-isnan(tr_dat_ci_descrs);
%     tr_dat_ci_descrs = tr_dat_ci_descrs(:,logical(iind(1,:)));
%     iind2 = 1-isnan(tt_dat_ci_descrs);
%     tt_dat_ci_labels = tt_dat_ci_labels(logical(iind2(1,:)));
%     tt_dat_ci_descrs = tt_dat_ci_descrs(:,logical(iind2(1,:)));
%
%     %     %tr_dat_ci           =    tr_dat(:,trls==ci);
%     %     [Dini_ci,~,mean_ci] =    Eigenface_f(tr_dat_ci_descrs,num_atom_ci-1);
%     %     Dini_ci             =    [Dini_ci mean_ci./norm(mean_ci)];
%     Dini_ci             =    normcol_equal(randn(size(tr_dat_ci_descrs,1),num_atom_ci));
%     D_ci                =    dict_learner2(tr_dat_ci_descrs, Dini_ci, eta);
%     %D_ci = tr_dat_ci_descrs;
%     %D_ci                =   Dini_ci;
%     %D_ci                =     tr_dat_ci_descrs;
%     D                   =    [D D_ci];
%     labelD              =    [labelD ci*ones(1,size(D_ci,2))];
%
%     tt_dat              =    [tt_dat tt_dat_ci_descrs];
%     ttls                =    [ttls tt_dat_ci_labels];
%     elapsed_time        =    toc;
%     fprintf(['Sub-dictionary for category_',num2str(ci),' is finished in ', num2str(elapsed_time),'s','\n'])
% end
%
% tr_dat = D;
% trls = labelD;
% save C:\Users\csjunxu\Desktop\Classification\Dataset\trdata.mat tr_dat trls;
% save C:\Users\csjunxu\Desktop\Classification\Dataset\ttdata.mat tt_dat ttls;
% clear D labelD;

load 'C:\Users\csjunxu\Desktop\CVPR2018 Classification\Dataset\imagenet_trdata.mat';
load 'C:\Users\csjunxu\Desktop\CVPR2018 Classification\Dataset\imagenet_ttdata.mat';

%-------------------------------------------------------------------------
%% PCA dimension
for nDim = nDimArray
    Par.nDim = nDim;
    for nSample = SampleArray
        %-------------------------------------------------------------------------
        %% tuning the parameters
        for s = [.7:.1:1.5]
            Par.s = s;
            for maxIter = [1:1:10]
                Par.maxIter  = maxIter;
                for rho = [.2:.2:1]
                    Par.rho = rho;
                    for lambda = [0:.1:1]
                        Par.lambda = lambda;
                        accuracy = zeros(nExperiment, 1) ;
                        for n = 1:nExperiment
                            %--------------------------------------------------------------------------
                            %% data loading
                            [dim, N] = size(tr_dat);
                            [D, N] = size(tr_dat);
                            if N < D
                                XTXinv = (tr_dat' * tr_dat + Par.rho/2 * eye(N))\eye(N);
                            else
                                XTXinv = (2/Par.rho * eye(N) - (2/Par.rho)^2 * tr_dat' / (2/Par.rho * (tr_dat * tr_dat') + eye(D)) * tr_dat );
                            end
                            %                             %--------------------------------------------------------------------------
                            %                             %% eigenface extracting
                            %                             if Par.nDim == 0 || Par.nDim == dim
                            %                                 tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [size(tr_dat,1), 1]) );
                            %                                 tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [size(tt_dat,1), 1]) );
                            %                             else
                            %                                 [disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,Par.nDim);
                            %                                 tr_dat  =  disc_set'*tr_dat;
                            %                                 tt_dat  =  disc_set'*tt_dat;
                            %                                 tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [Par.nDim,1]) );
                            %                                 tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [Par.nDim,1]) );
                            %                             end
                            %-------------------------------------------------------------------------
                            %% testing
                            class_num = max(trls);
                            if strcmp(ClassificationMethod, 'CROC') == 1
                                weight = Par.rho;
                                ID = croc_cvpr12(tt_dat, tr_dat, trls, Par.lambda, weight);
                                % ID = croc_cvpr12_v0(tt_dat, tr_dat, trls, Par.lambda, weight);
                            elseif strcmp(ClassificationMethod, 'ProCRC') == 1
                                global params
                                % set_params(dataset);
                                params.model_type        =      'ProCRC';
                                params.gamma             =     Par.rho; % [1e-2];
                                params.lambda            =      Par.lambda; % [1e-0];
                                params.class_num         =      max(trls);
                                data.tr_descr = tr_dat;
                                data.tt_descr = tt_dat;
                                data.tr_label = trls;
                                data.tt_label = ttls;
                                params.class_num = class_num;
                                coef = ProCRC(data, params);
                                [ID, ~] = ProMax(coef, data, params);
                            else
                                ID = [];
                                for indTest = 1:size(tt_dat,2)
                                    fprintf([num2str(indTest) '/' num2str(size(tt_dat,2)) ': ']);
                                    switch ClassificationMethod
                                        case 'SRC'
                                            rel_tol = 0.01;     % relative target duality gap
                                            [coef, status]=l1_ls(tr_dat, tt_dat(:,indTest), Par.lambda, rel_tol);
                                        case 'CRC'
                                            Par.lambda = .001 * size(Tr_DAT,2)/700;
                                            % projection matrix computing
                                            Proj_M = (tr_dat'*tr_dat+Par.lambda*eye(size(tr_dat,2)))\tr_dat';
                                            coef         =  Proj_M*tt_dat(:,indTest);
                                        case 'NNLSR'                   % non-negative
                                            coef = NNLS( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                        case 'NPLSR'               % non-positive
                                            coef = NPLSR( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                        case 'ANNLSR'                 % affine, non-negative, sum to 1
                                            coef = ANNLSR( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                        case 'ANPLSR'             % affine, non-negative, sum to -1
                                            coef = ANPLSR( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                        case 'DANNLSR'                 % affine, non-negative, sum to a scalar s
                                            coef = DANNLSR( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                        case 'DANPLSR'             % affine, non-positive, sum to a scalar -s
                                            coef = DANPLSR( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                        case 'ADANNLSR'                 % affine, non-negative, sum to a scalar s
                                            coef = ADANNLSR( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                        case 'ADANPLSR'             % affine, non-positive, sum to a scalar -s
                                            coef = ADANPLSR( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                    end
                                    % -------------------------------------------------------------------------
                                    %% assign the class  index
                                    if strcmp(ClassificationMethod, 'NSC') == 1
                                        for ci = 1:class_num
                                            Xc = tr_dat(:, trls==ci);
                                            Aci = Xc/(Xc'*Xc+Par.lambda*eye(size(Xc, 2)))*Xc';
                                            coef_c = Aci*tt_dat(:,indTest);
                                            error(ci) = norm(tt_dat(:,indTest)-coef_c,2)^2/sum(coef_c.*coef_c);
                                        end
                                        index      =  find(error==min(error));
                                        id         =  index(1);
                                        ID      =   [ID id];
                                    else
                                        [id, ~] = PredictID(coef, tr_dat, trls, class_num);
                                        ID      =   [ID id];
                                    end
                                end
                            end
                            cornum      =   sum(ID==ttls);
                            accuracy(n, 1)         =   [cornum/length(ttls)]; % recognition rate
                            fprintf(['Accuracy is ' num2str(accuracy(n, 1)) '.\n']);
                        end
                        % -------------------------------------------------------------------------
                        %% save the results
                        avgacc = mean(accuracy);
                        fprintf(['Mean Accuracy is ' num2str(avgacc) '.\n']);
                        if strcmp(ClassificationMethod, 'SRC') == 1 ...
                                || strcmp(ClassificationMethod, 'CRC') == 1 ...
                                || strcmp(ClassificationMethod, 'NSC') == 1
                            matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' ClassificationMethod '_DR' num2str(Par.nDim) '.mat']);
                            save(matname, 'accuracy', 'avgacc');
                        elseif strcmp(ClassificationMethod, 'ProCRC') == 1 || strcmp(ClassificationMethod, 'CROC') == 1
                            matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' ClassificationMethod '_DR' num2str(Par.nDim) '_lambda' num2str(Par.lambda) '_weight' num2str(Par.rho) '.mat']);
                            save(matname, 'accuracy', 'avgacc');
                        elseif strcmp(ClassificationMethod, 'NNLSR') == 1 || strcmp(ClassificationMethod, 'DANNLSR') == 1
                            matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' ClassificationMethod '_DR' num2str(Par.nDim) '_scale' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                            save(matname,'accuracy', 'avgacc');
                        end
                    end
                end
            end
        end
    end
end


