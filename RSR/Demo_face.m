clear;
% -------------------------------------------------------------------------
%% choosing the dataset
directory = 'C:/Users/csjunxu/Desktop/CVPR2018 Classification/Dataset/';
dataset = 'AR_DAT';
% AR_DAT
% YaleBCrop025
% GTfaceCrop
% ORLfaceCrop
% -------------------------------------------------------------------------
%% choosing classification methods
% ClassificationMethod = 'NSC';
% ClassificationMethod = 'SRC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\l1_ls_matlab'));
% ClassificationMethod = 'CRC';
% ClassificationMethod = 'CROC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\CROC CVPR2012'));
% ClassificationMethod = 'ProCRC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\ProCRC'));
% ClassificationMethod = 'NNLSR' ; % non-negative LSR
% ClassificationMethod = 'DANNLSR' ; % deformable, affine and non-negative LSR
ClassificationMethod = 'RSRLSR' ;

% -------------------------------------------------------------------------
%% number of repeations
if strcmp(dataset, 'YaleBCrop025') == 1 ...
        || strcmp(dataset, 'GTfaceCrop') == 1
    nExperiment = 10;
    nDimArray = 300; %[84 150 300];
elseif strcmp(dataset, 'ORLfaceCrop') == 1
    nExperiment = 10;
    nDimArray = [84 150 200];
elseif strcmp(dataset, 'AR_DAT') == 1
    nExperiment = 1;
    nDimArray = 54;% [54 120 300];
end
% -------------------------------------------------------------------------
%% directory to save the results
writefilepath  = ['C:/Users/csjunxu/Desktop/CVPR2018 Classification/Results/' dataset '/'];
if ~isdir(writefilepath)
    mkdir(writefilepath);
end
%-------------------------------------------------------------------------
%% Top-k accuracy
Top = 1;
%-------------------------------------------------------------------------
%% PCA dimension
for nDim = nDimArray
    Par.nDim = nDim;
    %-------------------------------------------------------------------------
    %% tuning the parameters
    for gamma = .05
        Par.gamma = gamma;
        for s = [.1:.1:2]
            Par.s = s;
            for maxIter = [5 10]
                Par.maxIter  = maxIter;
                for rho = [0.1 .01]
                    Par.rho = rho;
                    for lambda = [0 .01 .1]
                        Par.lambda = lambda;
                        accuracy = zeros(nExperiment, 1) ;
                        for n = 1:nExperiment
                            %-------------------------------------------------------------------------
                            %% data loading
                            load([directory dataset]);
                            %-------------------------------------------------------------------------
                            %% pre-processing
                            if strcmp(dataset, 'AR_DAT') == 1
                                nClass        =   max(trainlabels); % the number of classes in the subset of AR database
                                Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=nClass));
                                trls     =   trainlabels(trainlabels<=nClass);
                                Tt_DAT   =   double(NewTest_DAT(:,testlabels<=nClass));
                                ttls     =   testlabels(testlabels<=nClass);
                                clear NewTest_DAT NewTrain_DAT testlabels trainlabels
                            elseif strcmp(dataset, 'YaleBCrop025') == 1 ...
                                    || strcmp(dataset, 'GTfaceCrop') == 1 ...
                                    || strcmp(dataset, 'ORLfaceCrop') == 1
                                % randomly select half of the samples as training data;
                                [dim, nSample, nClass] = size(Y);
                                % nClass is the number of classes in the subset of AR database
                                Tr_DAT = [];
                                Tt_DAT = [];
                                trls = [];
                                ttls = [];
                                for i=1:nClass
                                    rng(n);
                                    RanCi = randperm(nSample);
                                    nTr = floor(length(RanCi)/2);
                                    nTt = length(RanCi) - nTr;
                                    Tr_DAT   =   [Tr_DAT double(Y(:, RanCi(1:nTr), i))];
                                    trls     =   [trls i*ones(1, nTr)];
                                    Tt_DAT   =   [Tt_DAT double(Y(:, RanCi(nTr+1:end), i))];
                                    ttls     =   [ttls i*ones(1, nTt)];
                                end
                                clear Y I Ind s
                            end
                            %--------------------------------------------------------------------------
                            %% eigenface extracting
                            if Par.nDim == 0
                                tr_dat  =  Tr_DAT./( repmat(sqrt(sum(Tr_DAT.*Tr_DAT)), [size(Tr_DAT,1), 1]) );
                                tt_dat  =  Tt_DAT./( repmat(sqrt(sum(Tt_DAT.*Tt_DAT)), [size(Tt_DAT,1), 1]) );
                            else
                                [disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,Par.nDim);
                                tr_dat  =  disc_set'*Tr_DAT;
                                tt_dat  =  disc_set'*Tt_DAT;
                                tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [Par.nDim,1]) );
                                tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [Par.nDim,1]) );
                            end
                            %-------------------------------------------------------------------------
                            %% testing
                            class_num = max(trls);
                            [D, N] = size(tr_dat);
                            if N < D
                                XTXinv = (tr_dat' * tr_dat + Par.rho/2 * eye(N))\eye(N);
                            else
                                XTXinv = (2/Par.rho * eye(N) - (2/Par.rho)^2 * tr_dat' / (2/Par.rho * (tr_dat * tr_dat') + eye(D)) * tr_dat );
                            end 
                            t = cputime;
                            tt_dat_temp = [tt_dat; Par.gamma*s*ones(1, size(tt_dat,2))];
                            tr_dat_temp = [tr_dat; Par.gamma*ones(1, size(tr_dat,2))];
                            coef = NNLSR( tt_dat_temp, tr_dat_temp, XTXinv, Par );
                            [ID, ~] = PredictID(coef, tr_dat, trls, class_num);
                        end
                        ttlsTop = repmat(ttls, [Top 1]);
                        cornum      =   sum(sum(ID==ttlsTop, 1));
                        accuracy(n, 1)         =   [cornum/length(ttls)]; % recognition rate
                        fprintf(['Accuracy is ' num2str(accuracy(n, 1)) '.\n']);
                    end
                    % -------------------------------------------------------------------------
                    %% save the results
                    avgacc = mean(accuracy);
                    fprintf(['Mean Accuracy is ' num2str(avgacc) '.\n']);
                    matname = sprintf([writefilepath dataset '_' ClassificationMethod '_DR' num2str(Par.nDim) '_gamma' num2str(Par.gamma) '_scale' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                    save(matname,'accuracy', 'avgacc');
                end
            end
        end
    end
end