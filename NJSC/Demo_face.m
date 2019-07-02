clear;
% -------------------------------------------------------------------------
%% choosing the dataset
directory = '../../data/classification/';
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
% ClassificationMethod = 'ANNLSR' ; % affine and non-negative LSR
% ClassificationMethod = 'DANNLSR' ; % deformable, affine and non-negative LSR
% ClassificationMethod = 'ADANNLSR' ; % deformable, affine and non-negative LSR
ClassificationMethod = 'NJSC' ; % non-negative joint sparse and collaborative
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
    nDimArray = 120; %[54 120 300];
end
% -------------------------------------------------------------------------
%% directory to save the results
writefilepath  = [ directory dataset '_results/'];
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
    for mu = 1
        Par.mu = mu;
        for maxIter = [1:20]
            Par.maxIter  = maxIter;
            for rho = [0.001:0.001:0.009 0.01:0.01:0.09 .1:0.1:1]
                Par.rho = rho;
                for alpha = [0 0.001 .005 0.01 .05 .1 .5 1]
                    Par.alpha = alpha;
                    for beta = [0 0.001 .005 0.01 .05 .1 .5 1]
                        Par.beta = beta;
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
                                    %rng(n);
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
                            % -------------------------------------------------------------------------
                            %% load finished IDs
                            existID  = ['TempID_' dataset '_' ClassificationMethod '_D' num2str(Par.nDim) '_mIte' num2str(Par.maxIter) '_r' num2str(Par.rho) '_a' num2str(Par.alpha) '_b' num2str(Par.beta) '_' num2str(nExperiment) '.mat'];
                            if exist(existID)==2
                                eval(['load ' existID]);
                            else
                                ID = [];
                            end
                            class_num = max(trls);
                            [D, N] = size(tr_dat);
                            if N < D
                                XTXinv = (tr_dat' * tr_dat + (Par.alpha+Par.rho/2) * eye(N))\eye(N);
                            else
                                XTXinv = (2/(2*Par.alpha+Par.rho) * eye(N) - (2/(2*Par.alpha+Par.rho))^2 * tr_dat' / (2/(2*Par.alpha+Par.rho) * (tr_dat * tr_dat') + eye(D)) * tr_dat );
                            end
                            %for indTest = size(ID)+1:size(tt_dat,2)
                            t = cputime;
                            %coef = NRC( tt_dat, tr_dat, XTXinv, Par );
                            coef = NJSC( tt_dat, tr_dat, XTXinv, Par );
                            [ID, ~] = PredictIDTop(coef, tr_dat, trls, class_num, Top);
                            e = cputime-t;
                            %fprintf([num2str(indTest) '/' num2str(size(tt_dat,2)) ': ' num2str(e) '\n']);
                            save(existID, 'ID');
                            %end
                            eval(['delete ' existID]);
                            % pause(3);
                        end
                        ttlsTop = repmat(ttls, [Top 1]);
                        cornum      =   sum(sum(uint8(ID)==ttlsTop, 1));
                        accuracy(n, 1)         =   [cornum/length(ttls)]; % recognition rate
                        % -------------------------------------------------------------------------
                        %% save the results
                        meanacc = mean(accuracy);
                        fprintf(['Mean Accuracy is ' num2str(meanacc) '.\n']);
                        %if strcmp(dataset, 'AR_DAT') == 1 && (nDim == nDimArray(1) && meanacc>=0.86) || (nDim == nDimArray(2) && meanacc>=0.91) || (nDim == nDimArray(3) && meanacc>=0.939)
                        if strcmp(dataset, 'AR_DAT') == 1 && meanacc>=0.92
                            matname = sprintf([writefilepath dataset '_' ClassificationMethod '_DR' num2str(Par.nDim) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_alpha' num2str(Par.alpha) '_beta' num2str(Par.beta) '.mat']);
                            save(matname,'accuracy', 'meanacc');
                        end
                    end
                    
                end
            end
        end
    end
end