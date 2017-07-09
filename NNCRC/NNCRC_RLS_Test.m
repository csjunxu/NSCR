clear;
% -------------------------------------------------------------------------
%% choosing the dataset
dataset = 'ExtendedYaleB';
% 'AR_DAT'
% 'ExtendedYaleB'
% 'MNIST2k2k'
% -------------------------------------------------------------------------
%% number of repeations
nExperiment = 10;
% -------------------------------------------------------------------------
%% choosing classification methods
ClassificationMethod = 'CRC';
% ClassificationMethod = 'NNCRC' ; % non-negative CRC
% ClassificationMethod = 'NPCRC' ; % non-positive CRC
% ClassificationMethod = 'ANNCRC' ; % affine and non-negative CRC
% ClassificationMethod = 'ANPCRC' ; % affine and non-positive CRC
% -------------------------------------------------------------------------
%% directory to save the results
writefilepath  = ['C:/Users/csjunxu/Desktop/Classification/Results/' dataset '/'];
if ~isdir(writefilepath)
    mkdir(writefilepath);
end
% -------------------------------------------------------------------------
%% PCA dimension
for nDim = [84 150 300]
    par.nDim = nDim;
    %-------------------------------------------------------------------------
    %% tuning the parameters 
    for maxIter = 1%[1:1:5]
        Par.maxIter  = maxIter;
        for rho = 1%[1:1:15]
            Par.rho = rho*10^(-3);
            for lambda = [0]
                Par.lambda = lambda * 10^(-4);
                accuracy = zeros(nExperiment, 1) ;
                for n = 1:nExperiment
                    %--------------------------------------------------------------------------
                    %% data loading
                    if strcmp(dataset, 'AR_DAT') == 1
                        load(['C:/Users/csjunxu/Desktop/Classification/Dataset/AR_DAT']);
                        par.nClass        =   max(trainlabels);                 % the number of classes in the subset of AR database
                        par.nDim          =   54; % 54 120 300  the eigenfaces dimension
                        Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=par.nClass));
                        trls     =   trainlabels(trainlabels<=par.nClass);
                        Tt_DAT   =   double(NewTest_DAT(:,testlabels<=par.nClass));
                        ttls     =   testlabels(testlabels<=par.nClass);
                        clear NewTest_DAT NewTrain_DAT testlabels trainlabels
                    elseif strcmp(dataset, 'ExtendedYaleB') == 1
                        par.nDim          =   84; % 84 150 300 the eigenfaces dimension
                        load(['C:/Users/csjunxu/Desktop/Classification/Dataset/YaleBCrop025']);
                        % randomly select half of the samples as training data;
                        [dim, nSample, nClass] = size(Y);
                        par.nClass        =   nClass; % the number of classes in the subset of AR database
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
                    elseif strcmp(dataset, 'MNIST2k2k') == 1
                        load(['C:/Users/csjunxu/Desktop/Classification/Dataset/MNIST2k2k']);
                        gnd = gnd + 1;
                        par.nClass        =   max(gnd); % the number of classes in the subset of AR database
                        par.nDim          =   100; % the eigenfaces dimension
                        Tr_DAT   =   double(fea(trainIdx, :))';
                        trls     =   gnd(trainIdx, 1)';
                        Tt_DAT   =   double(fea(testIdx, :))';
                        ttls     =   gnd(testIdx, 1)';
                        clear fea gnd trainIdx testIdx
                    end
                    
                    %--------------------------------------------------------------------------
                    %% eigenface extracting
                    [disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,par.nDim);
                    tr_dat  =  disc_set'*Tr_DAT;
                    tt_dat  =  disc_set'*Tt_DAT;
                    tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [par.nDim,1]) );
                    tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [par.nDim,1]) );
                    
                    %-------------------------------------------------------------------------
                    %% testing
                    ID = [];
                    for indTest = 1:size(tt_dat,2)
                        switch ClassificationMethod
                            case 'CRC'
                                Par.lambda = .001 * size(Tr_DAT,2)/700;
                                %projection matrix computing
                                Proj_M = (tr_dat'*tr_dat+Par.lambda*eye(size(tr_dat,2)))\tr_dat';
                                [id]    = CRC_RLS(tr_dat,Proj_M,tt_dat(:,indTest),trls);
                            case 'NNCRC'                   % non-negative
                                [id]    = NNCRC(tt_dat(:,indTest), tr_dat, Par, trls);
                            case 'NPCRC'               % non-negative, sum to 1
                                [id]    = NPCRC(tt_dat(:,indTest), tr_dat, Par, trls);
                            case 'ANNCRC'                 % affine, non-negative
                                [id]    = ANNCRC(tt_dat(:,indTest), tr_dat, Par, trls);
                            case 'ANPCRC'             % affine, non-negative, sum to 1
                                [id]    = ANPCRC(tt_dat(:,indTest), tr_dat, Par, trls);
                        end
                        ID      =   [ID id];
                    end
                    cornum      =   sum(ID==ttls);
                    accuracy(n, 1)         =   [cornum/length(ttls)]; % recognition rate
                    fprintf(['Accuracy is ' num2str(accuracy(n, 1)) '.\n']);
                end
                % -------------------------------------------------------------------------
                %% save the results
                avgacc = mean(accuracy);
                if strcmp(ClassificationMethod, 'CRC') == 1
                    matname = sprintf([writefilepath dataset '_' ClassificationMethod '_DR' num2str(par.nDim) '.mat']);
                    save(matname, 'accuracy', 'avgacc');
                else
                    matname = sprintf([writefilepath dataset '_' ClassificationMethod '_DR' num2str(par.nDim) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                    save(matname,'accuracy', 'avgacc');
                end
            end
        end
    end
end