clear;
% -------------------------------------------------------------------------
%% choosing the dataset
dataset = 'MNIST2k2k';
% 'AR_DAT'
% 'ExtendedYaleB'
% 'MNIST2k2k'
% -------------------------------------------------------------------------
%% number of repeations
if strcmp(dataset, 'ExtendedYaleB') == 1
    nExperiment = 10;
elseif strcmp(dataset, 'AR_DAT') == 1
    nExperiment = 1;
elseif strcmp(dataset, 'MNIST2k2k') == 1
    nExperiment = 1;
end
% -------------------------------------------------------------------------
%% choosing classification methods
% ClassificationMethod = 'SRC'; % PAMI2009
% ClassificationMethod = 'CRC'; % ICCV 2011
% ClassificationMethod = 'NNLSR' ; % non-negative LSR
% ClassificationMethod = 'NPLSR' ; % non-positive LSR
% ClassificationMethod = 'ANNLSR' ; % affine and non-negative LSR
% ClassificationMethod = 'ANPLSR' ; % affine and non-positive LSR
ClassificationMethod = 'DANNLSR' ; % deformable, affine and non-negative LSR
% ClassificationMethod = 'DANPLSR' ; % deformable, affine and non-positive LSR
% -------------------------------------------------------------------------
%% directory to save the results
writefilepath  = ['C:/Users/csjunxu/Desktop/Classification/Results/' dataset '/'];
if ~isdir(writefilepath)
    mkdir(writefilepath);
end
% -------------------------------------------------------------------------
%% PCA dimension
for nDim = [50 150 300]
    Par.nDim = nDim;
    %-------------------------------------------------------------------------
    %% tuning the parameters
    for s = [1:.1:2]
        Par.s = s;
        for maxIter = [3 4 5]
            Par.maxIter  = maxIter;
            for rho = [1 10 100]
                Par.rho = rho*10^(-3);
                for lambda = [0]
                    Par.lambda = lambda * 10^(-4);
                    accuracy = zeros(nExperiment, 1) ;
                    for n = 1:nExperiment
                        %--------------------------------------------------------------------------
                        %% data loading
                        if strcmp(dataset, 'AR_DAT') == 1
                            load(['C:/Users/csjunxu/Desktop/Classification/Dataset/AR_DAT']);
                            Par.nClass        =   max(trainlabels);                 % the number of classes in the subset of AR database
                            %                         Par.nDim          =   54; % 54 120 300  the eigenfaces dimension
                            Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=Par.nClass));
                            trls     =   trainlabels(trainlabels<=Par.nClass);
                            Tt_DAT   =   double(NewTest_DAT(:,testlabels<=Par.nClass));
                            ttls     =   testlabels(testlabels<=Par.nClass);
                            clear NewTest_DAT NewTrain_DAT testlabels trainlabels
                        elseif strcmp(dataset, 'ExtendedYaleB') == 1
                            %                         Par.nDim          =   84; % 84 150 300 the eigenfaces dimension
                            load(['C:/Users/csjunxu/Desktop/Classification/Dataset/YaleBCrop025']);
                            % randomly select half of the samples as training data;
                            [dim, nSample, nClass] = size(Y);
                            Par.nClass        =   nClass; % the number of classes in the subset of AR database
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
                            Par.nClass        =   max(gnd); % the number of classes in the subset of AR database
                            %                         Par.nDim          =   100; % the eigenfaces dimension
                            Tr_DAT   =   double(fea(trainIdx, :))';
                            trls     =   gnd(trainIdx, 1)';
                            Tt_DAT   =   double(fea(testIdx, :))';
                            ttls     =   gnd(testIdx, 1)';
                            clear fea gnd trainIdx testIdx
                        elseif strcmp(Dataset, 'USPS')==1
                            load('../USPS');
                            par.nClass        =  length(unique(gnd));
                            Tr_DAT   =   double(fea(1:7291, :)');
                            trls     =   gnd(1:7291)';
                            Tt_DAT   =   double(fea(7292:end, :)');
                            ttls     =   gnd(7292:end)';
                            Tr_DAT  =  Tr_DAT./( repmat(sqrt(sum(Tr_DAT.*Tr_DAT)), [size(Tr_DAT, 1), 1]) );
                            Tt_DAT  =  Tt_DAT./( repmat(sqrt(sum(Tt_DAT.*Tt_DAT)), [size(Tt_DAT, 1), 1]) );
                            clear fea gnd
                        end
                        
                        %--------------------------------------------------------------------------
                        %% eigenface extracting
                        [disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,Par.nDim);
                        tr_dat  =  disc_set'*Tr_DAT;
                        tt_dat  =  disc_set'*Tt_DAT;
                        tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [Par.nDim,1]) );
                        tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [Par.nDim,1]) );
                        
                        %-------------------------------------------------------------------------
                        %% testing
                        ID = [];
                        for indTest = 1:size(tt_dat,2)
                            switch ClassificationMethod
                                case 'CRC'
                                    Par.lambda = .001 * size(Tr_DAT,2)/700;
                                    %projection matrix computing
                                    Proj_M = (tr_dat'*tr_dat+Par.lambda*eye(size(tr_dat,2)))\tr_dat';
                                    coef         =  Proj_M*tt_dat(:,indTest);
                                case 'NNLSR'                   % non-negative
                                    coef = NNLSR( tt_dat(:,indTest), tr_dat, Par );
                                case 'NPLSR'               % non-positive
                                    coef = NPLSR( tt_dat(:,indTest), tr_dat, Par );
                                case 'ANNLSR'                 % affine, non-negative, sum to 1
                                    coef = ANNLSR( tt_dat(:,indTest), tr_dat, Par );
                                case 'ANPLSR'             % affine, non-negative, sum to -1
                                    coef = ANPLSR( tt_dat(:,indTest), tr_dat, Par );
                                case 'DANNLSR'                 % affine, non-negative, sum to a scalar s
                                    coef = DANNLSR( tt_dat(:,indTest), tr_dat, Par );
                                case 'DANPLSR'             % affine, non-positive, sum to a scalar -s
                                    coef = DANPLSR( tt_dat(:,indTest), tr_dat, Par );
                            end
                            % -------------------------------------------------------------------------
                            %% assign the class  index
                            for ci = 1:max(trls)
                                coef_c   =  coef(trls==ci);
                                Dc       =  tr_dat(:,trls==ci);
                                error(ci) = norm(tt_dat(:,indTest)-Dc*coef_c,2)^2/sum(coef_c.*coef_c);
                            end
                            index      =  find(error==min(error));
                            id         =  index(1);
                            ID      =   [ID id];
                        end
                        cornum      =   sum(ID==ttls);
                        accuracy(n, 1)         =   [cornum/length(ttls)]; % recognition rate
                        fprintf(['Accuracy is ' num2str(accuracy(n, 1)) '.\n']);
                    end
                    
                    % -------------------------------------------------------------------------
                    %% save the results
                    avgacc = mean(accuracy);
                    fprintf(['Mean Accuracy is ' num2str(avgacc) '.\n']);
                    if strcmp(ClassificationMethod, 'SRC') == 1 || strcmp(ClassificationMethod, 'CRC') == 1
                        matname = sprintf([writefilepath dataset '_' ClassificationMethod '_DR' num2str(Par.nDim) '.mat']);
                        save(matname, 'accuracy', 'avgacc');
                    else
                        matname = sprintf([writefilepath dataset '_' ClassificationMethod '_DR' num2str(Par.nDim) '_scale' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'accuracy', 'avgacc');
                    end
                end
            end
        end
    end
end