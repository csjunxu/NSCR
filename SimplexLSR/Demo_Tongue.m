clear;
% -------------------------------------------------------------------------
%% choosing the dataset
dataset = 'Tongue';
% -------------------------------------------------------------------------
%% directory to save the results
writefilepath  = ['C:/Users/csjunxu/Desktop/Classification/Results/' dataset '/'];
if ~isdir(writefilepath)
    mkdir(writefilepath);
end
% -------------------------------------------------------------------------
%% number of repeations
if  strcmp(dataset, 'Tongue') == 1
    nExperiment = 10;
elseif strcmp(dataset, 'Tongue') == 1
    nExperiment = 10;
end
%% Settings
if strcmp(dataset, 'Tongue') == 1
    SampleArray = [440];
elseif strcmp(dataset, 'USPS') == 1
    SampleArray = [50 100 200 300];
end
Par.nDim = 100;
% -------------------------------------------------------------------------
%% choosing classification methods
% ClassificationMethod = 'NSC';
% ClassificationMethod = 'SRC'; addpath(genpath('l1_ls_matlab'));
% ClassificationMethod = 'CRC';
% ClassificationMethod = 'CROC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\CROC CVPR2012'));
% ClassificationMethod = 'ProCRC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\ProCRC'));

% ClassificationMethod = 'NNLSR' ; % non-negative LSR
% ClassificationMethod = 'NPLSR' ; % non-positive LSR
% ClassificationMethod = 'ANNLSR' ; % affine and non-negative LSR
% ClassificationMethod = 'ANPLSR' ; % affine and non-positive LSR
ClassificationMethod = 'DANNLSR' ; % deformable, affine and non-negative LSR
% ClassificationMethod = 'DANPLSR' ; % deformable, affine and non-positive LSR

%-------------------------------------------------------------------------
%% tuning the parameters
for nSample = SampleArray % number of images for each digit
    for s = [1:.1:4]
        Par.s = s;
        for maxIter = [1:1:6]
            Par.maxIter  = maxIter;
            for rho = [1:1:10]
                Par.rho = rho*10^(-1);
                for lambda = 0 %[0:5:100]
                    Par.lambda = lambda*10^(-2);
                    accuracy = zeros(nExperiment, 1) ;
                    for i = 1:nExperiment
                        %--------------------------------------------------------------------------
                        %% Load data
                        load('C:\Users\csjunxu\Desktop\Classification\Dataset\data_example');
                        Par.nDim = min(Par.nDim, size(data_train{1}, 1));
                        nCluster = max(trls);
                        if length(nSample) == 1
                            nSample = ones(1, nCluster) * nSample;
                        end
                        mask = zeros(1, sum(nSample));
                        label = zeros(1, sum(nSample));
                        nSample_cum = [0, cumsum(nSample)];
                        for iK = 1:nCluster % randomly take data for each digit.
                            allpos = find( trls == iK );
                            rand('seed', i);
                            % rng( (i-1) * nCluster + iK );
                            selpos = allpos( randperm(length(allpos), nSample(iK)) );
                            mask( nSample_cum(iK) + 1 : nSample_cum(iK+1) ) = selpos;
                            label( nSample_cum(iK) + 1 : nSample_cum(iK+1) ) = iK * ones(1, nSample(iK));
                        end
                        % N = length(label);
                        Tr_DAT = data_train{1};
                        Tr_DAT = Tr_DAT(:, mask);
                        trls = label;
                        % trls       = trls';
                        Tt_DAT =   data_test{1};
                        ttls       =   ttls';
                        %--------------------------------------------------------------------------
                        %% eigenface extracting
                        [disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT, Par.nDim);
                        tr_dat  =  disc_set'*Tr_DAT;
                        tt_dat  =  disc_set'*Tt_DAT;
                        tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [Par.nDim,1]) );
                        tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [Par.nDim,1]) );
                        %-------------------------------------------------------------------------
                        %% testing
                        ID = [];
                        for indTest = 1:size(tt_dat,2)
                            switch ClassificationMethod
                                case 'SRC'
                                    rel_tol = 0.01;     % relative target duality gap
                                    [coef, status]=l1_ls(tr_dat, tt_dat(:,indTest), Par.lambda, rel_tol);
                                case 'CRC'
                                    Par.lambda = .001 * size(Tr_DAT,2)/700;
                                    %projection matrix computing
                                    Proj_M = (tr_dat'*tr_dat+Par.lambda*eye(size(tr_dat,2)))\tr_dat';
                                    coef         =  Proj_M*tt_dat(:,indTest);
                                case 'NNLSR'                   % non-negative
                                    coef = NNLS( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                    %                                     coef = NNLSR( tt_dat(:,indTest), tr_dat, Par );
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
                        accuracy(i, 1)         =   [cornum/length(ttls)]; % recognition rate
                        fprintf(['Accuracy is ' num2str(accuracy(i, 1)) '.\n']);
                    end
                    % -------------------------------------------------------------------------
                    %% save the results
                    avgacc = mean(accuracy);
                    fprintf(['Mean Accuracy is ' num2str(avgacc) '.\n']);
                    if strcmp(ClassificationMethod, 'NSC') == 1 || strcmp(ClassificationMethod, 'SRC') == 1 || strcmp(ClassificationMethod, 'CRC') == 1
                        matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' ClassificationMethod '_DR' num2str(Par.nDim) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname, 'accuracy', 'avgacc');
                    elseif strcmp(ClassificationMethod, 'CROC') == 1
                        matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' ClassificationMethod '_DR' num2str(Par.nDim) '_lambda' num2str(Par.lambda) '_weight' num2str(Par.rho) '.mat']);
                        save(matname, 'accuracy', 'avgacc');
                    elseif strcmp(ClassificationMethod, 'ProCRC') == 1
                        matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' ClassificationMethod '_DR' num2str(Par.nDim) '_lambda' num2str(Par.lambda) '_gamma' num2str(Par.rho) '.mat']);
                        save(matname, 'accuracy', 'avgacc');
                    elseif strcmp(ClassificationMethod, 'NNLSR') == 1 || strcmp(ClassificationMethod, 'NPLSR') == 1 || strcmp(ClassificationMethod, 'ANNLSR') == 1 || strcmp(ClassificationMethod, 'ANPLSR') == 1
                        matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' ClassificationMethod '_DR' num2str(Par.nDim) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'accuracy', 'avgacc');
                    elseif strcmp(ClassificationMethod, 'DANNLSR') == 1 || strcmp(ClassificationMethod, 'DANPLSR') == 1
                        matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' ClassificationMethod '_DR' num2str(Par.nDim) '_scale' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'accuracy', 'avgacc');
                    end
                end
            end
        end
    end
end



