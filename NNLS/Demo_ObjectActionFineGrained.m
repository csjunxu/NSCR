clear;
% -------------------------------------------------------------------------
%% choosing the dataset
dataset = 'Caltech-256_VGG';
% Flower-102_VGG
% CUB-200-2011_VGG
% Standford-40_VGG
% cifar-10
% cifar-100
% Caltech-256_VGG
% -------------------------------------------------------------------------
%% number of repeations
if strcmp(dataset, 'CUB-200-2011_VGG') == 1
    nExperiment = 1;
    nDimArray = [4096];
elseif strcmp(dataset, 'Flower-102_VGG') == 1
    v = 1;
    nDimArray = [4096];
elseif strcmp(dataset, 'Standford-40_VGG') == 1
    nExperiment = 1;
    nDimArray = [4096];
elseif strcmp(dataset, 'Caltech-256_VGG') == 1
    nExperiment = 1;
    nDimArray = [4096];
    SampleArray = [60 45 30 15];
elseif strcmp(dataset, 'cifar-100') == 1 || strcmp(dataset, 'cifar-10') == 1
    nExperiment = 10;
    nDimArray = [3072];
    SampleArray = [50 100 300 500];
end
% -------------------------------------------------------------------------
%% directory to save the results
writefilepath  = ['C:/Users/csjunxu/Desktop/Classification/Results/' dataset '/'];
if ~isdir(writefilepath) 
    mkdir(writefilepath);
end
% -------------------------------------------------------------------------
%% choosing classification methods
% ClassificationMethod = 'NSC';
% ClassificationMethod = 'SRC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\l1_ls_matlab'));
ClassificationMethod = 'CRC';
% ClassificationMethod = 'CROC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\CROC CVPR2012'));
% ClassificationMethod = 'ProCRC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\ProCRC'));
% ClassificationMethod = 'NNLSR' ; % non-negative LSR
% ClassificationMethod = 'NPLSR' ; % non-positive LSR
% ClassificationMethod = 'ANNLSR' ; % affine and non-negative LSR
% ClassificationMethod = 'ANPLSR' ; % affine and non-positive LSR
% ClassificationMethod = 'DANNLSR' ; % deformable, affine and non-negative LSR
% ClassificationMethod = 'DANPLSR' ; % deformable, affine and non-positive LSR
% ClassificationMethod = 'ADANNLSR' ; % deformable, affine and non-negative LSR
% ClassificationMethod = 'ADANPLSR' ; % deformable, affine and non-positive LSR
%-------------------------------------------------------------------------
%% PCA dimension
for nDim = nDimArray
    Par.nDim = nDim;
    for nSample = SampleArray
        %-------------------------------------------------------------------------
        %% tuning the parameters
        for s = [1]
            Par.s = s;
            for maxIter = [5]
                Par.maxIter  = maxIter;
                for rho = [.1]
                    Par.rho = rho;
                    for lambda = [0]
                        Par.lambda = lambda;
                        accuracy = zeros(nExperiment, 1) ;
                        for n = 1:nExperiment
                            %--------------------------------------------------------------------------
                            %% data loading
                            if strcmp(dataset, 'Caltech-256_VGG') == 1
                                load(['C:/Users/csjunxu/Desktop/Classification/Dataset/' dataset]);
                                % randomly select half of the samples as training data;
                                [dim, N] = size(descr);
                                nClass = length(unique(label));
                                % nClass is the number of classes in the subset of AR database
                                Tr_DAT = [];
                                Tt_DAT = [];
                                trls = [];
                                ttls = [];
                                for i=1:nClass
                                    descri = descr(:, label==i);
                                    Ni = size(descri, 2);
                                    rng(n);
                                    RpNi = randperm(Ni);
                                    Tr_DAT   =   [Tr_DAT double(descri(:, RpNi(1:nSample)))];
                                    trls     =   [trls i*ones(1, nSample)];
                                    Tt_DAT   =   [Tt_DAT double(descri(:, RpNi(nSample+1:end)))];
                                    ttls     =   [ttls i*ones(1, Ni-nSample)];
                                end
                                clear descr label descri RpNi Ni
                            elseif strcmp(dataset, 'cifar-10') == 1
                                            % training data
                                Tr_DATall = [];
                                trlsall = [];
                                for i=1:1:5
                                    load(['C:/Users/csjunxu/Desktop/Classification/Dataset/' dataset '/data_batch_' num2str(i)]);
                                    Tr_DATall = [Tr_DATall double(data')];
                                    trlsall     =   [trlsall labels'];
                                end
                                if min(trlsall)==0
                                    trlsall = trlsall + 1;
                                end
                                % randomly select half of the samples as training data
                                [dim, N] = size(Tr_DATall);
                                nClass = length(unique(trlsall));
                                % nClass is the number of classes in the subset of AR database
                                Tr_DAT = [];
                                trls = [];
                                for i=1:nClass
                                    Tr_DATi = Tr_DATall(:, trlsall==i);
                                    Ni = size(Tr_DATi, 2);
                                    rng(n);
                                    RpNi = randperm(Ni);
                                    Tr_DAT   =   [Tr_DAT double(Tr_DATi(:, RpNi(1:nSample)))];
                                    trls     =   [trls i*ones(1, nSample)];
                                end
                                clear Tr_DATall Tr_DATi trlsall RpNi Ni
                                % testing data
                                load(['C:/Users/csjunxu/Desktop/Classification/Dataset/' dataset '/test_batch']);
                                Tt_DAT = double(data');
                                ttls = labels';
                                if min(ttls)==0
                                    ttls = ttls + 1;
                                end
                                clear data labels
                            elseif strcmp(dataset, 'cifar-100') == 1
                                load(['C:/Users/csjunxu/Desktop/Classification/Dataset/' dataset '/train']);
                                % training data: randomly select half of the samples as training data;
                                data = data';
                                [dim, N] = size(data);
                                nClass = length(unique(fine_labels));
                                Tr_DAT = [];
                                trls = [];
                                for i=1:nClass
                                    datai = data(:,fine_labels==i-1);
                                    Ni = size(datai, 2);
                                    rng(n);
                                    RpNi = randperm(Ni);
                                    Tr_DAT   =   [Tr_DAT double(datai(:, RpNi(1:nSample)))];
                                    trls     =   [trls i*ones(1, nSample)];
                                end
                                clear data datai fine_label Ni RpNi
                                % testing data
                                load(['C:/Users/csjunxu/Desktop/Classification/Dataset/' dataset '/test']);
                                Tt_DAT = double(data');
                                ttls = fine_labels + 1;
                            else
                                load(['C:/Users/csjunxu/Desktop/Classification/Dataset/' dataset]);
                                nClass        =   max(tr_label);
                                Tr_DAT   =   double(tr_descr);
                                trls     =   tr_labels;
                                Tt_DAT   =   double(tt_descr);
                                ttls     =   tt_labels;
                                clear tr_descr tt_descr tr_labels tt_labels
                            end
                            %--------------------------------------------------------------------------
                            %% eigenface extracting
                            if Par.nDim == 0 || Par.nDim == dim
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
                            if strcmp(ClassificationMethod, 'CROC') == 1
                                weight = Par.rho;
                                ID = croc_cvpr12(tt_dat, tr_dat, trls, Par.lambda, weight);
                                % ID = croc_cvpr12_v0(tt_dat, tr_dat, trls, Par.lambda, weight);
                            else
                                ID = [];
                                for indTest = 1:size(tt_dat,2)
                                    switch ClassificationMethod
                                        case 'SRC'
                                            rel_tol = 0.01;     % relative target duality gap
                                            [coef, status]=l1_ls(tr_dat, tt_dat(:,indTest), Par.lambda, rel_tol);
                                        case 'CRC'
                                            Par.lambda = .001 * size(Tr_DAT,2)/700;
                                            % projection matrix computing
                                            Proj_M = (tr_dat'*tr_dat+Par.lambda*eye(size(tr_dat,2)))\tr_dat';
                                            coef         =  Proj_M*tt_dat(:,indTest);
                                            %                                 case 'CROC'
                                            %                                     [min_idx] = croc_cvpr12(testFea, tr_dat, trainGnd, lambda, weight);
                                        case 'ProCRC'
                                            params.dataset_name      =      'Extended Yale B';
                                            params.model_type        =      'ProCRC';
                                            params.gamma             =     Par.rho; % [1e-2];
                                            params.lambda            =      Par.lambda; % [1e-0];
                                            params.class_num         =      max(trls);
                                            data.tr_descr = tr_dat;
                                            data.tt_descr = tt_dat(:,indTest);
                                            data.tr_label = trls;
                                            data.tt_label = ttls;
                                            coef = ProCRC(data, params);
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
                                        case 'ADANNLSR'                 % affine, non-negative, sum to a scalar s
                                            coef = ADANNLSR( tt_dat(:,indTest), tr_dat, Par );
                                        case 'ADANPLSR'             % affine, non-positive, sum to a scalar -s
                                            coef = ADANPLSR( tt_dat(:,indTest), tr_dat, Par );
                                    end
                                    % -------------------------------------------------------------------------
                                    %% assign the class  index
                                    if strcmp(ClassificationMethod, 'NSC') == 1
                                        for ci = 1:max(trls)
                                            Xc = tr_dat(:, trls==ci);
                                            Aci = Xc/(Xc'*Xc+Par.lambda*eye(size(Xc, 2)))*Xc';
                                            coef_c = Aci*tt_dat(:,indTest);
                                            error(ci) = norm(tt_dat(:,indTest)-coef_c,2)^2/sum(coef_c.*coef_c);
                                        end
                                    else
                                        for ci = 1:max(trls)
                                            coef_c   =  coef(trls==ci);
                                            Dc       =  tr_dat(:,trls==ci);
                                            error(ci) = norm(tt_dat(:,indTest)-Dc*coef_c,2)^2/sum(coef_c.*coef_c);
                                        end
                                    end
                                    index      =  find(error==min(error));
                                    id         =  index(1);
                                    ID      =   [ID id];
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