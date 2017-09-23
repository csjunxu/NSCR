clear;
maxNumCompThreads(1);
% -------------------------------------------------------------------------
%% choosing the dataset
dataset = 'Standford-40_sift';
% Flower-102_VGG
% CUB-200-2011_VGG
% Standford-40_VGG
% Caltech-256_VGG

% Flower-102_sift
% CUB-200-2011_sift
% Standford-40_sift
% Caltech-256_sift
% -------------------------------------------------------------------------
%% number of repeations
if strcmp(dataset, 'Standford-40_VGG') == 1 ...
        || strcmp(dataset, 'Flower-102_VGG') == 1 ...
        || strcmp(dataset, 'CUB-200-2011_VGG') == 1
    nExperiment = 1;
    nDimArray = [4096];
    SampleArray = 0;
elseif strcmp(dataset, 'Caltech-256_VGG') == 1
    nExperiment = 1;
    nDimArray = [4096];
    SampleArray = 30; %[60 45 30 15];
elseif strcmp(dataset, 'Standford-40_sift') == 1 ...
        || strcmp(dataset, 'Flower-102_sift') == 1 ...
        || strcmp(dataset, 'CUB-200-2011_sift') == 1
    nExperiment = 1;
    nDimArray = [5120];
    SampleArray = 0;
elseif strcmp(dataset, 'Caltech-256_sift') == 1
    nExperiment = 1;
    nDimArray = [5120];
    SampleArray = 30; %[60 45 30 15];
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
% ClassificationMethod = 'CRC';
% ClassificationMethod = 'CROC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\CROC CVPR2012'));
% ClassificationMethod = 'ProCRC'; addpath(genpath('C:\Users\csjunxu\Desktop\Classification\ProCRC'));
ClassificationMethod = 'NNLSR' ; % non-negative LSR
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
            for maxIter = [3:1:5]
                Par.maxIter  = maxIter;
                for rho = [.5:.1:1]
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
                            elseif strcmp(dataset, 'Caltech-256_sift') == 1
                                load(['C:/Users/csjunxu/Desktop/Classification/Dataset/' dataset]);
                                % randomly select half of the samples as training data;
                                [dim, N] = size(Data);
                                nClass = length(unique(Label));
                                % nClass is the number of classes in the subset of AR database
                                Tr_DAT = [];
                                Tt_DAT = [];
                                trls = [];
                                ttls = [];
                                for i=1:nClass
                                    Datai = Data(:, Label==i);
                                    Ni = size(Datai, 2);
                                    rng(n);
                                    RpNi = randperm(Ni);
                                    Tr_DAT   =   [Tr_DAT double(Datai(:, RpNi(1:nSample)))];
                                    trls     =   [trls i*ones(1, nSample)];
                                    Tt_DAT   =   [Tt_DAT double(Datai(:, RpNi(nSample+1:end)))];
                                    ttls     =   [ttls i*ones(1, Ni-nSample)];
                                end
                                clear Data Label Datai RpNi Ni
                            elseif strcmp(dataset, 'Standford-40_VGG') == 1 ...
                                    || strcmp(dataset, 'Flower-102_VGG') == 1 ...
                                    || strcmp(dataset, 'CUB-200-2011_VGG') == 1
                                load(['C:/Users/csjunxu/Desktop/Classification/Dataset/' dataset]);
                                [dim, N] = size(tr_descr);
                                nClass        =   max(tr_label);
                                Tr_DAT   =   double(tr_descr);
                                trls     =   tr_label;
                                Tt_DAT   =   double(tt_descr);
                                ttls     =   tt_label;
                                clear tr_descr tt_descr tr_labels tt_labels
                            elseif strcmp(dataset, 'Standford-40_sift') == 1 ...
                                    || strcmp(dataset, 'Flower-102_sift') == 1 ...
                                    || strcmp(dataset, 'CUB-200-2011_sift') == 1
                                load(['C:/Users/csjunxu/Desktop/Classification/Dataset/' dataset]);
                                [dim, N] = size(TrData);
                                nClass        =   max(TrLabel);
                                Tr_DAT   =   double(TrData);
                                trls     =   TrLabel;
                                Tt_DAT   =   double(TtData);
                                ttls     =   TtLabel;
                                clear TrData TtData TrLabel TtLabel
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
                            class_num = max(trls);
                            if strcmp(ClassificationMethod, 'CROC') == 1
                                weight = Par.rho;
                                ID = croc_cvpr12(tt_dat, tr_dat, trls, Par.lambda, weight);
                                % ID = croc_cvpr12_v0(tt_dat, tr_dat, trls, Par.lambda, weight);
                            elseif strcmp(ClassificationMethod, 'ProCRC') == 1
                                global params
                                set_params(dataset);
                                %                                 params.model_type        =      'ProCRC';
                                %                                 params.gamma             =     Par.rho; % [1e-2];
                                %                                 params.lambda            =      Par.lambda; % [1e-0];
                                %                                 params.class_num         =      max(trls);
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
                                        %                                         for ci = 1:max(trls)
                                        %                                             coef_c   =  coef(trls==ci);
                                        %                                             Dc       =  tr_dat(:,trls==ci);
                                        %                                             error(ci) = norm(tt_dat(:,indTest)-Dc*coef_c,2)^2/sum(coef_c.*coef_c);
                                        %                                         end
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