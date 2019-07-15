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
ClassificationMethod = 'NCR' ; % non-negative LSR
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
    nDimArray = [54 120 300];
end
% -------------------------------------------------------------------------
%% directory to save the results
writefilepath  = [ dataset '/'];
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
    for s = [0]
        Par.s = s;
        for maxIter = [1:20]
            Par.maxIter  = maxIter;
            for mu = 1
                par.mu = mu;
                for rho = [0:1:6]
                    Par.rho = 10^(-rho);
                    for lambda = [0 0.01 .05 .1 .5]
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
                            class_num = max(trls);
                            [D, N] = size(tr_dat);
                            if N < D
                                XTXinv = (tr_dat' * tr_dat + Par.rho/2 * eye(N))\eye(N);
                            else
                                XTXinv = (2/Par.rho * eye(N) - (2/Par.rho)^2 * tr_dat' / (2/Par.rho * (tr_dat * tr_dat') + eye(D)) * tr_dat );
                            end
                            t = cputime;
                            if strcmp(ClassificationMethod, 'CRC') == 1
                                Par.lambda = .001 * size(Tr_DAT,2)/700;
                                % projection matrix computing
                                Proj_M = (tr_dat'*tr_dat+Par.lambda*eye(size(tr_dat,2)))\tr_dat';
                                coef         =  Proj_M*tt_dat(:,indTest);
                                [ID, ~] = PredictID(coef, tr_dat, trls, class_num);
                            elseif strcmp(ClassificationMethod, 'CROC') == 1
                                weight = Par.rho;
                                ID = croc_cvpr12(tt_dat, tr_dat, trls, Par.lambda, weight);
                                % ID = croc_cvpr12_v0(tt_dat, tr_dat, trls, Par.lambda, weight);
                            elseif strcmp(ClassificationMethod, 'ProCRC') == 1
                                global params
                                set_params(dataset);
                                data.tr_descr = tr_dat;
                                data.tt_descr = tt_dat;
                                data.tr_label = trls;
                                data.tt_label = ttls;
                                params.class_num = class_num;
                                % params.model_type        =      'ProCRC';
                                % params.gamma             =     Par.rho; % [1e-2];
                                % params.lambda            =      Par.lambda; % [1e-0];
                                % params.class_num         =      max(trls);
                                coef = ProCRC(data, params);
                                [ID, ~] = ProMax(coef, data, params);
                            elseif strcmp(ClassificationMethod, 'NCR') == 1
                                %coef = NNLS( tt_dat, tr_dat, XTXinv, Par );
                                coef = NCR( tt_dat, tr_dat, Par );
                                [ID, ~] = PredictIDTop(coef, tr_dat, trls, class_num, Top);
                            elseif strcmp(ClassificationMethod, 'DANNLSR') == 1 % affine, non-negative, sum to a scalar s
                                coef = DANNLSR( tt_dat, tr_dat, XTXinv, Par );
                                [ID, ~] = PredictID(coef, tr_dat, trls, class_num);
                            else
                                % -------------------------------------------------------------------------
                                %% load finished IDs
                                existID  = ['TempID_' dataset '_' ClassificationMethod '_D' num2str(Par.nDim) '_s' num2str(Par.s) '_mIte' num2str(Par.maxIter) '_r' num2str(Par.rho) '_l' num2str(Par.lambda)  '_' num2str(nExperiment) '.mat'];
                                if exist(existID)==2
                                    eval(['load ' existID]);
                                else
                                    ID = [];
                                end
                                if strcmp(ClassificationMethod, 'SRC') == 1
                                    for indTest = size(ID)+1:size(tt_dat,2)
                                        t = cputime;
                                        rel_tol = 0.01;     % relative target duality gap
                                        [coef, status]=l1_ls(tr_dat, tt_dat(:,indTest), Par.lambda, rel_tol);
                                        [id, ~] = PredictID(coef, tr_dat, trls, class_num);
                                        ID      =   [ID id];
                                        e = cputime-t;
                                        fprintf([num2str(indTest) '/' num2str(size(tt_dat,2)) ': ' num2str(e) '\n']);
                                        save(existID, 'ID');
                                    end
                                elseif strcmp(ClassificationMethod, 'NSC') == 1
                                    for indTest = size(ID)+1:size(tt_dat,2)
                                        for ci = 1:class_num
                                            Xc = tr_dat(:, trls==ci);
                                            Aci = Xc/(Xc'*Xc+Par.lambda*eye(size(Xc, 2)))*Xc';
                                            coef_c = Aci*tt_dat(:,indTest);
                                            error(ci) = norm(tt_dat(:,indTest)-coef_c,2)^2/sum(coef_c.*coef_c);
                                        end
                                        index      =  find(error==min(error));
                                        id         =  index(1);
                                        ID      =   [ID id];
                                        e = cputime-t;
                                        fprintf([num2str(indTest) '/' num2str(size(tt_dat,2)) ': ' num2str(e) '\n']);
                                        save(existID, 'ID');
                                    end
                                end
                                eval(['delete ' existID]);
                                pause(3);
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
                        if strcmp(ClassificationMethod, 'SRC') == 1 || strcmp(ClassificationMethod, 'CRC') == 1
                            matname = sprintf([writefilepath dataset '_' ClassificationMethod '_DR' num2str(Par.nDim) '.mat']);
                            save(matname, 'accuracy', 'avgacc');
                        elseif strcmp(ClassificationMethod, 'ProCRC') == 1 || strcmp(ClassificationMethod, 'CROC') == 1
                            matname = sprintf([writefilepath dataset '_' ClassificationMethod '_DR' num2str(Par.nDim) '_lambda' num2str(Par.lambda) '_weight' num2str(Par.rho) '.mat']);
                            save(matname, 'accuracy', 'avgacc');
                        elseif strcmp(ClassificationMethod, 'DANNLSR') == 1 || strcmp(ClassificationMethod, 'ANNLSR') == 1
                            matname = sprintf([writefilepath dataset '_' ClassificationMethod '_DR' num2str(Par.nDim) '_scale' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                            save(matname,'accuracy', 'avgacc');
                        else   matname = sprintf([writefilepath dataset '_' ClassificationMethod '_DR' num2str(Par.nDim) '_scale' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda)  '_Top' num2str(Top) '.mat']);
                            save(matname,'accuracy', 'avgacc');
                        end
                    end
                end
            end
        end
    end
end