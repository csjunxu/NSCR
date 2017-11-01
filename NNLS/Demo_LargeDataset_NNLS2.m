clear;
% -------------------------------------------------------------------------
%% choosing the dataset
dataset = 'Flower-102_VGG';
% Flower-102_VGG
% CUB-200-2011_VGG
% aircraft
% cars
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
        || strcmp(dataset, 'CUB-200-2011_VGG') == 1 ...
        || strcmp(dataset, 'aircraft') == 1 ...
        || strcmp(dataset, 'cars') == 1
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
            for maxIter = [3:1:6]
                Par.maxIter  = maxIter;
                for rho = [.2:.2:10]
                    Par.rho = rho;
                    for lambda = [0]
                        Par.lambda = lambda;
                        accuracy = zeros(nExperiment, 1) ;
                        for n = 1:nExperiment
                            existID  = ['TempID_' dataset '_' ClassificationMethod '_D' num2str(Par.nDim) '_s' num2str(Par.s) '_mIte' num2str(Par.maxIter) '_r' num2str(Par.rho) '_l' num2str(Par.lambda)  '_' num2str(nExperiment) '.mat'];
                            %--------------------------------------------------------------------------
                            %% data loading
                            if strcmp(dataset, 'Caltech-256_VGG') == 1
                                load(['C:/Users/csjunxu/Desktop/CVPR2018 Classification/Dataset/' dataset]);
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
                                load(['C:/Users/csjunxu/Desktop/CVPR2018 Classification/Dataset/' dataset]);
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
                                load(['C:/Users/csjunxu/Desktop/CVPR2018 Classification/Dataset/' dataset]);
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
                                load(['C:/Users/csjunxu/Desktop/CVPR2018 Classification/Dataset/' dataset]);
                                [dim, N] = size(TrData);
                                nClass        =   max(TrLabel);
                                Tr_DAT   =   double(TrData);
                                trls     =   TrLabel;
                                Tt_DAT   =   double(TtData);
                                ttls     =   TtLabel;
                                clear TrData TtData TrLabel TtLabel
                            elseif strcmp(dataset, 'aircraft') == 1 ...
                                    || strcmp(dataset, 'cars') == 1
                                load(['C:/Users/csjunxu/Desktop/CVPR2018 Classification/Dataset/' dataset]);
                                [dim, N] = size(trainFV);
                                nClass        =   max(trainY);
                                Tr_DAT   =   double(trainFV);
                                trls     =   trainY';
                                Tt_DAT   =   double(valFV);
                                ttls     =   valY';
                                clear trainFV valFV trainY valY
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
                            [D, N] = size(tr_dat);
                            if N < D
                                XTXinv = (tr_dat' * tr_dat + Par.rho/2 * eye(N))\eye(N);
                            else
                                XTXinv = (2/Par.rho * eye(N) - (2/Par.rho)^2 * tr_dat' / (2/Par.rho * (tr_dat * tr_dat') + eye(D)) * tr_dat );
                            end
                            %% load finished IDs
                            if exist(existID)==2
                                eval(['load ' existID]);
                            else
                                ID = [];
                            end
                            t = cputime;
                            TestSeg = length(ID):3000:size(tt_dat,2);
                            TestSeg = [TestSeg size(tt_dat,2)];
                            for set  = 1:length(TestSeg)-1
                                indTest = TestSeg(set)+1:TestSeg(set+1);
                                coef = NNLS( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                [id, ~] = PredictID(coef, tr_dat, trls, class_num);
                                ID      =   [ID id];
                                e = cputime-t;
                                fprintf([num2str(TestSeg(set+1)) '/' num2str(size(tt_dat,2)) ': ' num2str(e) '\n']);
                                save(existID, 'ID');
                            end
                            cornum      =   sum(ID==ttls);
                            accuracy(n, 1)         =   [cornum/length(ttls)]; % recognition rate
                            fprintf(['Accuracy is ' num2str(accuracy(n, 1)) '.\n']);
                            eval(['delete ' existID]);
                            pause(3);
                        end
                        % -------------------------------------------------------------------------
                        %% save the results
                        avgacc = mean(accuracy);
                        fprintf(['Mean Accuracy is ' num2str(avgacc) '.\n']);
                        matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' ClassificationMethod '_DR' num2str(Par.nDim) '_scale' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
                        save(matname,'accuracy', 'avgacc');
                    end
                end
            end
        end
    end
end