clear;
addpath('C:\Users\csjunxu\Desktop\SC\Datasets\MNISThelpcode');
addpath(genpath('C:\Users\csjunxu\Documents\GitHub\SubspaceCluteringCode\SSCOMP_Code\scatnet-0.2'));

% -------------------------------------------------------------------------
%% choosing the dataset
dataset = 'MNIST';
% MNIST2k2k
% MNIST
% USPS
% -------------------------------------------------------------------------
%% number of repeations
if  strcmp(dataset, 'MNIST2k2k') == 1
    nExperiment = 10;
elseif strcmp(dataset, 'MNIST') == 1
    nExperiment = 10;
elseif strcmp(dataset, 'USPS') == 1
    nExperiment = 10;
end
% -------------------------------------------------------------------------
%% choosing classification methods
ClassificationMethod = 'SRC'; addpath(genpath('l1_ls_matlab'));
% ClassificationMethod = 'CRC';
% ClassificationMethod = 'NNLSR' ; % non-negative LSR
% ClassificationMethod = 'NPLSR' ; % non-positive LSR
% ClassificationMethod = 'ANNLSR' ; % affine and non-negative LSR
% ClassificationMethod = 'ANPLSR' ; % affine and non-positive LSR
% ClassificationMethod = 'DANNLSR' ; % deformable, affine and non-negative LSR
% ClassificationMethod = 'DANPLSR' ; % deformable, affine and non-positive LSR
% -------------------------------------------------------------------------
%% directory to save the results
writefilepath  = ['C:/Users/csjunxu/Desktop/Classification/Results/' dataset '/'];
if ~isdir(writefilepath)
    mkdir(writefilepath);
end

%% Settings
Par.nDim = 500;

for nSample = [50 100 300 600] % number of images for each digit
    %-------------------------------------------------------------------------
    %% tuning the parameters
    for s = [1]
        Par.s = s;
        for maxIter = [5]
            Par.maxIter  = maxIter;
            for rho = [1]
                Par.rho = rho*10^(-1);
                for lambda = [.1 1 10]
                    Par.lambda = lambda*10^(-1);
                    accuracy = zeros(nExperiment, 1) ;
                    for i = 1:nExperiment
                        
                        %--------------------------------------------------------------------------
                        %% data loading
                        if strcmp(dataset, 'MNIST2k2k') == 1
                            load(['C:/Users/csjunxu/Desktop/Classification/Dataset/MNIST2k2k']);
                            gnd = gnd + 1;
                            Par.nClass        =   max(gnd); % the number of classes in the subset of AR database
                            %                         Par.nDim          =   100; % the eigenfaces dimension
                            Tr_DAT   =   double(fea(trainIdx, :))';
                            trls     =   gnd(trainIdx, 1)';
                            Tt_DAT   =   double(fea(testIdx, :))';
                            ttls     =   gnd(testIdx, 1)';
                            clear fea gnd trainIdx testIdx
                        elseif strcmp(dataset, 'MNIST') == 1
                            %% Load data
                            addpath('C:\Users\csjunxu\Desktop\SC\Datasets\MNIST\')
                            if ~exist('tr_MNIST_DATA', 'var') || ~exist('tt_MNIST_DATA', 'var')
                                try
                                    % MNIST_SC_DATA is a D by N matrix. Each column contains a feature
                                    % vector of a digit image and N = 60,000.
                                    % MNIST_LABEL is a 1 by N vector. Each entry is the label for the
                                    % corresponding column in MNIST_SC_DATA.
                                    load tr_MNIST_C.mat tr_MNIST_C_DATA tr_MNIST_LABEL;
                                catch
                                    % training data
                                    tr_MNIST_DATA = loadMNISTImages('train-images.idx3-ubyte');
                                    tr_MNIST_LABEL = loadMNISTLabels('train-labels.idx1-ubyte');
                                    tr_MNIST_C_DATA = SCofDigits(tr_MNIST_DATA);
                                    save C:\Users\csjunxu\Desktop\SC\Datasets\tr_MNIST_C.mat tr_MNIST_C_DATA tr_MNIST_LABEL;
                                    % testing data
                                    tt_MNIST_DATA = loadMNISTImages('t10k-images.idx3-ubyte');
                                    tt_MNIST_LABEL = loadMNISTLabels('t10k-labels.idx1-ubyte');
                                    tt_MNIST_C_DATA = SCofDigits(tt_MNIST_DATA);
                                    save C:\Users\csjunxu\Desktop\SC\Datasets\tt_MNIST_C.mat tt_MNIST_C_DATA tt_MNIST_LABEL;
                                end
                                tr_MNIST_DATA = tr_MNIST_C_DATA;
                                tt_MNIST_DATA = tt_MNIST_C_DATA;
                            end
                            nCluster = 10;
                            % set of digits to test on, e.g. [2, 0]. Pick randomly if empty.
                            digit_set = 0:9;
                            % prepare data
                            if isempty(digit_set)
                                rng(i); Digits = randperm(10, nCluster) - 1;
                            else
                                Digits = digit_set;
                            end
                            if length(nSample) == 1
                                nSample = ones(1, nCluster) * nSample;
                            end
                            mask = zeros(1, sum(nSample));
                            gnd = zeros(1, sum(nSample));
                            nSample_cum = [0, cumsum(nSample)];
                            for iK = 1:nCluster % randomly take data for each digit.
                                allpos = find( tr_MNIST_LABEL == Digits(iK) );
                                rng( (i-1) * nCluster + iK );
                                selpos = allpos( randperm(length(allpos), nSample(iK)) );
                                
                                mask( nSample_cum(iK) + 1 : nSample_cum(iK+1) ) = selpos;
                                gnd( nSample_cum(iK) + 1 : nSample_cum(iK+1) ) = iK * ones(1, nSample(iK));
                            end
                            Tr_DAT = tr_MNIST_DATA(:, mask);
                            N = length(gnd);
                            
                            trls = gnd;
                            Tt_DAT   =   double(fea(testIdx, :))';
                            ttls     =   gnd(testIdx, 1)';
                        elseif strcmp(dataset, 'USPS')==1
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
                        accuracy(i, 1)         =   [cornum/length(ttls)]; % recognition rate
                        fprintf(['Accuracy is ' num2str(accuracy(i, 1)) '.\n']);
                    end
                    
                    % -------------------------------------------------------------------------
                    %% save the results
                    avgacc = mean(accuracy);
                    fprintf(['Mean Accuracy is ' num2str(avgacc) '.\n']);
                    if strcmp(ClassificationMethod, 'CRC') == 1 % strcmp(ClassificationMethod, 'SRC') == 1 ||
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



