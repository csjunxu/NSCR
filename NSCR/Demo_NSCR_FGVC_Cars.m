clear,clc;
maxNumCompThreads(1);
% -------------------------------------------------------------------------
%% choosing the dataset
directory = '../../data/classification/';
dataset = 'cars';
% -------------------------------------------------------------------------
%% number of repeations
nDimArray = [4096];
nExperiment = 1;
SampleArray = 0;
% -------------------------------------------------------------------------
%% directory to save the results
writefilepath  = [ directory dataset '_results/'];
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
% ClassificationMethod = 'NRC' ; % non-negative LSR
ClassificationMethod = 'NSCR' ; % non-negative joint sparse and collaborative
%-------------------------------------------------------------------------
%% PCA dimension
for nDim = nDimArray
    Par.nDim = nDim;
    for nSample = SampleArray
        %-------------------------------------------------------------------------
        %% tuning the parameters
        %         for mu = 1
        %             Par.mu = mu;
        for maxIter = [1:20]
            Par.maxIter  = maxIter;
        for rho = [.1 1 10 100]
            Par.rho = rho;
            for alpha = [0 0.001 0.01 .1 1]
                Par.alpha = alpha;
                for beta = [0 0.001 0.01 .1 1]
                    Par.beta = beta;
                        accuracy = zeros(nExperiment, 1) ;
                        for n = 1:nExperiment
                            existID  = ['TempID_' dataset '_' ClassificationMethod '_D' num2str(Par.nDim) '_mIte' num2str(Par.maxIter) '_r' num2str(Par.rho) '_a' num2str(Par.alpha) '_b' num2str(Par.beta)  '_' num2str(nExperiment) '.mat'];
                            %--------------------------------------------------------------------------
                            %% data loading
                            load(['../../data/classification/' dataset]);
                            [dim, N] = size(trainFV);
                            nClass        =   max(trainY);
                            Tr_DAT   =   double(trainFV);
                            trls     =   trainY';
                            Tt_DAT   =   double(valFV);
                            ttls     =   valY';
                            clear trainFV valFV trainY valY
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
                                XTXinv = (tr_dat' * tr_dat + (Par.alpha+Par.rho/2) * eye(N))\eye(N);
                            else
                                XTXinv = (2/(2*Par.alpha+Par.rho) * eye(N) - (2/(2*Par.alpha+Par.rho))^2 * tr_dat' / (2/(2*Par.alpha+Par.rho) * (tr_dat * tr_dat') + eye(D)) * tr_dat );
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
                                %coef = NRC( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                coef = NSCR( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                [id, ~] = PredictID(coef, tr_dat, trls, class_num);
                                ID      =   [ID id];
                                e = cputime-t;
                                fprintf([num2str(TestSeg(set+1)) '/' num2str(size(tt_dat,2)) ': ' num2str(e) '\n']);
                                save(existID, 'ID');
                            end
                            cornum         =   sum(ID==ttls);
                            accuracy(n, 1) =   [cornum/length(ttls)]; % recognition rate
                            fprintf(['Accuracy is ' num2str(accuracy(n, 1)) '.\n']);
                            eval(['delete ' existID]);
                            pause(3);
                        end
                        % -------------------------------------------------------------------------
                        %% save the results
                        avgacc = mean(accuracy);
                        fprintf(['Mean Accuracy is ' num2str(avgacc) '.\n']);
                        if avgacc>=0.90
                            matname = sprintf([writefilepath dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' ClassificationMethod '_DR' num2str(Par.nDim) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_a' num2str(Par.alpha) '_b' num2str(Par.beta) '.mat']);
                            save(matname,'accuracy', 'avgacc');
                        end
                    end
                end
            end
        end
    end
end
% end