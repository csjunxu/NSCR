clear
% maxNumCompThreads(1);
warning off;

addpath('.\invChol');

dataDir = fullfile('C:\Users\csjunxu\Desktop\Classification\Dataset','imagenet12-feat-caffe-alex') ;
% dataDir = fullfile('C:\Users\csjunxu\Desktop\Classification\Dataset','imagenet12-sbow-split') ;

dirTrain=dir(fullfile(dataDir,'train_category_*.mat'));
TrainfileNames={dirTrain.name}';
dirTest=dir(fullfile(dataDir,'valid_category_*.mat'));
TestfileNames={dirTest.name}';

dataset = 'ImageNet';
nExperiment = 1;
nDimArray = [4096];
SampleArray = 0;
normalization.flag = 1;
normalization.type = 1;
% -------------------------------------------------------------------------
%% directory to save the results
writefilepath  = [dataset];
if ~isdir(writefilepath)
    mkdir(writefilepath);
end

ClassificationMethod = 'NNLS' ;

load 'imagenet_trdata.mat';
load 'imagenet_ttdata.mat';

%% PCA dimension
for nDim = nDimArray
    Par.nDim = nDim;
    for nSample = SampleArray
        %% tuning the parameters
        for s = [1]
            Par.s = s;
            for maxIter = [2:1:4]
                Par.maxIter  = maxIter;
                for rho = [.3:.2:.7]
                    Par.rho = rho;
                    for lambda = [0]
                        Par.lambda = lambda;
                        accuracy = zeros(nExperiment, 1) ;
                        for n = 1:nExperiment
                            [D, N] = size(tr_dat);
                            if N < D
                                XTXinv = (tr_dat' * tr_dat + Par.rho/2 * eye(N))\eye(N);
                            else
                                XTXinv = (2/Par.rho * eye(N) - (2/Par.rho)^2 * tr_dat' / (2/Par.rho * (tr_dat * tr_dat') + eye(D)) * tr_dat );
                            end
                            %% testing
                            class_num = max(trls);
                            %% load finished IDs
                            existID  = ['TempID_' dataset '.mat'];
                            if exist(existID)
                                eval(['load ' existID]);
                            else
                                ID = [];
                            end
                            for indTest = size(ID)+1:size(tt_dat,2)
                                t = cputime;
                                coef = NNLS( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                %% assign the class  index
                                [id, ~] = PredictID(coef, tr_dat, trls, class_num);
                                ID      =   [ID id];
                                e = cputime-t;
                                fprintf([num2str(indTest) '/' num2str(size(tt_dat,2)) ': ' num2str(e) '\n']);
                                save(existID, 'ID');
                            end
                            cornum      =   sum(ID==ttls);
                            accuracy(n, 1)         =   [cornum/length(ttls)]; % recognition rate
                            fprintf(['Accuracy is ' num2str(accuracy(n, 1)) '.\n']);
                            eval(['delete ' existID]);
                        end
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


