clear
warning off;
addpath('.\invChol');
maxNumCompThreads(1);
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
%-------------------------------------------------------------------------
%% data loading
load 'imagenet_trdata.mat';
load 'imagenet_ttdata.mat';
%-------------------------------------------------------------------------
%% Top-k accuracy
Top = 1; % 1 or 5
%-------------------------------------------------------------------------
%% PCA dimension
for nDim = nDimArray
    Par.nDim = nDim;
    for nSample = SampleArray
        %% tuning the parameters
        for s = [1]
            Par.s = s;
            for maxIter = [3:1:9]
                Par.maxIter  = maxIter;
                for rho = [.1:.1:1]
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
                            existID  = ['TempID_' dataset '_' ClassificationMethod '_D' num2str(Par.nDim) '_s' num2str(Par.s) '_mIte' num2str(Par.maxIter) '_r' num2str(Par.rho) '_l' num2str(Par.lambda)  '_' num2str(nExperiment) '.mat'];
                            if exist(existID)==2
                                eval(['load ' existID]);
                            else
                                ID = [];
                            end
                            t = cputime;
                            TestSeg = length(ID):5000:size(tt_dat,2);
                            TestSeg = [TestSeg size(tt_dat,2)];
                            for set  = 1:length(TestSeg)-1
                                indTest = TestSeg(set)+1:TestSeg(set+1);
                                coef = NNLS( tt_dat(:,indTest), tr_dat, XTXinv, Par );
                                %% assign the class  index
                                [id, ~] = PredictIDTop(coef, tr_dat, trls, class_num, Top);
                                % [id, ~] = PredictID(coef, tr_dat, trls, class_num);
                                ID      =   [ID id];
                                e = cputime-t;
                                fprintf([num2str(TestSeg(set+1)) '/' num2str(size(tt_dat,2)) ': ' num2str(e) '\n']);
                                save(existID, 'ID');
                            end
                            %% compute accuracy
                            ttlsTop = repmat(ttls, [Top 1]);
                            cornum      =   sum(sum(ID==ttlsTop, 1));
                            accuracy(n, 1)         =   [cornum/length(ttls)]; % recognition rate
                            fprintf(['Accuracy is ' num2str(accuracy(n, 1)) '.\n']);
                            eval(['delete ' existID]);
                            pause(3);
                        end
                        %% save the results
                        avgacc = mean(accuracy);
                        fprintf(['Mean Accuracy is ' num2str(avgacc) '.\n']);
                        matname = sprintf([writefilepath '/' dataset '_' num2str(nSample(1)) '_' num2str(nExperiment) '_' ClassificationMethod '_DR' num2str(Par.nDim) '_scale' num2str(Par.s) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '_Top' num2str(Top) '.mat']);
                        save(matname,'accuracy', 'avgacc');
                    end
                end
            end
        end
    end
end


