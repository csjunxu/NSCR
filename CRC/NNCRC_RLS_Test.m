
clear;

% -------------------------------------------------------------------------
% parameter setting
dataset = 'AR_DAT';
% 'AR_DAT'
% 'MNIST2k2k'
writefilepath = '';
%--------------------------------------------------------------------------
%data loading (here we use the AR dataset as an example)
if strcmp(dataset, 'AR_DAT') == 1
    load(['Dataset/AR_DAT']);
    par.nClass        =   max(trainlabels);                 % the number of classes in the subset of AR database
    par.nDim          =   54;                 % the eigenfaces dimension
    Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=par.nClass));
    trls     =   trainlabels(trainlabels<=par.nClass);
    Tt_DAT   =   double(NewTest_DAT(:,testlabels<=par.nClass));
    ttls     =   testlabels(testlabels<=par.nClass);
    clear NewTest_DAT NewTrain_DAT testlabels trainlabels
elseif strcmp(dataset, 'MNIST2k2k') == 1
    load(['Dataset/MNIST2k2k']);
    gnd = gnd + 1;
    par.nClass        =   max(gnd);                 % the number of classes in the subset of AR database
    par.nDim          =   100;                 % the eigenfaces dimension
    Tr_DAT   =   double(fea(trainIdx, :))';
    trls     =   gnd(trainIdx, 1)';
    Tt_DAT   =   double(fea(testIdx, :))';
    ttls     =   gnd(testIdx, 1)';
    clear fea gnd trainIdx testIdx
end

%--------------------------------------------------------------------------
%eigenface extracting
[disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,par.nDim);
tr_dat  =  disc_set'*Tr_DAT;
tt_dat  =  disc_set'*Tt_DAT;
tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [par.nDim,1]) );
tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [par.nDim,1]) );

%% Subspace segmentation methods
% SegmentationMethod = 'NNCRC' ; % non-negative CRC
% SegmentationMethod = 'NPCRC' ; % non-positive CRC
SegmentationMethod = 'ANNCRC' ; % affine and non-negative CRC
% SegmentationMethod = 'ANPCRC' ; % affine and non-positive CRC

%-------------------------------------------------------------------------
% tuning the parameters
for maxIter = [1:1:10]
    Par.maxIter  = maxIter;
    for rho = [1:1:10]
        Par.rho = rho*10^(-2);
        for lambda = [0 .1 1 10]
            Par.lambda = lambda * 10^(-3);
            %-------------------------------------------------------------------------
            %testing
            ID = [];
            for indTest = 1:size(tt_dat,2)
                switch SegmentationMethod
                    case 'NNCRC'                   % non-negative
                        [id]    = NNCRC(tt_dat(:,indTest), tr_dat, Par, trls);
                    case 'NPCRC'               % non-negative, sum to 1
                        [id]    = NPCRC(tt_dat(:,indTest), tr_dat, Par, trls);
                    case 'ANNCRC'                 % affine, non-negative
                        [id]    = ANNCRC(tt_dat(:,indTest), tr_dat, Par, trls);
                    case 'ANPCRC'             % affine, non-negative, sum to 1
                        [id]    = ANPCRC(tt_dat(:,indTest), tr_dat, Par, trls);
                end
                
                ID      =   [ID id];
            end
            cornum      =   sum(ID==ttls);
            Rec         =   [cornum/length(ttls)]; % recognition rate
            fprintf(['recogniton rate is ' num2str(Rec)]);
            matname = sprintf([writefilepath dataset '_' SegmentationMethod '_DR' num2str(par.nDim) '_maxIter' num2str(Par.maxIter) '_rho' num2str(Par.rho) '_lambda' num2str(Par.lambda) '.mat']);
            save(matname,'Rec');
        end
    end
end