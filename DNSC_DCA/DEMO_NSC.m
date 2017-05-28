clear all;

Dataset = 'AR_DAT';

% -------------------------------------------------------------------------
% parameter setting
Par.maxIter = 100;
for lambda = [2]
    Par.lambda = lambda;
    for gamma = [0.1 0.5 1]
        Par.gamma = gamma;
        if strcmp(Dataset, 'USPS')==1
            %             lambda1 = 0.023; % 0.92576
            %--------------------------------------------------------------------------
            %data loading (USPS dataset as an example of Handwritten digit recognition)
            load('../USPS');
            par.nClass        =  length(unique(gnd));
            Tr_DAT   =   double(fea(1:7291, :)');
            trls     =   gnd(1:7291)';
            Tt_DAT   =   double(fea(7292:end, :)');
            ttls     =   gnd(7292:end)';
            Tr_DAT  =  Tr_DAT./( repmat(sqrt(sum(Tr_DAT.*Tr_DAT)), [size(Tr_DAT, 1), 1]) );
            Tt_DAT  =  Tt_DAT./( repmat(sqrt(sum(Tt_DAT.*Tt_DAT)), [size(Tt_DAT, 1), 1]) );
            clear fea gnd
            %-------------------------------------------------------------------------
            %projection matrix computing for each class
            A = cell(max(trls),1);
            for c = 1:max(trls)
                Xc = Tr_DAT(:, trls==c);
                Xnc = Tr_DAT(:, trls~=c);
                A{c} = DNSC_DCA(Xc, Xnc, Par);
            end
            
            %-------------------------------------------------------------------------
            %testing
            ID = [];
            for indTest = 1:size(Tt_DAT, 2)
                [id]    = DNSC(A,Tt_DAT(:, indTest),trls);
                ID      =   [ID id];
            end
            cornum      =   sum(ID==ttls);
            Rec         =   [cornum/length(ttls)]; % recognition rate
            fprintf(['lambda = ' num2str(lambda) ', gamma =' num2str(gamma)  ', recogniton rate is ' num2str(Rec) '.\n']);
            
        elseif strcmp(Dataset, 'AR_DAT')==1
            %         lambda1 = 0.015; % 0.91559
            %--------------------------------------------------------------------------
            %data loading (here we use the AR dataset as an example)
            load(['../AR_DAT']);
            par.nClass        =  length(unique(trainlabels));              % the number of classes in the subset of AR database
            par.nDim          =   300;                 % the eigenfaces dimension
            Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=par.nClass));
            trls     =   trainlabels(trainlabels<=par.nClass);
            Tt_DAT   =   double(NewTest_DAT(:,testlabels<=par.nClass));
            ttls     =   testlabels(testlabels<=par.nClass);
            clear NewTest_DAT NewTrain_DAT testlabels trainlabels
            %--------------------------------------------------------------------------
            % eigenface extracting
            [disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,par.nDim);
            tr_dat  =  disc_set'*Tr_DAT;
            tt_dat  =  disc_set'*Tt_DAT;
            tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [par.nDim,1]) );
            tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [par.nDim,1]) );
            %-------------------------------------------------------------------------
            %projection matrix computing for each class
            
             for c = 1:max(trls)
                Xc = Tr_DAT(:, trls==c);
            end
            A = cell(max(trls),1);
            for c = 1:max(trls)
                Xc = Tr_DAT(:, trls==c);
                Xnc = Tr_DAT(:, trls~=c);
                A{c} = DNSC_DCA(Xc, Xnc, Par);
            end
            
            %-------------------------------------------------------------------------
            %testing
            ID = [];
            for indTest = 1:size(tt_dat,2)
                [id]    = NSC(A,tt_dat(:,indTest),trls);
                ID      =   [ID id];
            end
            cornum      =   sum(ID==ttls);
            Rec         =   [cornum/length(ttls)]; % recognition rate
            fprintf(['lambda = ' num2str(lambda) ', gamma =' num2str(gamma)  ', recogniton rate is ' num2str(Rec) '.\n']);
        end
    end
end