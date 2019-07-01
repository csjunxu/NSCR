function Df = dict_learner2(X, D, eta)

nAtom = size(D,2);
nData = size(X,2);

Dict         =    D;
Coef         =    zeros(nAtom,nData);
%meanMatrix   =    zeros(nAtom,nData);

wayreg = 'l2';
etam = 0;
maxiter = 30;
for k = 1 : maxiter
    %% update coef
    Dsym = Dict'*Dict;
    switch lower(wayreg)
        case {'l2'}
            Coef = invChol_mex(Dsym+eta*diag(ones(1,nAtom)))*Dict'*X;
        case {'lm'}
            Coef = invChol_mex(Dsym+(eta+etam)*diag(ones(1,nAtom)))*(Dict'*X+etam*repmat(sum(Coef,2)/nData,[1,nData]));
        case {'l1'}
            parfor i = 1 : nData
                %Coef(:,i) = feature_sign(D_init,X(:,i),lambda);
                Coef(:,i) = L1QP_FeatureSign_yang(0.5*eta, Dsym, -Dict'*X(:,i));
            end
    end
    %% update dictionary
%     newD        =   [];
%     newCoef     =   [];
%     for i =  1 : size(Dict,2)
%         ai      =    Coef(i,:);
%         Y       =    X-Dict*Coef+Dict(:,i)*ai;
%         di      =    Y*ai';
%         if norm(di,2) < 1e-6
%             di        =    zeros(size(di));
%             %newD      =    [newD di];
%         else
%             di        =    di./norm(di,2);
%             newD      =    [newD di];
%             newCoef   =    [newCoef;ai];
%         end
%         Dict(:,i)  =    di;
%     end
%     Dict       =    newD;
%     Coef       =    newCoef;
Dict = l2ls_learn_basis_dual(X, Coef, 1);
% Dict = X*Coef'*inv(Coef*Coef'+eps*diag(ones(1,size(Coef,1))));
% Dict = normcol_equal(Dict);
%% update mean coef
%meanMatrix = Coef*ones(nData,nData)/nData;

end
Df = Dict;
%meanvec = sum(Coef,2)/nData;