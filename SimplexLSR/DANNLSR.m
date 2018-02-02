function C = DANNLSR( Y , X, XTXinv, Par )

% Input
% Y          input test data point
% X          Data matrix, dim * num
% Par        parameters

% Objective function:
%      min_{A}  ||Y - X * A||_{F}^{2} + lambda * ||A||_{2}^{2}
%      s.t.  1'*A = s*1', A>=0

% Notation: L
% Y ... (D x M) input data vector, where D is the dimension of the features
%
% X ... (D x N) data matrix, where L is the dimension of features, and
%           N is the number of samples.
% A ... (N x M) is a column vector used to select
%           the most representive and informative samples
% s ... a non-negative scale, default s=1
% Par ...  structure of the regularization parameters

if ~isfield(Par, 's')
    Par.s = 1;
end

[~, M] = size(Y);
[~, N] = size(X);

%% initialization
% A   = eye (N);
% A   = rand (N);
A       = zeros (N, M); % satisfy NN constraint
C       = A;
Delta = C - A;

%%
tol   = 1e-4;
iter    = 1;
% objErr = zeros(Par.maxIter, 1);
err1(1) = inf; err2(1) = inf;
terminate = false;
while  ( ~terminate )
    %% update A the coefficient matrix
    A = XTXinv * (X' * Y + Par.rho/2 * C + 0.5 * Delta);
    
    %% update C the data term matrix
    Q = (Par.rho*A - Delta)/( (2*Par.lambda+Par.rho)*Par.s );
    C  = Par.s*solver_BCLS_closedForm(Q);
%     Q1 = (Par.rho*A - Delta)/(2*Par.lambda+Par.rho);
%     C1  = solver_BCLS_closedForm(Q1);
    
    %% update Deltas the lagrange multiplier matrix
    Delta = Delta + Par.rho * ( C - A);
    
    %     %% update rho the penalty parameter scalar
    %     Par.rho = min(1e4, Par.mu * Par.rho);
    
    %% computing errors
    err1(iter+1) = errorCoef(C, A);
    err2(iter+1) = errorLinSys(Y, X, A);
    if (  (err1(iter+1) <=tol && err2(iter+1)<=tol) ||  iter >= Par.maxIter  )
        terminate = true;
        fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
    else
        if (mod(iter, Par.maxIter)==0)
            fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
        end
    end
    
    %         %% convergence conditions
    %     objErr(iter) = norm( X - X*C, 'fro' ) + Par.lambda * norm(C, Par.p);
    %     fprintf('[%d] : objErr: %f \n', iter, objErr(iter));
    %     if ( iter>=2 && mod(iter, 10) == 0 || stopCC < tol)
    %         stopCC = max(max(abs(objErr(iter) - objErr(iter-1))));
    %         disp(['iter ' num2str(iter) ',stopADMM=' num2str(stopCC,'%2.6e')]);
    %         if stopCC < tol
    %             break;
    %         end
    %     end
    %% next iteration number
    iter = iter + 1;
end
A = Par.s*A;
C = Par.s*C;
end
