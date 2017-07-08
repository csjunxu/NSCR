function c = ANNLSR( y, X , Par )

% Input
% y          input test data point
% X          Data matrix, dim * num
% Par        parameters

% Objective function:
%      min_{A}  ||y - X * a||_{2}^{2} + lambda * ||a||_{2}^{2}
%      s.t.  1'*a = 1', a>=0

% Notation: L
% y ... (D x 1) input data vector, where D is the dimension of the features
%
% X ... (D x N) data matrix, where L is the dimension of features, and
%           N is the number of samples.
% a ... (N x 1) is a column vector used to select
%           the most representive and informative samples
% Par ...  structure of the regularization parameters

[D , N] = size (X);

%% initialization

% A       = eye (N);
% A   = rand (N);
a       = zeros (N, 1);
c       = a;
Delta = c - a;

%%
tol   = 1e-4;
iter    = 1;
% objErr = zeros(Par.maxIter, 1);
err1(1) = inf; err2(1) = inf;
terminate = false;
if N < D
    XTXinv = (X' * X + Par.rho/2 * eye(N))\eye(N);
else
    XTXinv = (2/Par.rho * eye(N) - (2/Par.rho)^2 * X' / (2/Par.rho * (X * X') + eye(D)) * X );
end
while  ( ~terminate )
    %% update A the coefficient matrix
    a = XTXinv * (X' * y + Par.rho/2 * c + 0.5 * Delta);
    
    %% update C the data term matrix
    q = (Par.rho*a - Delta)/(2*Par.lambda+Par.rho);
    c  = solver_BCLS_closedForm(q);
    
    %% update Deltas the lagrange multiplier matrix
    Delta = Delta + Par.rho * ( c - a);
    
    %     %% update rho the penalty parameter scalar
    %     Par.rho = min(1e4, Par.mu * Par.rho);
    
    %% computing errors
    err1(iter+1) = errorCoef(c, a);
    err2(iter+1) = errorLinSys(y, X, a);
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
end
