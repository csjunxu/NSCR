function C = NPLSR( Y , X, XTXinv, Par )

% Input
% Y           Testing data vector
% X           Training Data matrix, dim * num
% Par         parameters

% Objective function:
%      min_{A}  ||Y - X * A||_{F}^{2}  s.t.  A<=0

% Notation: L
% Y ... (D x M) the testing data vector where D is the dimension of input
% data
% X ... (D x N) the training data matrix, where D is the dimension of features, and
%           N is the number of training samples.
% A ... (N x M) is a column vector used to select
%           the most representive and informative samples to represent the
%           input sample y
% Par ...  struture of regularization parameters
[~, M] = size(Y);
[~, N] = size(X);

%% initialization
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
    Q = (A - Delta/Par.rho)/(2*Par.lambda/Par.rho+1);
    C = min(0, Q);
    
    %% update Deltas the lagrange multiplier matrix
    Delta = Delta + Par.rho * ( C - A);
    
    %     %% update rho the penalty parameter scalar
    %     Par.rho = min(1e4, Par.mu * Par.rho);
    
    %% computing errors
    err1(iter+1) = errorCoef(C, A);
    err2(iter+1) = errorLinSys(Y, X, A);
    if (  (err1(iter+1) <= tol && err2(iter+1) <= tol) ||  iter >= Par.maxIter  )
        terminate = true;
%         fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
%     else
%         if (mod(iter, Par.maxIter)==0)
%             fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
%         end
    end
    
    %% next iteration number
    iter = iter + 1;
end
end
