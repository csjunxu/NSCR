function c = NPLSR( y , X, Par )

% Input
% y           Testing data vector
% X           Training Data matrix, dim * num
% Par         parameters

% Objective function:
%      min_{a}  ||y - X * a||_{2}^{2} s.t.  a<=0

% Notation: L
% y ... (D x 1) the testing data vector where D is the dimension of input
% data
% X ... (D x N) the training data matrix, where D is the dimension of features, and
%           N is the number of training samples.
% a ... (N x 1) is a column vector used to select
%           the most representive and informative samples to represent the
%           input sample y
% Par ...  struture of regularization parameters

[D, N] = size (X);

%% initialization
% A       = eye (N);    % satisfy ANN consttraint
% A   = rand (N);
a       = zeros (N, 1); % satisfy NN constraint
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
    q = (a - Delta/Par.rho)/(2*Par.lambda/Par.rho+1);
    c = min(0, q);
    
    %% update Deltas the lagrange multiplier matrix
    Delta = Delta + Par.rho * ( c - a);
    
    %     %% update rho the penalty parameter scalar
    %     Par.rho = min(1e4, Par.mu * Par.rho);
    
    %% computing errors
    err1(iter+1) = errorCoef(c, a);
    err2(iter+1) = errorLinSys(y, X, a);
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
