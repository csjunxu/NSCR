function c = NNLS( y, X, XTXinv, Par )

% Input
% y           Testing data vector
% X           Training Data matrix, dim * num
% XTXinv   inv( X'*X+rho/2*eye(N) )
% Par         parameters

% Objective function:
%      min_{a}  ||y - X * a||_{2}^{2} + lambda * ||a||_{2}^{2} s.t.  a>=0

% Notation: L
% y ... (D x 1) the testing data vector where D is the dimension of input
% data
% X ... (D x N) the training data matrix, where D is the dimension of features, and
%           N is the number of training samples.
% a ... (N x 1) is a column vector used to select
%           the most representive and informative samples to represent the
%           input sample y
% Par ...  struture of regularization parameters

[~, N] = size (X);

%% initialization
a       = zeros (N, 1); % satisfy NN constraint
c       = a;
Delta = c - a;

for iter = 1:Par.maxIter
    %% update A the coefficient matrix
    a = XTXinv * (X' * y + Par.rho/2 * c + 0.5 * Delta);
    
    %% update C the data term matrix
    q = (a - Delta/Par.rho)/(2*Par.lambda/Par.rho+1);
    c = max(0, q);
    
    %% update Deltas the lagrange multiplier matrix
    Delta = Delta + Par.rho * ( c - a);
    
    %     %% update rho the penalty parameter scalar
    %     Par.rho = min(1e4, Par.mu * Par.rho);
end
