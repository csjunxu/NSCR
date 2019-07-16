function z = NRC( y, X, XTXinv, Par )

% Input
% y           Testing data matrix
% X           Training Data matrix, dim * num
% XTXinv   inv( X'*X+rho/2*eye(N) )
% Par         parameters

% Objective function:
%      min_{A}  ||Y - X * A||_{F}^{2}  s.t.  A>=0

% Notation:
% y ... (D x 1) the testing data vector where D is the dimension of input
% data
% X ... (D x N) the training data matrix, where D is the dimension of features, and
%           N is the number of training samples.
% a ... (N x 1) is a column vector used to select
%           the most representive and informative samples to represent the
%           input sample y
% Par ...  struture of regularization parameters
[~, M] = size(y);
[~, N] = size(X);

%% initialization
c       = zeros (N, M); % satisfy NN constraint
z       = c;
Delta = z - c;
Par.display = 1;
for iter = 1:Par.maxIter
    
    Prec=c;
    Prez=z;
    %% update A the coefficient matrix
    c = XTXinv * (X' * y + Par.rho/2 * z + 0.5 * Delta);
    
    %% update C the data term matrix
    Q = (c - Delta/Par.rho)/(2*Par.alpha/Par.rho+1);
    z = max(0, Q);
    
    %% computing errors
    %     err1(iter+1) = errorCoef(c, z);
    %     err2(iter+1) = errorCoef(c, Prec);
    %     err3(iter+1) = errorCoef(z, Prez);
    %     err0(iter+1) = errorLinSys(y, X, z);
    %     if (  (err1(iter+1) <= tol && err2(iter+1) <= tol && err32(iter+1) <= tol && err0(iter+1) <= tol) ||  iter >= Par.maxIter  )
    %         fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
    %         break;
    %     else
    %         if (mod(iter, Par.maxIter)==0)
    %             fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
    %         end
    %     end
    
    %% update Deltas the lagrange multiplier matrix
    Delta = Delta + Par.rho * ( z - c);
    
    %     %% update rho the penalty parameter scalar
    %     Par.rho = min(1e4, Par.mu * Par.rho);
end

return;