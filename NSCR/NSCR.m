function z = NSCR( y, X, XTXinv, Par )

% Input
% y           Testing data vector
% X           Training Data matrix, dim * num
% Par         parameters

% Objective function:
%      min_{c} ||y - X * c||_{2}^{2} + alpha * ||c||_{2}^{2} + alpha * ||c||_{1} s.t. c>=0

% Notation: L
% y ... (D x 1) the testing data vector where D is the dimension of input
% data
% X ... (D x N) the training data matrix, where D is the dimension of features, and
%           N is the number of training samples.
% c ... (N x 1) is a column vector used to select
%           the most representive and informative samples to represent the
%           input sample y
% Par ...  struture of regularization parameters

[D, N] = size (X);

%% initialization
% A       = eye (N);    % satisfy ANN consttraint
% A   = rand (N);
c     = zeros(N, 1); % satisfy NN constraint
z     = c;
Delta = z - c;

%%
tol   = 1e-4;
% objErr = zeros(Par.maxIter, 1);
err1(1) = inf; err2(1) = inf;
terminate = false;
for iter=1:Par.maxIter
    %% update C the coefficient matrix
    c = XTXinv * (X' * y + Par.rho/2 * z + 0.5 * Delta-Par.beta/2);
    
    %% update C the data term matrix
    z = (c - Delta/Par.rho);%/(2*Par.lambda/Par.rho+1);
    z = max(0, z);
    
    %% update Deltas the lagrange multiplier matrix
    Delta = Delta + Par.rho * ( z - c);
    
    %% update rho the penalty parameter scalar
    Par.rho = min(1e4, Par.mu * Par.rho);
    
    %% computing errors
    %err1(iter+1) = errorCoef(c, a);
    %err2(iter+1) = errorLinSys(y, X, a);
    %if (  (err1(iter+1) <= 1e-4 && err2(iter+1) <= 1e-4) ||  iter >= Par.maxIter  )
    %    terminate = true;
    %    fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
    %else
    %    if (mod(iter, Par.maxIter)==0)
    %        fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end), err2(end), iter);
    %    end
    %end
    
    %     %% convergence conditions
    %     objErr(iter) = norm( y - X*C, 'fro' );
    %     fprintf('[%d] : objErr: %f \n', iter, objErr(iter));
    %     if ( iter>=2 && mod(iter, 10) == 0 || stopCC < 1e-4)
    %         stopCC = max(max(abs(objErr(iter) - objErr(iter-1))));
    %         disp(['iter ' num2str(iter) ',stopADMM=' num2str(stopCC,'%2.6e')]);
    %         if stopCC < tol
    %             break;
    %         end
    %     end
end
