
function B= DSRIC_DCA(X, Xall, XTXall, Par)
%------------------------------------------------------------------------
% DSRIC classification function
[h, w] = size(X);
B = zeros(h, h);
for iter = 1:Par.maxIter
    C = 2*Par.lambda2*(B*XTXall-XTXall-X*X');
    B = (X*X'+0.5*C)/(X*X'+Par.lambda1*eye(h));
    
       %% convergence conditions
    objErr(iter) = (1+Par.lambda2)*norm( X - B*X, 'fro') + Par.lambda1 * norm(B, 'fro') ...
        - Par.lambda2 * norm(Xall - B*Xall, 'fro') ;
    fprintf('[%d] : objErr: %f \n', iter, objErr(iter));
end
