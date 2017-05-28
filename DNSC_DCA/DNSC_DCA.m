function A= DNSC_DCA(X, Xn, Par)
%------------------------------------------------------------------------
% DSRIC classification function
[h, w] = size(X);
A = zeros(w, w);
for ite = 1:Par.maxIter
    C = 2*Par.gamma*Xn'*(Xn*A-Xn);
    A = (X'*X+lambda*eye(w))\(X'*X+0.5*C);
end
