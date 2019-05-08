%--------------------------------------------------------------------------
% This function computes the maximum L2-norm error among the columns of the 
% residual of a linear system 
% y: Dx1 data vector of 1 data point in a D-dimensional space
% X: DxN data matrix of N data point in a D-dimensional space
% z: Nx1 sparse coefficient vector
% err: maximum L2-norm of the columns of y-Xz
%--------------------------------------------------------------------------
% Copyright @ Jun Xu, 2017
%--------------------------------------------------------------------------

function err = errorLinSys(y, X, z)

[R,N] = size(z);
if (R > N) 
    e = X(:,N+1:end) * z(N+1:end,:);
    y = y(:,1:N);
    Y0 = y - e;
    C = z(1:N,:);
else
    y = X;
    Y0 = X;
    C = z;
end

[Yn,n] = matrixNormalize(Y0);
M = repmat(n,size(y,1),1);
S = Yn - y * C ./ M;
err = sqrt( max( sum( S.^2,1 ) ) );