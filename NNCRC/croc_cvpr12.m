function [min_idx] = croc_cvpr12(testFea, trainFea, trainGnd, lambda, weight)

Class = unique(trainGnd);
[~, testNum] = size(testFea);
[trainNum] = max(size(Class));
sq_lambda = sqrt(lambda);
loss = 1e10 * ones(testNum,trainNum);
c = 1;
n_pos = 0; 
for j = 1 : trainNum    
    A = trainFea(:, trainGnd == j);
    B = trainFea(:, trainGnd ~= j);
    nA = size(A, 2);
    D_all = [A B];
    proj_M = inv(D_all'*D_all + lambda * eye(size(D_all, 2))) * (D_all)';
    invA = inv( A'*A + 1e-4*eye(nA, nA) )*A';
    for i = 1 : testNum
        x = testFea(:,i);
        alpha = proj_M*x;
        alpha_A = alpha(1 : size(A, 2));
        
        loss0 = norm(x - A*invA*x, 2)^2;
        loss1 = norm(x - A * alpha_A, 2)^2;
        loss2 = 1 / (sum(alpha_A.*alpha_A));
        cur_loss =  ((1-weight)*loss0 + weight*loss1) * loss2;
        loss(i,j) = cur_loss;
        
    end
end
[~, min_idx] = min(loss');
