function [id]= DSRIC(B,y,Dlabels)
%------------------------------------------------------------------------
% DSRIC classification function
for c = 1:max(Dlabels)
    coef_c = B{c}*y;
%         error(c) = norm(y-coef_c,2)^2; % inferior 
    error(c) = norm(y-coef_c,2)^2/sum(coef_c.*coef_c);
end

index      =  find(error==min(error));
id         =  index(1);