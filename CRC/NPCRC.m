function [id]= NPCRC(y, D, Par, Dlabels)
%------------------------------------------------------------------------
% NPCRC_RLS classification function
coef = NPLSR( y, D, Par );
for ci = 1:max(Dlabels)
    coef_c   =  coef(Dlabels==ci);
    Dc       =  D(:,Dlabels==ci);
    error(ci) = norm(y-Dc*coef_c,2)^2/sum(coef_c.*coef_c);
end

index      =  find(error==min(error));
id         =  index(1);