function [pred_ttls, pred_matrix] = PredictID(coefs, tr_dat, trls, class_num)

%% perform prediction class-by-class
tt_num     =  size(coefs, 2);
pred_matrix      =   zeros(class_num, tt_num);
recon_tr_descr  =   tr_dat * coefs;

for ci = 1:class_num
    loss_ci = recon_tr_descr - tr_dat(:, trls == ci) * coefs(trls == ci,:);
    pci = sum(loss_ci.^2, 1);
    pred_matrix(ci,:) = pci;
end

[~,pred_ttls] = min(pred_matrix,[],1);

end