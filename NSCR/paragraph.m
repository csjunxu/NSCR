load ca.mat;
para = 1:1:200;
CZ = stopCA(1,para);
C = stopA(1,para);
% set(gca, 'YScale', 'log')
plot(para,CZ,'--s','LineWidth',1.5,...
    'MarkerFaceColor','k',...
    'MarkerSize',3);
hold on;
plot(para,C,'-.s','LineWidth',1.5,...
    'MarkerFaceColor','k',...
    'MarkerSize',3);
axis([1 200 0 0.3]);
legend('|c_{k+1}-z_{k+1}|','|c_{k+1}-c_{k}|');
xlabel({'Iteration k'});
ylabel('Maximal Value');
grid on;