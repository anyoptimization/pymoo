%...wilcoxon signed rank test ....
clear all
close all
clc
a = 1:10;
b = 2*a;
format shortE
pivot_data = load('NSGA-II_MEMO_Journal\NSGA-II_ZDT6.out' );

comp_data = load('NSGA-II_Python_ZDT6.out' );
alpha_sig = 0.05;
tail_type = 'both';
p = signrank(pivot_data,comp_data,'tail',tail_type,'alpha',alpha_sig);

fprintf('min \t median \t max\n');
fprintf('pivot data Stats\n');
fprintf('%.3e \t %.3e \t %.3e\n',min(pivot_data), median(pivot_data), max(pivot_data));
fprintf('comp data stats\n');
fprintf('%.3e \t %.3e \t %.3e\n',min(comp_data),median(comp_data),max(comp_data));
fprintf('p value = %.3e\n',p);

%disp(p)