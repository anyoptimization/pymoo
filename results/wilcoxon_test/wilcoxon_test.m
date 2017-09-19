%..Journal MEMO...Wilcoxon Rank Test...
clear all
close all
clc

alpha_sig = 0.1;
tail_type = 'right'
fprintf('alph = %f\n',alpha_sig);
data_only_DTLZ = load('only_dtlz.txt');
data_only_ZDT = load('only_zdt.txt');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
memo_data = data_only_ZDT(:,1);
nsga_data = data_only_ZDT(:,2);
moead_tch = data_only_ZDT(:,3);
moead_pbi = data_only_ZDT(:,4);

[p_nsga,h_nsga] = signrank(memo_data,nsga_data,'tail',tail_type,'alpha',alpha_sig);
[p_moead_tch,h_moead_tch] = signrank(memo_data,moead_tch,'tail',tail_type,'alpha',alpha_sig);
[p_moead_pbi,h_moead_pbi] = signrank(memo_data,moead_pbi,'tail',tail_type,'alpha',alpha_sig);

fprintf('\nResults for ONLY ZDT');
fprintf('\np_nsga = %f, h_nsga = %d',p_nsga, h_nsga);
fprintf('\np_moead_tch = %f, h_moead_tch = %d',p_moead_tch, h_moead_tch);
fprintf('\np_moead_pbi = %f, h_moead_pbi = %d\n',p_moead_pbi, h_moead_pbi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
memo_data = data_only_DTLZ(:,1);
nsga_data = data_only_DTLZ(:,2);
moead_tch = data_only_DTLZ(:,3);
moead_pbi = data_only_DTLZ(:,4);

[p_nsga,h_nsga] = signrank(memo_data,nsga_data,'tail',tail_type,'alpha',alpha_sig);
[p_moead_tch,h_moead_tch] = signrank(memo_data,moead_tch,'tail',tail_type,'alpha',alpha_sig);
[p_moead_pbi,h_moead_pbi] = signrank(memo_data,moead_pbi,'tail',tail_type,'alpha',alpha_sig);
fprintf('\nResults for ONLY DTLZ');
fprintf('\np_nsga = %f, h_nsga = %d',p_nsga, h_nsga);
fprintf('\np_moead_tch = %f, h_moead_tch = %d',p_moead_tch, h_moead_tch);
fprintf('\np_moead_pbi = %f, h_moead_pbi = %d\n',p_moead_pbi, h_moead_pbi);