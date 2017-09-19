%.... Calculate Statistics MOEAD, 2 Objective problems....
clear all
close all
clc
%...GD, IGD, HyperVolume...

%... load related file for comparison...
%data_pareto = load('pareto_ZDT1_500.txt');
algo_name = 'NSGA-II_Python'
prob_name = 'ZDT6';
max_run = 20;
gd = 0;
igd = 0;
hv = 1;

hv_ref_pt = [1.1,1.1];
if (strcmp(prob_name,'ZDT6'))
    hv_ref_pt = [1.05,0.9];
end
fprintf('Metric  Min \t Mean \t Median \t Max');
if gd == 1
    %fprintf('Compute GD\n');
    gd_array = zeros(max_run,1);
    for run_no = 1:max_run
        file_exp = strcat(prob_name,'_RUN',num2str(run_no),'.out');
        data_exp = load(file_exp);
        n_data_points = size(data_exp,1);
        dist_array = zeros(n_data_points,1);
        for i = 1:n_data_points
            a = [data_pareto(:,1) - data_exp(i,1),data_pareto(:,2) - data_exp(i,2)];
            b = sqrt(sum(a.^2,2));
            dist_array(i) = min(b);
        end
        gd_array(run_no) = mean(dist_array);
    end
    fprintf('\nGD:  %f \t %f \t %f \t %f',min(gd_array), mean(gd_array), median(gd_array), max(gd_array));
end

if igd == 1
    %fprintf('Compute IGD\n');
    igd_array = zeros(max_run,1);
    for run_no = 1:max_run
        file_exp = strcat(prob_name,'_RUN',num2str(run_no),'.out');
        data_exp = load(file_exp);
        n_data_points = size(data_pareto,1);
        dist_array = zeros(n_data_points,1);
        for i = 1:n_data_points
            a = [data_exp(:,1) - data_pareto(i,1),data_exp(:,2) - data_pareto(i,2)];
            b = sqrt(sum(a.^2,2));
            dist_array(i) = min(b);
        end
        igd_array(run_no) = mean(dist_array);
    end
    fprintf('\nIGD: %f \t %f \t %f \t %f',min(igd_array),mean(igd_array), median(igd_array), max(igd_array));
end

if hv == 1
    %fprintf('Compute HV\n');
    ref_pt = hv_ref_pt;
    
    hv_array = zeros(max_run,1);
    hv_array2 = hv_array;
    for run_no = 1:max_run
        file_exp = strcat(prob_name,'_RUN',num2str(run_no),'.out');
        data_exp = load(file_exp);
        sorted_pareto = sortrows(data_exp,-2);
        n_data_points = size(data_exp,1);
        %..filtering out the outliers..
        b = zeros(n_data_points,2);
        b(:,1) = ref_pt(1) - sorted_pareto(:,1);
        b(:,2) = ref_pt(2) - sorted_pareto(:,2);

        %hv_set = [sorted_pareto,b];
        n_hv = 0;
        hv_set = zeros(n_data_points,2);
        for i = 1:n_data_points
            if ((b(i,1) > 0) && (b(i,2) > 0))
                n_hv = n_hv + 1;
                hv_set(n_hv,:) = sorted_pareto(i,:);
            end
        end

        if n_hv > 0
            hv_set_y1 = [ref_pt(2);hv_set(1:n_hv - 1,2)];
            x_diff = ref_pt(1) - hv_set(1:n_hv,1);
            y_diff = hv_set_y1 - hv_set(1:n_hv,2);
            area_array = x_diff.*y_diff;
            hv_value = sum(area_array);
            hv_array(run_no) = hv_value;
        end
        
        area_array2 = zeros(n_hv,1);
        for i = 1:n_hv
            if i == 1
                area_array2(i) = (ref_pt(1) - hv_set(i,1))*(ref_pt(2) - hv_set(i,2));
            else
                area_array2(i) = (ref_pt(1) - hv_set(i,1))*(hv_set(i-1,2) - hv_set(i,2));
            end 
        end
        hv_array2(run_no) = sum(area_array2);
    end
    
    fprintf('\nHV:  %f \t %f \t %f \t %f', min(hv_array), mean(hv_array), median(hv_array), max(hv_array));
    fprintf('\n\nHV2:  %f \t %f \t %f \t %f', min(hv_array2), mean(hv_array2), median(hv_array2), max(hv_array2));   
end

fprintf('\n');

file_name = strcat('wilcoxon_test/',algo_name,'_',prob_name,'.out');
save(file_name,'hv_array','-ascii')