%...plot pareto..
clear all
close all
clc

data = load('SRN_RUN1.out');

x = data(:,1);
y = data(:,2);

plot(x,y,'*');
xlabel('obj 1');
ylabel('obj 2');
