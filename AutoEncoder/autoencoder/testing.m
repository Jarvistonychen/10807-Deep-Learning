clc;
clear;
%%single layer
load('Problem_g_0.1_0.9_200_600_0.001_0.5.mat');
test_set = csvread('digitstest.txt');
%test
test_data = test_set(:,1:784);
test_label_single = test_set(:,785);
test_label = zeros(3000,10);
for iter=1:1000
    test_label(iter,test_label_single(iter)+1) = 1;
end

out1 = forward(test_data', weights1, bias1);
out1_act = forward_act(out1);
out1_act = out1_act * (1-dropout_prob);
out2 = forward(out1_act,weights2, bias2);
final_out = forward_softmax(out2);
final_error_num = errorNum(final_out,(test_label_single+1)');
final_loss = loss_func(final_out, test_label');
disp(final_error_num/3000)
disp(sum(final_loss)/3000)

%%two layers
load('Problem_h_0.1_0.9_200_200_600_0.001_0.5-twohidden.mat')
test_set = csvread('digitstest.txt');
%test
test_data = test_set(:,1:784);
test_label_single = test_set(:,785);
test_label = zeros(3000,10);
for iter=1:1000
    test_label(iter,test_label_single(iter)+1) = 1;
end
 out1 = forward(test_data', weights1, bias1);
out1_act = forward_act(out1);
out1_act = out1_act * (1-dropout_prob);
out12 = forward(out1_act,weights12, bias12);
out12_act = forward_act(out12);
out12_act = out12_act * (1-dropout_prob);
out2 = forward(out12_act,weights2, bias2);
final_out = forward_softmax(out2);                           
final_error_num = errorNum(final_out,(test_label_single+1)');
final_loss = loss_func(final_out, test_label');
disp(final_error_num/3000)
disp(sum(final_loss)/3000)