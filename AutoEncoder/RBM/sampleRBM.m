clc;
clear; 

hidden1_num_neuron=500;
input_num_neuron = 784;
input_num = 100;

load('HW2_Problem_g_RBM0.01_500_500_5_1_1.mat')

sample_CD_step=100;
randInput = rand(input_num_neuron,input_num) > 0.5;
temp_input = randInput;
for k=1:sample_CD_step
    h_x = forward_prob(temp_input,weights,bias_b);
    hidden = rand(hidden1_num_neuron,input_num)<h_x;
    x_h = back_prob(hidden, weights, bias_c);    
    temp_input = rand(input_num_neuron,input_num)<x_h;
end

images = permute(reshape(temp_input,[28,28,input_num]),[2,1,3]);

figure()
for i=1:100
    h = subplot(10,10,i);
    imshow(images(:,:,i));
    p = get(h,'pos');
    p(3) = p(3) + 0.01;
    p(4) = p(4) + 0.01;
    set(h,'pos',p);
    axis('off');
end