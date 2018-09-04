function [ gradient_weight1, gradient_bias1, gradient_weight2, gradient_bias2 ] = backward_pass( NN_out, label )
%UNTITLED18 Summary of this function goes here
%   Detailed explanation goes here

back1 = back_loss_func(NN_out, label);
gradient_weight2 = backward_weight(back1,100,10);
gradient_bias2 = backward_bias(back1,10);

back2 = backward_act(back1);
gradient_weight1 = backward_weight(back2,784,100);
gradient_bias1 = backward_bias(100);

end

