function [ output_args ] = forward_pass( input, label, weights1, weights2 )
%UNTITLED17 Summary of this function goes here
%   Detailed explanation goes here

    out1 = forward(input, weights1);
    out1_act = forward_act(out1);
    out2 = forward(out1_act,weights2);
    final_out = forward_softmax(out2);
    final_loss = loss_func(final_out, label);
    disp('final loss', final_loss);
    
    back1 = back_loss_func(NN_out, label);
    gradient_weight2 = backward_weight(back1,100,10);
    gradient_bias2 = backward_bias(back1,10);

    back2 = backward_act(back1);
    gradient_weight1 = backward_weight(back2,784,100);
    gradient_bias1 = backward_bias(100);
end

