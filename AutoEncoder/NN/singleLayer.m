clc;
clear; 
% rbm_lr=0.01;
% rbm_hidden1_num_neuron=100;
% rbm_epoch_num=200;
% rbm_eval_step=1;
% rbm_batch_size=1;
% rbm_CD_step = 5;
% load(['../RBM/HW2_Problem_b_',num2str(rbm_lr),'_',num2str(rbm_hidden1_num_neuron),'_',num2str(rbm_epoch_num),'_',num2str(rbm_CD_step),'_',num2str(rbm_eval_step),'_',num2str(rbm_batch_size),'.mat'],'weights','bias_b')
%load('../autoencoder/HW2_Problem_e_entropy_0.01_100_16_1_0.mat','weights');
%load('../RBM/HW2_Problem_c_0.01_100_100_5_1_1_1.mat','weights');
load('../autoencoder/HW2_Problem_f_dropout_0.01_100_10_1_0.25.mat','weights');
temp_train_set = csvread('../digitstrain.txt');
valid_set = csvread('../digitsvalid.txt');
test_set = csvread('../digitstest.txt');

%%
 train_set = temp_train_set(randperm(size(temp_train_set,1)),:);
train_data = train_set(:,1:784);
train_label_single = train_set(:,785);
train_label = zeros(3000,10);
for iter=1:3000
    train_label(iter,train_label_single(iter)+1) = 1;
end

%train_data = train_data/max(max(train_data));
valid_data = valid_set(:,1:784);
test_data = test_set(:,1:784);

valid_label_single = valid_set(:,785);
valid_label = zeros(1000,10);
for iter=1:1000
    valid_label(iter,valid_label_single(iter)+1) = 1;
end
test_label = test_set(:,785);

input_num_neuron = 784;
hidden1_num_neuron = 100;
output_num_neuron = 10;

lr=0.01;
momentum=0.9;
w_decay = 0.0005;
dropout_prob = 0.5; 
epoch_num = 600;
batch_size = 30;
batch_num = 3000/batch_size;


%for runs=5:10
% for lr=[0.01,0.1]
%     for momentum=[0.5, 0.9]
%         for hidden1_num_neuron=[100,200]
%             for epoch_num=[200, 600]
%                 for w_decay=[0.0005, 0.001]
%                     for dropout_prob=[0.5, 0.9]
                        %disp(['HW2_Problem_g_',num2str(lr),'_',num2str(momentum),'_',num2str(hidden1_num_neuron),'_',num2str(epoch_num),'_',num2str(w_decay),'_',num2str(dropout_prob)]);
                        %initialize weights
                        rand_bound1 = sqrt(6)/sqrt(hidden1_num_neuron+input_num_neuron);
                        rand_bound2 = sqrt(6)/sqrt(hidden1_num_neuron+output_num_neuron);

                        rand_weights1 = -rand_bound1 + 2*rand_bound1*rand(hidden1_num_neuron, input_num_neuron);
                        rand_weights2 = -rand_bound2 + 2*rand_bound2*rand(output_num_neuron, hidden1_num_neuron);
                        rand_bias1 = -rand_bound1 + 2*rand_bound1*rand(hidden1_num_neuron,1);%zeros(hidden1_num_neuron,1);
                        rand_bias2 = -rand_bound2 + 2*rand_bound2*rand(output_num_neuron,1);%zeros(output_num_neuron,1);

                        weights1 = weights;
                        weights2 = rand_weights2;
                        bias1 = rand_bias1;
                        bias2 = rand_bias2;

                        train_loss = zeros(1,epoch_num);
                        valid_loss = zeros(1,epoch_num);
                        train_error = zeros(1,epoch_num);
                        valid_error = zeros(1,epoch_num);
                        old_gradient_weights2 = zeros(output_num_neuron,hidden1_num_neuron);
                        old_gradient_weights1 = zeros(hidden1_num_neuron, input_num_neuron);
                        for epoch=1:epoch_num  
%                             rand_shuffle = randperm(size(temp_train_set,1));
%                             train_data = train_data(rand_shuffle,:);
%                             train_label = train_label(rand_shuffle,:);
                            %disp(['Epoch: ', num2str(epoch)]);
                            for batch=1:batch_num
                                input = train_data((batch-1)*batch_size+1:batch*batch_size,:)';
                                label = train_label((batch-1)*batch_size+1:batch*batch_size,:)';
                                %forward

                                out1 = forward(input, weights1, bias1);               
                                out1_act = forward_act(out1);
                                [out1_act, mask] = dropout(out1_act, dropout_prob);
                                out2 = forward(out1_act,weights2, bias2);
                                final_out = forward_softmax(out2);
                                final_error_num = errorNum(final_out,(train_label_single((batch-1)*batch_size+1:batch*batch_size)+1)');
                                final_loss = loss_func(final_out, label);     
                                train_loss(epoch) = train_loss(epoch) + sum(final_loss);
                                train_error(epoch) = train_error(epoch) + final_error_num;
                                %disp([num2str(batch), ': ', num2str(mean(final_loss))]);

                                %backward
                                back1 = back_loss_func(final_out, label);
                                temp_gradient_weights2 = backward_weight_slides(out1_act);
                                gradient_weights2 = back1 * temp_gradient_weights2;
                                gradient_bias2 = sum(back1,2);%backward_bias(output_num_neuron,batch_size)*reshape(back1,[],1);       
                                back2 = backward(weights2)*back1;
                                back3 = backward_act(out1).*back2;
                                temp_gradient_weights1 = backward_weight_slides(input);
                                gradient_weights1 = back3 * temp_gradient_weights1;
                                gradient_bias1 = sum(back3,2);%backward_bias(hidden1_num_neuron,batch_size)*reshape(back3,[],1);

                                %weight_update

                                new_gradient_weights1 = momentum * old_gradient_weights1 - lr * gradient_weights1/batch_size;
                                new_gradient_weights2 = momentum * old_gradient_weights2 - lr * gradient_weights2/batch_size;

                                weights1 = weights1 - w_decay*lr*weights1 + new_gradient_weights1;
                                weights2 = weights2 - w_decay*lr*weights2  + new_gradient_weights2;
                                bias1 = bias1 - lr*gradient_bias1/batch_size;
                                bias2 = bias2 - lr*gradient_bias2/batch_size;

                                old_gradient_weights1 = new_gradient_weights1;
                                old_gradient_weights2 = new_gradient_weights2;

                                %disp(' ');

                            end
                            %disp([num2str(epoch), ' finished']);
                %             disp('Training set loss');
                %             disp((train_loss/3000));
                %             disp('Training set error');
                %             disp((train_error/3000));

                            %validation
                            out1 = forward(valid_data', weights1, bias1);
                            out1_act = forward_act(out1);
                            out1_act = out1_act * (1-dropout_prob);
                            out2 = forward(out1_act,weights2, bias2);
                            final_out = forward_softmax(out2);
                            final_error_num = errorNum(final_out,(valid_label_single+1)');
                            final_loss = loss_func(final_out, valid_label');
                            valid_loss(epoch) = valid_loss(epoch) + sum(final_loss);
                            valid_error(epoch) = valid_error(epoch) + final_error_num;
                %             disp('validation set loss');
                %             disp((valid_loss/1000));
                %             disp('validation set misclassification rate');
                %             disp((valid_error/1000));

                        %test
                    %     for batch=1:batch_num
                    %         out1 = forward(valid_data', weights1, bias1);
                    %         out1_act = forward_act(out1);
                    %         out2 = forward(out1_act,weights2, bias2);
                    %         final_out = forward_softmax(out2);
                    %         final_loss = loss_func(final_out, valid_label');
                    %         valid_loss(epoch) = valid_loss(epoch) + sum(final_loss);
                    %         disp('validation set loss');
                    %         disp((valid_loss/1000));
                    %     end
                        end
                        final_error_rate = valid_error(epoch_num)/1000;
                        disp(final_error_rate);
                        %save(['Problem_f_lr',num2str(lr),'_','momentum',num2str(momentum),'_','numneuron',num2str(hidden1_num_neuron),'-2kEpoch-dropout.mat']);
                        save(['Problem_f_',num2str(lr),'_',num2str(momentum),'_',num2str(hidden1_num_neuron),'_',num2str(epoch_num),'_',num2str(w_decay),'_',num2str(dropout_prob),'.mat']);
%                     end
%                 end
%             end
%         end
%     end
%     
% end