clc;
clear; 
temp_train_set = csvread('../digitstrain.txt');
valid_set = csvread('../digitsvalid.txt');
test_set = csvread('../digitstest.txt');

%%
train_set = temp_train_set(randperm(size(temp_train_set,1)),:);
train_data = train_set(:,1:784);
train_data = train_data>0.5;
train_label_single = train_set(:,785);
train_label = zeros(3000,10);
for iter=1:3000
    train_label(iter,train_label_single(iter)+1) = 1;
end

%train_data = train_data/max(max(train_data));
valid_data = valid_set(:,1:784);
test_data = test_set(:,1:784);
valid_data = valid_data > 0.5;
test_data = test_data>0.5;

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
momentum=0.5;
w_decay = 0.0; %0.0005;
dropout_prob = 0.5; %0.5;
epoch_num = 500;
batch_size = 1;
batch_num = 3000/batch_size;

CD_step = 5;
eval_step = 1;
% for runs=5:10
% for lr=[0.01,0.1]
%     for momentum=[0.5, 0.9]
%        for hidden1_num_neuron=[100,200]
%            for epoch_num=[200, 600]
%                for w_decay=[0.0005, 0.001]
%                    for dropout_prob=[0.5, 0.9]
 
%for CD_step=[1,5,10,20]
for hidden1_num_neuron=[500]
                        %disp(['HW2_Problem_b_',num2str(lr),'_',num2str(hidden1_num_neuron),'_',num2str(epoch_num),'_',num2str(CD_step),'_',num2str(eval_step),'_',num2str(batch_size)]);
                       %initialize weights
                        rand_bound = sqrt(6)/sqrt(hidden1_num_neuron+input_num_neuron);

                        rand_weights = -rand_bound + 2*rand_bound*rand(hidden1_num_neuron, input_num_neuron);
                        rand_bias_c = -rand_bound + 2*rand_bound*rand(input_num_neuron,1);%zeros(hidden1_num_neuron,1);
                        rand_bias_b = -rand_bound + 2*rand_bound*rand(hidden1_num_neuron,1);%zeros(output_num_neuron,1);

                        weights = rand_weights;
                        bias_c = rand_bias_c;
                        bias_b = rand_bias_b;

                        train_loss = zeros(1,epoch_num);
                        valid_loss = zeros(1,epoch_num);
                        train_error = zeros(1,epoch_num);
                        valid_error = zeros(1,epoch_num);
                      
                        for epoch=1:epoch_num  
%                             rand_shuffle = randperm(size(temp_train_set,1));
%                             train_data = train_data(rand_shuffle,:);
%                             train_label = train_label(rand_shuffle,:);
                            disp(['Epoch: ', num2str(epoch)]);
                            for batch=1:batch_num
                                input = train_data((batch-1)*batch_size+1:batch*batch_size,:)';
                                label = train_label((batch-1)*batch_size+1:batch*batch_size,:)';
                                %forward

                                h1_given_x = forward_prob(input, weights, bias_b);
                                temp_input = input;
                                for k=1:CD_step
                                    h_x = forward_prob(temp_input,weights,bias_b);
                                    hidden = rand(hidden1_num_neuron,batch_size)<h_x;
                                    x_h = back_prob(hidden, weights, bias_c);
                                    if k==eval_step
                                        eval_x_h = x_h;
                                    end
                                    temp_input = rand(input_num_neuron,batch_size)<x_h;
                                end
                                h1_given_sampled_x = forward_prob(temp_input, weights, bias_b);

                                gradient_w = -h1_given_x*input' + h1_given_sampled_x*temp_input';
                                gradient_c = sum(-input + temp_input,2);
                                gradient_b = sum(-h1_given_x + h1_given_sampled_x,2);
                                weights = weights - lr*gradient_w/batch_size;
                                bias_c = bias_c - lr*gradient_c/batch_size;
                                bias_b = bias_b - lr*gradient_b/batch_size;
                                positive = input.*eval_x_h;
                                negative = (1-input).*(1-eval_x_h);
                                %train_loss(epoch) = train_loss(epoch) + (-sum(sum(input.*log(eval_x_h) + (1-input).*log(1-eval_x_h))));
                                temp_loss = (-sum(sum(log(positive+(positive==0)) + log(negative+(negative==0)))));
                               train_loss(epoch) = train_loss(epoch) + temp_loss;
                            end
                            %disp([num2str(epoch), ' finished']);
                             %disp('Training set loss');
                            % disp((train_loss/3000));
                %             disp('Training set error');
                %             disp((train_error/3000));

                            %validation
                            temp_input_valid = valid_data';
                            h_x = forward_prob(temp_input_valid,weights,bias_b);
                            hidden = rand(hidden1_num_neuron,size(valid_data',2))<h_x;
                            x_h = back_prob(hidden, weights, bias_c);
                            positive = valid_data'.*x_h;
                            negative = (1-valid_data').*(1-x_h);
                            temp_loss = (-sum(sum(log(positive+(positive==0)) + log(negative+(negative==0)))));
                            valid_loss(epoch) = valid_loss(epoch) + temp_loss;
                            %valid_loss(epoch) = valid_loss(epoch) + (-sum(sum(valid_data'.*log(x_h) + (1-valid_data').*log(1-x_h))));
                            
                            % disp('validation set loss');
                            % disp((valid_loss/1000));
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
                        save(['HW2_Problem_g_RBM',num2str(lr),'_',num2str(hidden1_num_neuron),'_',num2str(epoch_num),'_',num2str(CD_step),'_',num2str(eval_step),'_',num2str(batch_size),'.mat']);
%                      end
%                 end
%             end
%         end
%     end  
end