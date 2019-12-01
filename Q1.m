

clc;
clear all;
filename_train = '/Users/parisha/Desktop/ECE6604-MP2/Data sets/N-data.txt';
filename_test = '/Users/parisha/Desktop/ECE6604-MP2/Data sets/N-valdata.txt';
train = importdata(filename_train); 
train_X1 = train(:,1); %118*1
train_X2 = train(:,2);%118*1
X = transpose([ones(size(train,1),1) train_X1 train_X2]);%3*118
Y_target = train(:,3);%118*1

test = importdata(filename_test); 
test_X1 = test(:,1); 
test_X2 = test(:,2);
X_test = transpose([ones(size(test,1),1) test_X1 test_X2]);
Y_target_test = test(:,3);

alpha = 10^-2;

for C=1:20
a1=-sqrt(6/(size(X,1)+C));
a2=-sqrt(6/(C+size(Y_target,2)));
b1=sqrt(6/(size(X,1)+C));
b2=sqrt(6/(C+size(Y_target,2)));
Beta0 = (b1-a1).*rand(size(X,1),C) + a1; % 
Beta=Beta0;
Beta_out0 = (b2-a2).*rand(C+1,size(Y_target,2)) + a2;
Beta_out=Beta_out0;
iteration = 2500;

%for iteration = 500:500:3000



    
for j=1:iteration

  for i=1:size(Y_target,1) %118 times
 
   [loss_value,d_weights_0,d_weights_1] = loss_func(Beta,Beta_out,X(:,i),Y_target(i,:),alpha);
    Beta = Beta - (alpha .*transpose(d_weights_0));
    Beta_out(2:end) = Beta_out(2:end) - (alpha .*d_weights_1);
    
  end
  
end

y_capfinal = zeros(size(Y_target,1),1); %118*1
%after 1 epoch
for i=1:size(Y_target,1) %118
 % Hidden Layer= 11*1, y_cap = 1*1
 [hidden,y_cap]= feedfwd(Beta,Beta_out,X(:,i));
 if(y_cap>=0.5)
     y_capfinal(i) =1;
 else 
     y_capfinal(i) =0;
 end
end
count=0;
for i=1:size(Y_target,1)
   
    if(y_capfinal(i)==Y_target(i))
        count=count+1;
    end
    
end

% Feedforward for testing data:

iteration
C
Train_accuracy = count*100/length(Y_target)

y_capfinal_test = zeros(size(Y_target_test,1),1);

for i=1:size(Y_target_test,1)
     [hidden_test,y_cap_test]= feedfwd(Beta,Beta_out,X_test(:,i));
 if(y_cap_test>=0.5)
     y_capfinal_test(i) =1;
 else 
     y_capfinal_test(i) =0;
 end
end

count_test=0;

for i=1:size(Y_target_test,1)
   
    if(y_capfinal_test(i)==Y_target_test(i))
        count_test=count_test+1;
    end
    
end

Test_accuracy = count_test*100/length(Y_target_test)


%end


end
function [hidden_vector, y_cap] = feedfwd(Beta,Beta_out,X) %X is 3*1 and Beta is 3*C = 3*10
      
    activation = transpose(Beta)*X;
    h = sigmoid(activation);
    h_bar = [1;h];
    activation_y = transpose(Beta_out)*h_bar;
    y_cap = sigmoid(activation_y);
     
    hidden_vector = h_bar;
   
    
 end
  


function [loss_value,d_weights_0,d_weights_1 ] = loss_func(Beta,Beta_out,X,Y,alpha)% Beta=3*10, Beta_out=11*1, X is 3*1, Y = 1
    
  [hidden_layer , y_cap] = feedfwd(Beta,Beta_out,X);
  loss_value = -(Y*log(y_cap)+(1-Y).*log(1-y_cap));
  XX =ones(size(Beta_out(2:end),1),1).* transpose(X);
  d_1 = (-Y+y_cap)/(y_cap*(1-y_cap));
  d_weights_1 = d_1*y_cap*(1-y_cap).* hidden_layer(2:end);
  Beta_out1 = Beta_out;
  Beta_out1(2:end) = Beta_out(2:end) - alpha *d_weights_1;
  d_weights_0 = d_1.*y_cap*(1-y_cap).*Beta_out1(2:end).*hidden_layer(2:end).*(1-hidden_layer(2:end)) .*XX;
 
  
  %d_weights_1 = ()
  %der_EY = -log(y_cap) + log(1-y_cap);d
  %der_outYY = y_cap*(1-y_cap);
  %der_Yw = y_cap*(1-y_cap)*hidden_layer;
  %d_weights = der_EY*der_outYY * der_Yw;
  

end

function y = sigmoid(a)
  y = 1./(1 + exp(-a));
end
