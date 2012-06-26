function [x,t] = genKinN(k,n)
% find string of k 1s in length-n binary string
l=2^n-1;
x = de2bi(1:l)';

%t = zeros(2,length(x));
t = zeros(1,length(x));

for i = 1:length(x)
   t(i) = (length(find(diff(find(x(:,i)==1))==1))) >= k;
%     b = (length(find(diff(find(x(:,i)==1))==1))) >= k;
%     if b
%         t(:,i) = [1 0]';
%     else
%         t(:,i) = [0 1]';
%     end
end