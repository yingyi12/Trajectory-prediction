function [x,p,xn,pn,u]=imm(F,H,Q,R,A,xn,pn,u,y)

mu = zeros(3,3);
psi = zeros(1,3);
% State interaction
for i =1:3
    psi = psi+A(i,:)*u(i); %normalization factor
end
for i=1:3
    mu(i,:) = A(i,:)*u(i)./psi;%conditional model probabilities
end
x0 = cell(3,1);
p0 = cell(3,1);
for j =1:3
    x0{j} = zeros(6,1);
    p0{j} = zeros(6);
    for i = 1:3
        x0{j} = x0{j}+xn{i}*mu(i,j);%mixed state estimate
    end
    for i =1:3
        p0{j} = p0{j}+mu(i,j)*(pn{i}+(xn{i}-x0{j})*(xn{i}-x0{j})');%mixed state covariance
    end
end
% Kalman filter
for j = 1:3
    x_pre{j} = F{j}*x0{j};%state prediction
    p_pre{j} = F{j}*p0{j}*F{j}'+Q{j};%covariance update
%     z{j} = y-H*x_pre{j};%innovation
    K{j} = p_pre{j}*H'/(H*p_pre{j}*H'+R);%Kalman gain
    xn{j} = x_pre{j}+K{j}*(y-H*x_pre{j});%state estimate
    pn{j} = (eye(6)-K{j}*H)*p_pre{j};%state covariance

end

% Model probablity update
D=zeros(1,3);
for j = 1:3
    z{j} = y - H*x_pre{j};%innovation  y-

    s{j} = H*p_pre{j}*H'+R;%innovation covariance
    n{j} = length(s{j})/2;
    D(j) = 1/((2*pi)^n{j}*sqrt(det(s{j})))*exp(-0.5*z{j}'*inv(s{j})*z{j});%likelihood
end
c=sum(D.*psi);%normalization factor
u=D.*psi./c;

% State estimate combination
x = zeros(6,1);
p = zeros(6);
for j = 1:3
    x = x+xn{j}.*u(j);
end
for j = 1:3
    p = p+u(j).*(pn{j}+(xn{j}-x)*(xn{j}-x)');
end

