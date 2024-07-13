function [W,obj] = DADR(Xs,Y_s,Xt,Yt,W)
%%
% ||VU-(Ys+B.*M)||F^2+lambda1*||WXs-U||F^2+lambda2*||W||2,1+lambda3*Tr(WX(S-D)X'W')
% s.t. VV'=I , M>=0
%%
global options;
C = length(unique(Y_s));
[ns,d] = size(Xs);
Xs = Xs';
Xt = Xt';

X = [Xs,Xt];
label = unique(Y_s);
Ys  = bsxfun(@eq, Y_s, label');
Ys  = double(Ys');
Y = [Y_s',Yt'];
options.num = length(Y);
W = W';
B = Construct_B(Ys);
[x33,y33]=size(Ys);
M=ones(x33,y33);
Ys = Ys+B.*M;
U = Ys;
V = ones(C);

zj = sqrt(sum(W.*W,1)+eps);
z = 0.5./zj;
Z = diag(z);

model= svmtrain(Y_s,Xs','-s 0 -t 0 -c 1 -g 1 ');
[predict_label, ~, ~] = svmpredict(Yt,Xt', model);
[A,S] = constructS((W*Xt)',Y_s,options);
LS = A-S;
[A1,D] = constructD([Y_s',predict_label'],options);
LD = A1-D;

%% µü´ú
for iter = 1:options.Max_Iter
    % -------- W -------- %
    Wa = options.lambda1*Xs*Xs'+options.lambda2*Z+options.lambda3*(X*(LS-options.alfa*LD)*X');
    Wb = options.lambda1*U*Xs';
    W = Wb*inv(Wa);
    
    % -------- U -------- %
    Ua = V'*V+options.lambda1*eye(size(V,1));
    Ub = options.lambda1*W*Xs+V'*Ys;
    U = inv(Ua)*Ub;
    
    % -------- V -------- %
    Va = U*U'+eye(size(U,1));
    Vb = Ys*U';
    V = Vb*inv(Va);
    
    % -------- Z -------- %
    zj = sqrt(sum(W.*W,1)+eps);
    z = 0.5./zj;
    Z = diag(z);
    
    % -------- M -------- %
    E=V*U-Ys;
    M_original=B.*E;
    [x11,y11]=size(E);
    %%
    for i=1:x11
        for j=1:y11
            M(i,j)=max(M_original(i,j),0);
        end
    end
    Ys = Ys+B.*M;
    
    %% sam+obj
    f1 = norm((Ys+B.*M)-V*U, 'fro') ^ 2;
    f2 = norm(W*Xs-U, 'fro') ^ 2;
    f3 = trace(W*Z*W');
    f4 = trace(W*X*(LS-options.alfa*LD)*X'*W');
    obj(iter) =f1 + options.lambda1*f2 + options.lambda2*f3 + options.lambda3*f4;
    if iter>5 && abs(obj(iter) - obj(iter-1)) < 0.01
        break;
    end
    [~,Yt_Pre] = max((W*Xt)',[],2);
    [A,S] = constructS((W*Xt)',Y_s,options);
    LS = A-S;
    [A1,D] = constructD([Y_s',Yt_Pre'],options);
    LD = A1-D;
end