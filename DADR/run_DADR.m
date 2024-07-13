clear;clc;
warning off;
addpath('./data');
addpath('./libsvm-new');
addpath('./liblinear-2.1/matlab');
addpath('./tool');
global options
load('CVE_feature.mat');
S = double(feature);
load('CVE_label.mat');
S_Label=double(label);
load('berlin_feature.mat');
T = double(feature(1:218,:));
Ttest= double(feature(219:end,:));
load('berlin_label.mat');
T_Label=double(label(1:218,:));
Ttest_Label = double(label(219:end,:));
%%
S = normalization(S',1);
S =S';
T = normalization(T',1);
T = T';
Xt = T(:,1:end);
Xs = S(:,1:end);
Ys = S_Label;
Yt = T_Label;
Ttest=normalization(Ttest',1);
Ttest = Ttest';
Ttest = Ttest(:,1:end);
%%
Options = [];
XS = [Xs;Xt;Ttest];
[~,SCORE, latent] = pca(XS);
SelectNum = cumsum(latent)./sum(latent);
index = find(SelectNum >= 0.98);
pca_dim = index(1);
XS=SCORE(:,1:pca_dim);
Xs = XS(1:size(Xs,1),:);
Xt = XS(size(Xs,1)+1:size(Xs,1)+size(Xt,1),:);
Ttest = XS(size(Xs,1)+size(Xt,1)+1:end,:);
clear COEFF index latent pca_dim SelectNum SCORE;
option = [];
option.ReducedDim = 5;
[P1,~] = PCA1(Xs, option);
%%
options.Max_Iter = 50;
options.k = 0.1;
options.mu = 0.01;
options.lambda1 = 0.001;
options.lambda2 = 1000;
options.lambda3 = 0.1;
options.alfa = 0.01;
[W,obj] = DADR(Xs,Ys,Xt,Yt,P1);
Y_tes = W*Ttest';
[~,Cls] = max(Y_tes',[],2);
Acc = mean(Ttest_Label == Cls);

fprintf("------------ Acc:%3.4f ------------\n",Acc*100);