% =========================================================================
% （1）只用BMP2的一类配置，BTR70,T72的一类配置，所以总共训练样本只有大概699个左右；
% （2）用原始图像，而不用PCA特征；原始图像截取中心的一块小窗的区域；
% （3）学习得到字典D后，用D去稀疏表示当前的测试样本。这样，看是否能得到较好的方位角扇区分类
% %
%  20190111 7类
%  20190115 7类的类型识别
% =========================================================================


clear all;
clc;
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP
% load('.\trainingdata\featurevectors.mat','training_feats', 'testing_feats', 'H_train', 'H_test');
% load('datasetAddSNR0.mat');
% load('AllDataSet.mat');
%  testing_feats=YTestAll{1,1};training_feats = XTrainAll{1,1};
% testing_feats=testSet0;training_feats =trainSet0;


%%
%训练样本、方位角区间、提取
bmp2sn9563_train_17=load('sample set/BMP2SN9563dep17allazsort.mat');%---1
[mtrbmp9563_17, ntrbmp9563_17] = size(bmp2sn9563_train_17.alldata0);
bmp2_9566train_17=load('sample set/BMP2SN9566dep17allazsort.mat');%---1
[mtrbmp2_9566, ntrbmp2_9566] = size(bmp2_9566train_17.alldata0);
bmp2_C21train_17=load('sample set/BMP2SNC21dep17allazsort.mat');%---1
[mtrbmp2_C21, ntrbmp2_C21] = size(bmp2_C21train_17.alldata0);
btr70snc71_train_17 = load('sample set/BTR70SNC21dep17allazsort.mat');%---2
[mtrbtrc71_17, ntrbtrc71_17] = size(btr70snc71_train_17.alldata0);
t72sn132_train_17 = load('sample set/T72SN132dep17allazsort.mat');%---3
[mtrt132_17, ntrt132_17] = size(t72sn132_train_17.alldata0);
t72_812train_17 = load('sample set/T72SN812dep17allazsort.mat');%---3
[mtrt72_812, ntrt72_812] = size(t72_812train_17.alldata0);
t72_S7train_17 = load('sample set/T72SNS7dep17allazsort.mat');%---3
[mtrt72_S7, ntrt72_S7] = size(t72_S7train_17.alldata0);


%取图像窗口的大小参数，aa和bb 
aa = 49;
bb = aa+31; %取32*32大小的窗口

trainSet=[]; trAspect=[];
% trls = [ones(mtrbmp2_C21, 1); 2.*ones(mtrbtrc71_17, 1); 3.*ones(mtrt132_17, 1)];
% Label = [mtrbmp2_C21, mtrbtrc71_17, mtrt132_17];

trls=[ones(mtrbmp9563_17+mtrbmp2_9566+mtrbmp2_C21,1);2.*ones(mtrbtrc71_17,1);3.*ones(mtrt132_17+mtrt72_812+mtrt72_S7,1)];
Label=[mtrbmp9563_17+mtrbmp2_9566+mtrbmp2_C21 mtrbtrc71_17 mtrt132_17+mtrt72_812+mtrt72_S7]; 

% BMP2
for ii =1:mtrbmp9563_17
    xorig = bmp2sn9563_train_17.alldata0{ii,2};%取出我们需要的图像数据---
    hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    trainSet=[trainSet fea];
    aspect=bmp2sn9563_train_17.alldata0{ii,1};
trAspect=[trAspect;aspect];
end

for ii =1:mtrbmp2_9566
    xorig = bmp2_9566train_17.alldata0{ii,2};%取出我们需要的图像数据---
    hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    trainSet=[trainSet fea];
     aspect=bmp2_9566train_17.alldata0{ii,1};
trAspect=[trAspect;aspect];
end

for ii =1:mtrbmp2_C21
    xorig = bmp2_C21train_17.alldata0{ii,2};%取出我们需要的图像数据---
    hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    trainSet=[trainSet fea];
     aspect=bmp2_C21train_17.alldata0{ii,1};
trAspect=[trAspect;aspect];
end

% BTR70
for ii =1:mtrbtrc71_17
    xorig = btr70snc71_train_17.alldata0{ii,2};%取出我们需要的图像数据---
    hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    trainSet=[trainSet fea];
    aspect=btr70snc71_train_17.alldata0{ii,1};
trAspect=[trAspect;aspect];
end

% T72
for ii =1:mtrt132_17
    xorig = t72sn132_train_17.alldata0{ii,2};%取出我们需要的图像数据---
     
    hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    trainSet=[trainSet fea];
    aspect=t72sn132_train_17.alldata0{ii,1}; 
trAspect=[trAspect;aspect];
end

for ii =1:mtrt72_812
    xorig = t72_812train_17.alldata0{ii,2};%取出我们需要的图像数据---
    hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    trainSet=[trainSet fea];
     aspect=t72_812train_17.alldata0{ii,1};
trAspect=[trAspect;aspect];
end

for ii =1:mtrt72_S7
    xorig = t72_S7train_17.alldata0{ii,2};%取出我们需要的图像数据---
    hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    trainSet=[trainSet fea];
     aspect=t72_S7train_17.alldata0{ii,1};
trAspect=[trAspect;aspect];
end


deltaaz = 10;    %方位角间隔30,60,90,120
aznum = 360./deltaaz;

%--对训练数据分类取标签----------------------------------------------------------
TrainLabel90 = zeros(size(trAspect));
for ii = 1:size(trAspect,1)
    tempaz = trAspect(ii,1);
    for jj = 1:aznum
        sectordown = (jj-1)*deltaaz+0.0001;
        sectorup = jj*deltaaz;
        if tempaz >= sectordown & tempaz< sectorup
            TrainLabel90(ii) = jj; 
        end
    end
end

%%
%测试样本、方位角区间、提取
bmp2sn9563_test_15=load('sample set/BMP2SN9563dep15allazsort.mat');%---1
[mttbmp9563_15, nttbmp9563_15] = size(bmp2sn9563_test_15.alldata0);
bmp2_9566test_15=load('sample set/BMP2SN9566dep15allazsort.mat');%---1
[mttbmp2_9566, nttbmp2_9566] = size(bmp2_9566test_15.alldata0);
bmp2_C21test_15=load('sample set/BMP2SNC21dep15allazsort.mat');%---1
[mttbmp2_C21, nttbmp2_C21] = size(bmp2_C21test_15.alldata0);
btr70snc71_test_15 = load('sample set/BTR70SNC21dep15allazsort.mat');%---2
[mttbtrc71_15, nttbtrc71_15] = size(btr70snc71_test_15.alldata0);
t72sn132_test_15 = load('sample set/T72SN132dep15allazsort.mat');%---3
[mttt132_15, nttt132_15] = size(t72sn132_test_15.alldata0);
t72_812test_15 = load('sample set/T72SN812dep15allazsort.mat');%---3
[mttt72_812, nttt72_812] = size(t72_812test_15.alldata0);
t72_S7test_15 = load('sample set/T72SNS7dep15allazsort.mat');%---3
[mttt72_S7, nttt72_S7] = size(t72_S7test_15.alldata0);

testSet=[]; ttAspect=[]; 
ttls=[ones(mttbmp9563_15+mttbmp2_9566+mttbmp2_C21,1);2.*ones(mttbtrc71_15,1);3.*ones(mttt132_15+mttt72_812+mttt72_S7,1)];
% BMP2
for ii =1:mttbmp9563_15
    xorig = bmp2sn9563_test_15.alldata0{ii,2};%取出我们需要的图像数据---
    hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    testSet=[testSet fea];
    aspect=bmp2sn9563_test_15.alldata0{ii,1};
ttAspect=[ttAspect;aspect];
end

for ii =1:mttbmp2_9566
    xorig = bmp2_9566test_15.alldata0{ii,2};%取出我们需要的图像数据---
    hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    testSet=[testSet fea];
    aspect=bmp2_9566test_15.alldata0{ii,1};
ttAspect=[ttAspect;aspect];
end

for ii =1:mttbmp2_C21
    xorig = bmp2_C21test_15.alldata0{ii,2};%取出我们需要的图像数据---
     hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    testSet=[testSet fea];
    aspect=bmp2_C21test_15.alldata0{ii,1};
ttAspect=[ttAspect;aspect];
end

% BTR70
for ii =1:mttbtrc71_15
    xorig = btr70snc71_test_15.alldata0{ii,2};%取出我们需要的图像数据---
   hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    testSet=[testSet fea];
    aspect=btr70snc71_test_15.alldata0{ii,1};
ttAspect=[ttAspect;aspect];
end

% T72
for ii =1:mttt132_15
    xorig = t72sn132_test_15.alldata0{ii,2};%取出我们需要的图像数据---
    hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    testSet=[testSet fea];
    aspect=t72sn132_test_15.alldata0{ii,1};
ttAspect=[ttAspect;aspect];
end

 for ii =1:mttt72_812
    xorig = t72_812test_15.alldata0{ii,2};%取出我们需要的图像数据---
    hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    testSet=[testSet fea];
    aspect=t72_812test_15.alldata0{ii,1};
ttAspect=[ttAspect;aspect];
 end

for ii =1:mttt72_S7
    xorig = t72_S7test_15.alldata0{ii,2};%取出我们需要的图像数据---
   hall = xorig(aa:bb,aa:bb);
    [m,n]=size(hall);
    fea =reshape(hall,m*n,1);
    testSet=[testSet fea];
    aspect=t72_S7test_15.alldata0{ii,1};
ttAspect=[ttAspect;aspect];
end


TestLabel90 = zeros(size(ttAspect));
for ii = 1:size(ttAspect,1)
    tempaz = ttAspect(ii);
    for jj = 1:aznum
        sectordown = (jj-1)*deltaaz+0.0001;
        sectorup = jj*deltaaz;
        if tempaz >= sectordown & tempaz< sectorup
            TestLabel90(ii) = jj; 
        end
    end
end


%%%  方位角标签 TestLabel90 TrainLabel90 
%%%  分类标签   trls ttls
%%%  样本集 trainSet testSet 1024*698


%% 利用ELM得扇区
kerneloption = [5];
c = 1024;
xapp =[TrainLabel90 trainSet'];
xtest=[TestLabel90 testSet'];
[TTrain,TTest,TrainAC,accur_ELM,TY,label] = elm_kernel(xapp,xtest,1,c,'RBF_kernel',kerneloption);%label为行向量


%% 用lcksvd选择方位角
H_test=[];

% constant    参数选择
sparsitythres = 30; % sparsity prior
% sqrt_alpha = 4; % weights for label constraint term
sqrt_alpha = 8;
% sqrt_beta = 2; % weights for classification err term
sqrt_beta = 0.5;
% dictsize = 570; % dictionary size
dictsize = 9;
iterations = 50; % iteration number
% iterations4ini = 20; % iteration number for initialization
iterations4ini = 20;


H_train0=zeros(size(Label,2),size(trls,1)); %
 a=1; b=0;
for i=1:size(Label,2)
   b=b+Label(1,i);
   H_train0(i,a:1:b)=ones(1,Label(1,i));
   a=a+Label(1,i); 
end

% train1=[ones(1,233) zeros(1,233) zeros(1,232)];
% train2=[zeros(1,233) ones(1,233) zeros(1,232)];
% train3=[zeros(1,233) zeros(1,233) ones(1,232)];
% H_train0=[train1;train2;train3];
% test1=[ones(1,195) zeros(1,196) zeros(1,196)];
% test2=[zeros(1,195) ones(1,196) zeros(1,196)];
% test3=[zeros(1,195) zeros(1,196) ones(1,196)];
% H_test0=[test1;test2;test3];
H_train=[];
predictionLabel=[];
realttls=[];
acc=[];
% dictsize_arr = [50 80 90 100 115 120 130 150 ];

% for aa=1:size(dictsize_arr,2)

% dictsize=dictsize_arr(aa)
for ii =1:max(label)
training_feats=trainSet(:,(TrainLabel90==ii));
H_train=H_train0(:,(TrainLabel90==ii));   % 按照TrainLabel90==ii对应的列抽取H_train0,构成样本标签矩阵
testing_feats=testSet(:,label==ii); 
realttls_temp=ttls(label==ii,:);
realttls=[realttls;realttls_temp];
[Dinit,Tinit,Winit,Q_train] = initialization4LCKSVD(training_feats,H_train,dictsize,iterations4ini,sparsitythres);
[D1,X1,T1,W1] = labelconsistentksvd1(training_feats,Dinit,Q_train,Tinit,H_train,iterations,sparsitythres,sqrt_alpha);
[prediction1] = classificationAspect(D1, W1, testing_feats, H_test, sparsitythres);
predictionLabel =[predictionLabel prediction1];
end
acc_temp=sum(predictionLabel'==realttls)./size(predictionLabel,2)
acc=[acc acc_temp];






