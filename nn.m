function [] = nn(ps,noteIt,plotErr,plotEnd,ergodicErr,noFunc,waitBarSwitch,fmFilterSwitch,useToolbox)
clear all; close all;

%-------------------------------------
% requires MatlabBGL
% http://dgleich.github.com/matlab-bgl/
% Run
% >>help Contents 
% to view methods defined in MatlabBGL
%
% requires Waterloo file and matrix utilities
% Direct download: http://goo.gl/NjSKE
% http://sigtool.sourceforge.net/
% http://www.mathworks.com/matlabcentral/fileexchange/12250
% uses: nmatrix
%-------------------------------------

%% Define switches
ps = 'mixing'; % test | mixing | grid | rand

noteIt     = 1000;
plotErr    = 1;
plotEnd    = 1;
ergodicErr = 1;
partialIO  = 0;
noFunc     = 1;
popTopEvo  = 1;

waitBarSwitch  = 0;
fmFilterSwitch = 1; 
useToolbox     = 0;


%% Define parameters
N = 100;            % population size, must be perfect square
T = 2000;         % number of training steps
trS = 50;           % size of training set
Ninp = 8;           % number of inputs
Nhid = 10;          % number of hidden layer nodes
Nout = 1;           % number of output nodes
Nch = 3;            % number of consecutive 1's in the K-in-N game
WR = [0 1];    % range of nn weights; the neural.m file uses [0 1] and paper uses [-.5 .5]
delta = 0.05;       % negative increment for form-meaning mapping
epsil = 0.01;       % positive increment for form-meaning mapping
LR = 2.6;           % learning rate

datFname = ['nnErr' datestr(now,'yyyymmddHHMMSS') '.mat'];
writeIter = 1000;

%% Initialize variables

% error
if ergodicErr
    E1t = zeros(2^Ninp-1,1);
else
    E1t = zeros(N,2^Ninp-1);
end
E2t = zeros(2^Ninp-1,1);


if partialIO
    %save(datFname,'E','-v6');clear E;
    %E = nmatrix(datFname,'/E');
    E = zeros(writeIter,2);
    datH=MATOpen(datFname,'a') ;
else
    E = zeros(T,2);
end

% fitness
fitVal = zeros(N,1);


%% Generate training data

[x,t]=genKinN(Nch,Ninp);
%[y,u]=genKinN(Nch,Nhid);
rind = randperm(2^Ninp-1);

%% Generate population of networks
if useToolbox
    net = cell(N,1);
    for i=1:N
        net{i} = feedforwardnet(Nhid,'trainscg');
        net{i}.trainParam.showWindow = false;
        net{i} = configure(net{i},x,t);
        net{i}.layers{2}.transferFcn = 'logistic';
    end
else
    netH = WR(1) + diff(WR).*rand(Nhid,Ninp,N);
    netO = WR(1) + diff(WR).*rand(1,Nhid,N);
end

%% Initialize form-meaning matrices
fm = zeros(Nhid,Nhid,N);

%% Train networks
%hf1=figure('Visible','on');
if plotErr
    hf1=figure();
    hold on;
    set(hf1,'Color','w');
    axis([0 T 0 1]);
    xlabel('time'); ylabel('error');
end

if useToolbox
    ol = newp(y,u,'logistic','learnp');
    hl = newp(x,y,'logistic','learnp');
end

if waitBarSwitch; hw = waitbar(0,''); end
tstart = tic;

% initialize population structure graph
% s1 sender r1 receiver
if strcmp(ps,'test')
    s1=1; r1=2;
elseif strcmp(ps,'mixing')
%    rp=randperm(N); s1=rp(1); r1=rp(2);
elseif strcmp(ps,'grid')
    [G xy]=grid_graph(sqrt(N),sqrt(N));
%    [Gi Gj Gw]=find(G);
elseif strcmp(ps,'rand')
    p = 1/3;
    G=erdos_reyni(N,p);
%    [Gi Gj Gw]=find(G);
end

%% Main loop (time)
for i=1:T
    if mod(i,noteIt)==1; titer = tic; end

% Select indices for training given a population structure

if strcmp(ps,'test')
    % s1 sender r1 receiver
    s1=1; r1=2;
elseif strcmp(ps,'mixing')
    rp=randperm(N); s1=rp(1); r1=rp(2);
elseif strcmp(ps,'grid')
%    [G xy]=grid_graph(sqrt(N),sqrt(N));
    [Gi Gj Gw]=find(G);
    rp=randperm(num_edges(G));
    s1=Gi(rp(1)); r1=Gj(rp(1));
    rp(3) = randsample(N,1);  % index to random individual for sender error computation
    rp(4) = randsample(Gi(Gj==r1),1); % index to random individual with arrow directed to r1 to serve as a sender in error computation
elseif strcmp(ps,'rand')
%     p = 1/3;
%     [G xy]=erdos_reyni(N,p);
    [Gi Gj Gw]=find(G);
    rp=randperm(num_edges(G));
    s1=Gi(rp(1)); r1=Gj(rp(1));
    rp(3) = randsample(N,1);  % random individual for sender error computation
    rp(4) = randsample(Gi(Gj==r1),1); % random individual with arrow directed to r1 to serve as a sender in error computation
end
    
% Select a training example
    si = randi([1 trS]);
    xi = rind(si);
      
% Decompose the receiver into parts representing hl and ol
if useToolbox
    ol.IW{1} = net{r1}.LW{2,1};
    ol.b{1} = net{r1}.b{2};
    
    hl.IW{1} = net{r1}.IW{1};
    hl.b{1} = net{r1}.b{1};    
else
    Shl = netH(:,:,s1);
    Sol = netO(:,:,s1);
    Rhl = netH(:,:,r1);
    Rol = netO(:,:,r1);
end

% Compute sender's hidden layer activation levels
if useToolbox
    hact = +(logistic(net{s1}.IW{1}*x(:,xi)+net{s1}.b{1},0.1)>0.5); % sender hidden layer activation levels
else
    if noFunc
        hact = +((1./(1+exp(-Shl*x(:,xi))))>0.5);
    else
        hact = +(logistic(Shl*x(:,xi))>0.5);
    end
end

% Update sender's form-meaning mapping and encode message with it
    if fmFilterSwitch
        fmInd = find(hact==1);
        fmMat = fm(:,:,s1);
        
        for j=1:length(fmInd)
            maxInd = find( fmMat(:,fmInd(j)) == max(fmMat(:,fmInd(j))), 1 );
            fmMat(maxInd,fmInd(j)) = fmMat(maxInd,fmInd(j))+delta+2*epsil;
            fmMat(:,fmInd(j)) = fmMat(:,fmInd(j)) - epsil;
            fmMat(maxInd,:) = fmMat(maxInd,:) - epsil;
        end
        
        fm(:,:,s1) = fmMat;
        Stxfr = fmMat>0;
        menc = Stxfr*hact;
                
% Decode sender's message with receiver's form-meaning mapping
        RfmMat = fm(:,:,r1);
        Rtxfr = RfmMat>0;
        mdec = Rtxfr'*menc;
        mdec(mdec==1)=0.9; 
        mdec(mdec==0)=0.1;
        hact = mdec;
    end
    
% Use sender's hidden layer activation levels on receiver's hidden layer
% and perform back propagation on this basis to compute expected hidden
% layer activation levels
if useToolbox
    d = t(xi) - sim(ol,hact); % error
    ht = hact + d*(ol.IW{1}'+ol.b{1}); % training set for hidden layer
else
    if noFunc
        d = (t(xi) - 1./(1+exp(-Rol*hact)));
    else
        d = (t(xi) - logistic(Rol*hact));
    end
    ht = hact + LR*d*Rol';
    z = ht>0;   
end

% Update receiver's form-meaning mapping
if fmFilterSwitch
    fmInd = find(z==1);
        fmMat = fm(:,:,r1);
        
        for j=1:length(fmInd)
            maxInd = find( fmMat(:,fmInd(j)) == max(fmMat(:,fmInd(j))), 1 );
            fmMat(maxInd,fmInd(j)) = fmMat(maxInd,fmInd(j))+delta+2*epsil;
            fmMat(:,fmInd(j)) = fmMat(:,fmInd(j)) - epsil;
            fmMat(maxInd,:) = fmMat(maxInd,:) - epsil;
        end
    fm(:,:,r1) = fmMat;
end

% Update/train the decomposed receiver and sender
if useToolbox    
    % Train receiver
    [hl,Yhl,Ehl,Pfhl] = adapt(hl,x(:,xi),ht);
    [ol,Yol,Eol,Pfol] = adapt(ol,hact,t(xi));
    net{r1}.LW{2,1} = ol.IW{1};
    net{r1}.b{2} = ol.b{1};
    net{r1}.IW{1} = hl.IW{1};
    net{r1}.b{1} = hl.b{1};

    % Train the sender
    net{s1} = adapt(net{s1},x(:,xi),t(xi));
else        
    % Train the sender
    if noFunc
        H = 1./(1+exp(-Shl*x(:,xi)));
        O = 1./(1+exp(-Sol*H));
    else
        H = logistic(Shl*x(:,xi));
        O = logistic(Sol*H);
    end
    dSol = O.*(1-O).*(t(xi) - O); % should O.*(1-O) be used when weights aren't 0-1?
    dShl = H.*(1-H).*Sol'*dSol; % should H.*(1-H) be used when weights aren't 0-1?
    Sol = Sol + LR*dSol*H';
    Shl = Shl + LR*dShl*x(:,xi)';
    
    % Train the receiver
    H = 1./(1+exp(-Rhl*x(:,xi)));
    O = 1./(1+exp(-Rol*hact));    
    if noFunc
        dRol = O.*(1-O).*(t(xi) - O); % computed above as d    
        dRhl = H.*(1-H).*(Rol'*dRol); % not sure if this is intended (z - H);
    else
        dRol = (t(xi) - logistic(Rol*hact)); % computed above as d    
        dRhl = (z - logistic(Rhl*x(:,xi)));    
    end
    Rol = Rol + LR*dRol*hact'; 
    Rhl = Rhl + LR*dRhl*x(:,xi)';
    
    % Put updated weights back into population
    netH(:,:,s1) = Shl;
    netO(:,:,s1) = Sol;
    netH(:,:,r1) = Rhl;
    netO(:,:,r1) = Rol;
end

% Compute data for plotting
    if plotErr
            if useToolbox
                for j=1:N
                    E1t(j) = mean((t-net{j}(x)).^2);
                end
                E(:,1) = mean(E1t);
                E(:,2) = mean((t-net{r1}(x)).^2);
            else
                if ergodicErr
                    for j=rp(3)
                        for k=1:(2^Ninp-1)
                            if noFunc
                                E1t(k,1) = (t(k) - ((1./(1+exp(-netO(:,:,j)*(1./(1+exp(-netH(:,:,j)*x(:,k)))))))>0.5)).^2;
                            else
                                E1t(k,1) = (t(k) - (logistic(netO(:,:,j)*logistic(netH(:,:,j)*x(:,k)))>0.5)).^2;
                            end
                        end
                    end
                    E(i,1) = mean([mean(E1t); E(max([i-100 1]):max([i-1 1]),1)]);
                else
                    for j=1:N
                        for k=1:(2^Ninp-1)
                            if noFunc
                                E1t(j,k) = (t(k) - ((1./(1+exp(-netO(:,:,j)*(1./(1+exp(-netH(:,:,j)*x(:,k)))))))>0.5)).^2;
                            else
                                E1t(j,k) = (t(k) - (logistic(netO(:,:,j)*logistic(netH(:,:,j)*x(:,k)))>0.5)).^2;
                            end
                        end
                    end
                    E(i,1) = mean(mean(E1t));
                end
                
                fitVal(s1,1) = mean(E1t);
                
                fmMat = fm(:,:,rp(4));
                Stxfr = fmMat>0;
                RfmMat = fm(:,:,r1);
                Rtxfr = RfmMat>0;
                for k=1:(2^Ninp-1)
                    if noFunc
                        %E2t(k,1) = (t(k) - ((1./(1+exp(-Rol*(1./(1+exp(-Rhl*x(:,k)))))))>0.5)).^2;
                        hact=(1./(1+exp(-netH(:,:,rp(4))*x(:,k))));
                        menc = Stxfr*hact;
                        mdec = Rtxfr'*menc;
                        mdec(mdec==1)=0.9;
                        mdec(mdec==0)=0.1;
                        hact = mdec;                        
                        E2t(k,1) = (t(k) - ((1./(1+exp(-Rol*hact)))>0.5)).^2;
                    else
                        E2t(k,1) = (t(k) - (logistic(Rol*logistic(Rhl*x(:,k)))>0.5)).^2;
                    end
                end
                E(i,2) = mean([mean(E2t); E(max([i-100 1]):max([i-1 1]),2)]);
            end
        if ~plotEnd
            plot(gca,i,E(i,1),'k.','MarkerSize',10);
            plot(gca,i,E(i,2),'r.','MarkerSize',10);
        end
    end
    
    if exist('G','var')
        if popTopEvo
            s1Neigh = Gj(Gi==s1);
            s1Rec = s1Neigh(find(fitVal(s1Neigh)...
                            ==max(fitVal(s1Neigh))));
            r1Neigh = randsample(Gi(Gj==r1),1);
            G(r1Neigh,r1) = 0;
            G(s1Rec,r1) = 1;           
        end
    end
    
    if waitBarSwitch; 
        waitbar(i/T,hw,''); 
    else
        if mod(i,noteIt)==1
            iterT = toc(titer);
            fprintf('current iter: %0.0f time to finish: %0.2f mins at %0.5f sec per it\n',i,iterT*(T-i)/60,iterT);
        end
    end
end
time=toc(tstart);

if plotEnd
    plot(gca,1:T,E(:,1)','k.','MarkerSize',10);
    plot(gca,1:T,E(:,2)','r.','MarkerSize',10);
    if ~partialIO
        save(datFname,'E');
    end
end

if waitBarSwitch
    waitbar(1,hw,...
    sprintf('Finished in %0.2f seconds at %0.5f per iteration',time,time/T));
else
    fprintf('Finished in %0.2f seconds at %0.5f per iteration\n',time,time/T);
end

if plotErr || plotEnd
    %legend('sender','receiver');
    hold off
end
end

function x = logistic(x)
x = (1+exp(-x)).^(-1);
end