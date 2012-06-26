function [] = nn(plotErr,plotEnd,noFunc,waitBarSwitch,fmFilterSwitch,useToolbox)
clear all; close all;

%% Define switches
plotErr = 1;
plotEnd = 1;
noFunc = 1;
waitBarSwitch = 0;
fmFilterSwitch = 1; 
useToolbox = 0;
noteIt = 300;


%% Define parameters
N = 9;      % population size
T = 20000;     % number of training steps
trS = 50;   % size of training set
Ninp = 8;   % number of inputs
Nhid = 10;  % number of hidden layer nodes
Nout = 1;   % number of output nodes
Nch = 3;    % number of consecutive 1's in the K-in-N game
WR = [-0.5 0.5]; % range of nn weights;
delta = 0.05;
epsil = 0.01;

%% Initialize variables
E1t = zeros(N,2^Ninp-1);
E2t = zeros(2^Ninp-1,1);
E1 = zeros(T,1);
E2 = zeros(T,1);

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

%% Select indices for training given a population structure
ps = 'mixing';
if strcmp(ps,'mixing')
    rp=randperm(N); i1=rp(1); j1=rp(2);
elseif strcmp(ps,'simple')
    i1=1; j1=2;
elseif strcmp(ps,'graph')
    i1=1; j1=2;
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
for i=1:T
    if mod(i,noteIt)==1; titer = tic; end
% Select a training example
    si = randi([1 trS]);
    xi = rind(si);
      
% Decompose the receiver into parts representing hl and ol
if useToolbox
    ol.IW{1} = net{j1}.LW{2,1};
    ol.b{1} = net{j1}.b{2};
    
    hl.IW{1} = net{j1}.IW{1};
    hl.b{1} = net{j1}.b{1};    
else
    Shl = netH(:,:,i1);
    Sol = netO(:,:,i1);
    Rhl = netH(:,:,j1);
    Rol = netO(:,:,j1);
end

% Compute sender's hidden layer activation levels
if useToolbox
    hact = +(logistic(net{i1}.IW{1}*x(:,xi)+net{i1}.b{1},0.1)>0.5); % sender hidden layer activation levels
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
        fmMat = fm(:,:,i1);
        
        for j=1:length(fmInd)
            maxInd = find( fmMat(:,fmInd(j)) == max(fmMat(:,fmInd(j))), 1 );
            fmMat(maxInd,fmInd(j)) = fmMat(maxInd,fmInd(j))+delta+2*epsil;
            fmMat(:,fmInd(j)) = fmMat(:,fmInd(j)) - epsil;
            fmMat(maxInd,:) = fmMat(maxInd,:) - epsil;
        end
        
        fm(:,:,i1) = fmMat;
        Stxfr = fmMat>0;
        menc = Stxfr*hact;
        
% Decode sender's message with receiver's form-meaning mapping
        RfmMat = fm(:,:,j1);
        Rtxfr = RfmMat>0;
        mdec = Rtxfr'*menc;
        mdec(mdec==1)=0.9; 
        mdec(mdec==0)=0.1;
        hact = mdec;
    end
    
% Use sender's hidden layer activation levels on receiver's hidden layer
% and perform back propagation on this basis to compute expected activation
% levels
if useToolbox
    d = t(xi) - sim(ol,hact); % error
    ht = hact + d*(ol.IW{1}'+ol.b{1}); % training set for hidden layer
else
    if noFunc
        d = (t(xi) - 1./(1+exp(-Rol*hact)));
    else
        d = (t(xi) - logistic(Rol*hact));
    end
    ht = hact + d*Rol';
    z = ht>0;   
end

% Update receiver's form-meaning mapping
if fmFilterSwitch
    fmInd = find(z==1);
        fmMat = fm(:,:,j1);
        
        for j=1:length(fmInd)
            maxInd = find( fmMat(:,fmInd(j)) == max(fmMat(:,fmInd(j))), 1 );
            fmMat(maxInd,fmInd(j)) = fmMat(maxInd,fmInd(j))+delta+2*epsil;
            fmMat(:,fmInd(j)) = fmMat(:,fmInd(j)) - epsil;
            fmMat(maxInd,:) = fmMat(maxInd,:) - epsil;
        end
    fm(:,:,j1) = fmMat;
end

% Update/train the decomposed receiver and sender
if useToolbox    
    % Train receiver
    [hl,Yhl,Ehl,Pfhl] = adapt(hl,x(:,xi),ht);
    [ol,Yol,Eol,Pfol] = adapt(ol,hact,t(xi));
    net{j1}.LW{2,1} = ol.IW{1};
    net{j1}.b{2} = ol.b{1};
    net{j1}.IW{1} = hl.IW{1};
    net{j1}.b{1} = hl.b{1};

    % Train the sender
    net{i1} = adapt(net{i1},x(:,xi),t(xi));
else        
    % Train the sender
    if noFunc
        H = 1./(1+exp(-Shl*x(:,xi)));
        O = 1./(1+exp(-Sol*H));
    else
        H = logistic(Shl*x(:,xi));
        O = logistic(Sol*H);
    end
    dSol = (t(xi) - O);
    dShl = Sol'*dSol;
    Sol = Sol + dSol*H';
    Shl = Shl + dShl*x(:,xi)';
    
    % Train the receiver
    if noFunc
        dRol = (t(xi) - 1./(1+exp(-Rol*hact))); % computed above as d    
        dRhl = (z - 1./(1+exp(-Rhl*x(:,xi))));    
    else
        dRol = (t(xi) - logistic(Rol*hact)); % computed above as d    
        dRhl = (z - logistic(Rhl*x(:,xi)));    
    end
    Rol = Rol + dRol*hact'; 
    Rhl = Rhl + dRhl*x(:,xi)';
    
    % Put updated weights back into population
    netH(:,:,i1) = Shl;
    netO(:,:,i1) = Sol;
    netH(:,:,j1) = Rhl;
    netO(:,:,j1) = Rol;
end

% Compute data for plotting
    if plotErr
            if useToolbox
                for j=1:N
                    E1t(j) = mean((t-net{j}(x)).^2);
                end
                E1 = mean(E1t);
                E2 = mean((t-net{j1}(x)).^2);
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
                E1(i,1) = mean(mean(E1t));
                for k=1:(2^Ninp-1)
                    if noFunc
                        E2t(k,1) = (t(k) - ((1./(1+exp(-Rol*(1./(1+exp(-Rhl*x(:,k)))))))>0.5)).^2;
                    else
                        E2t(k,1) = (t(k) - (logistic(Rol*logistic(Rhl*x(:,k)))>0.5)).^2;
                    end
                end
                E2(i,1) = mean(E2t);
            end
        if ~plotEnd
            plot(gca,i,E1(i),'k.','MarkerSize',10);
            plot(gca,i,E2(i),'r.','MarkerSize',10);
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
    plot(gca,1:T,E1','k.','MarkerSize',10);
    plot(gca,1:T,E2','r.','MarkerSize',10);
    save(['nnErr' datestr(now,'yyyymmddHHMMSS')],'E1','E2');
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