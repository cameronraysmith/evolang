function [status] = nn(N,T,ps,noteIt,plotErr,plotEnd,ergodicErr,noComm,partialIO,noFunc,popTopEvo,plotGph,waitBarSwitch,fmFilterSwitch,useToolbox)
%clear all; close all;

%-------------------------------------
% 
% Dependencies:
% m-files
%   genKinN.m - necessary to generate training set    
% 
% requires MatlabBGL
%   http://dgleich.github.com/matlab-bgl/
%   Run
%   >>help Contents 
%   to view methods defined in MatlabBGL
%
%   may eventually require Waterloo file and matrix utilities
%   Direct download: http://goo.gl/NjSKE
%   http://sigtool.sourceforge.net/
%   http://www.mathworks.com/matlabcentral/fileexchange/12250
%   uses: nmatrix
% 
% Example usage:
%   nn(N,T,ps,noteIt,plotErr,plotEnd,ergodicErr,noComm,partialIO,noFunc,popTopEvo,plotGrph,waitBarSwitch,fmFilterSwitch,useToolbox)
%   nn(100,2000,'grid',1000,1,1,1,0,0,1,1,0,0,1,0)
% 
% Abbreviations
%   NYI = not yet implemented
%-------------------------------------

%% Define switches
% ps = 'grid'; % test | mixing | grid | rand
% 
% noteIt     = 1000; % print out timing information every noteIt iterations
% plotErr    = 1;    % save training error
% plotEnd    = 1;    % plot training error when simulation finishes
% ergodicErr = 1;    % use running average rather than iterating through entire population to compute error
% noComm     = 0;    % NYI test training on population of non-interacting individuals
% partialIO  = 0;    % not yet working but may attempt to use Waterloo file and matrix utilities 
% noFunc     = 1;    % avoid all function calls for efficiency
% popTopEvo  = 1;    % population topology evolves as a function of interindividual communication and recommendation
% plotGph    = 0;    % plot population structure graph at end of simulation...probably a bad idea
% 
% waitBarSwitch  = 0; % show wait bar during computation, slows down simulation
% fmFilterSwitch = 1; % filter communication through encoding/decoding permutation matrices
% useToolbox     = 0; % use neural network toolbox...not a good idea as it is extremely inefficient
% 
% %% Define parameters
% N = 100;           % population size, must be perfect square
% T = 2000;        % number of training steps
trS = 50;           % size of training set
Ninp = 8;           % number of inputs
Nhid = 10;          % number of hidden layer nodes
Nout = 1;           % number of output nodes
Nch = 3;            % number of consecutive 1's in the K-in-N game
WR = [0 1];         % range of nn weights; the neural.m file uses [0 1] and paper uses [-.5 .5]
delta = 0.05;       % negative increment for form-meaning mapping
epsil = 0.01;       % positive increment for form-meaning mapping
LR = 2.0;           % learning rate neural.m uses 2.6
ergErrWindow = 50;  % error window for moving average
pRandAttach = 0.1;  % probability of receiver moving an out-edge to a random node rather than to the recommendation of the sender

datFname = datestr(now,'yyyymmddHHMMSS'); % directory name for simulation output files
system(['mkdir ' datFname]);
writeIter = 1000;   % not yet in use...will indicate the iteration modulus at which data stored in memory will be written to disk

lfid = fopen([datFname '/nn.log'],'a');
fprintf( lfid,'Run ID: %s\n', datFname );
save([datFname '/params.mat']);


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
    datH=MATOpen(['nnErr' datFname],'a') ;
else
    E = zeros(T,2);
end

% fitness
fitVal = zeros(N,1);    %initialize fitness vector to be measured


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
    r1=Gi(rp(1)); s1=Gj(rp(1)); % receiver has out-edge directed at sender
    rp(3) = randsample(N,1);  % index to random individual for sender error computation
    rp(4) = randsample(Gj(Gi==r1),1); % index to random individual with arrow directed to r1 to serve as a sender in error computation
elseif strcmp(ps,'rand')
%     p = 1/3;
%     [G xy]=erdos_reyni(N,p);
    [Gi Gj Gw]=find(G);
    rp=randperm(num_edges(G));
    r1=Gi(rp(1)); s1=Gj(rp(1));% receiver has out-edge directed at sender
    rp(3) = randsample(N,1);  % random individual for sender error computation
    rp(4) = randsample(Gj(Gi==r1),1); % random individual with arrow directed to r1 to serve as a sender in error computation
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

if ~noComm
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

end % noComm

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

    
if ~noComm    
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
    netH(:,:,r1) = Rhl;
    netO(:,:,r1) = Rol;
end % noComm    
    netH(:,:,s1) = Shl;
    netO(:,:,s1) = Sol;
    
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
                    E(i,1) = mean([mean(E1t); E(max([i-ergErrWindow 1]):max([i-1 1]),1)]);
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
if ~noComm                
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
                E(i,2) = mean([mean(E2t); E(max([i-ergErrWindow 1]):max([i-1 1]),2)]);
end % noComm                
            end
        if ~plotEnd
            plot(gca,i,E(i,1),'k.','MarkerSize',10);
            plot(gca,i,E(i,2),'r.','MarkerSize',10);
        end
    end
    
    if exist('G','var')
        if popTopEvo
            if binornd(1,pRandAttach)
                s1Rec = randsample(N,1);
            else
                if length(Gj(Gi==s1))>0
                    s1Neigh = Gj(Gi==s1);
                    s1Neigh = s1Neigh(s1Neigh~=s1);
                    s1Rec = s1Neigh(find(fitVal(s1Neigh)...
                        ==min(fitVal(s1Neigh))));
                    if length(s1Rec)>1
                        s1Rec = randsample(s1Rec,1);
                    elseif length(s1Rec)==0
                        s1Rec = s1;
                    end
                else
                    s1Rec = s1;
                end
            end
                if length(Gj(Gi==r1))>0
                    r1Neigh = randsample(Gj(Gi==r1),1);
                else
                    r1Neigh = s1;
                end
            
                G(r1,r1Neigh) = 0;
                G(r1,s1Rec) = 1;
            end
        end
        
        if waitBarSwitch;
            waitbar(i/T,hw,'');
        else
            if mod(i,noteIt)==1
                iterT = toc(titer);
                fprintf('current iter: %0.0f time to finish: %0.2f mins at %0.5f sec per it\n',i,iterT*(T-i)/60,iterT);
                fprintf(lfid,'current iter: %0.0f time to finish: %0.2f mins at %0.5f sec per it\n',i,iterT*(T-i)/60,iterT);
            end
        end
    end
    time=toc(tstart);
    
    if plotEnd
        plot(gca,1:T,E(:,1)','k.','MarkerSize',10);
        plot(gca,1:T,E(:,2)','r.','MarkerSize',10);
        if ~partialIO
            save([datFname '/nnErr' ],'E','G');
        end
        [Gi Gj Gw]=find(G);
        Gout = [Gi Gj];
        fid = fopen([datFname '/gephi.csv'],'w');
        %fprintf( fid,'Source, \t Target,\n' );
        %fprintf( fid,'%d, \t %d, \t %f,\n', Gout );
        fprintf( fid,'%d,\t%d,\n', Gout );
        fclose(fid);
        fid = fopen([datFname '/fitVal.csv'],'w');
        fprintf( fid,'%f,\n', fitVal );
        fclose(fid);
        if plotGph
            gVizPlot(G,1,datFname);
        end
    end
    
    if waitBarSwitch
        waitbar(1,hw,...
            sprintf('Finished in %0.2f seconds at %0.5f per iteration',time,time/T));
    else
        fprintf('Finished in %0.2f seconds at %0.5f per iteration\n',time,time/T);
        fprintf(lfid,'Finished in %0.2f seconds at %0.5f per iteration\n',time,time/T);
    end
    
    if plotErr || plotEnd
        %legend('sender','receiver');
        hold off
    end
    
    fclose(lfid);
    status = 1;
end

    function x = logistic(x)
        x = (1+exp(-x)).^(-1);
    end