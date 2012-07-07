% Batch script for nn.m
%
% see http://www.it.northwestern.edu/research/sscc/batchjobs.html for
% example using SGE
%
% Example usage:
%   nn(N,T,ps,noteIt,plotErr,plotEnd,ergodicErr,noComm,partialIO,noFunc,popTopEvo,plotGrph,waitBarSwitch,fmFilterSwitch,useToolbox)
%   nn(100,2000,'grid',1000,1,1,1,0,0,1,1,0,0,1,0)

job1 = batch('nn',1,{100,2000,'grid',1000,1,1,1,0,0,1,1,0,0,1,0},'FileDependencies','genKinN')
%wait(job1);
%load(job1)
%destroy(job1)

pause(1)
job2 = batch('nn',1,{1024,2000,'grid',1000,1,1,1,0,0,1,1,0,0,1,0},'FileDependencies','genKinN')

%destroy(job1)
%destroy(job2)