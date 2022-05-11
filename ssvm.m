function [w, gamma, trainCorr, testCorr, cpu_time, nu] = ssvm(C,d,k,nu,output,step_size,tol,maxIter,w0,gamma0)
% version 1.1
% last revision: 01/24/03
%=============================================================================
%Usage: [w gamma trainCorr testCorr nu] = 
%            ssvm (C,d,k,nu,output,step_size,tol,maxIter,w0,gamma0)
%
%
%A and d are both required, everything else has a default
%An example: [w gamma train test nu] = ssvm(A, d, 10);
%
%==============================================================================
%Input parameters:
%
%	A: 	Represent data points (mxn)                        
%	d: 	d is a m dimensional vector of 1's or -1's         
%			containing the corresponding labels for            
%			each data point in A.
%	k: 	way to divide the data set into test and training set
%		if k = 0: simply run the algorithm without any correctness
%							calculation
%	        if k = 1: run the algorithm and calculate correctness on 
%							the whole data set
%		if k = any value less than the # of rows in the data set:
%		       divide up the data set into test and training
%		       using k-fold method
%		if k = # of rows in the data set:
%					use the 'leave 1' method
%						
%	output:	0 - no output, 1 - produce output, default is 0
%	nu:		weighted parameter
%                       -1 - easy estimation
%                       0  - hard estimation
%                       any other value - used as nu by the algorithm
%                       default - 0
%	[w0; gamma0]: 	Initial point
%	step_size:	1 indicates Armijo stepsize, 0 indicates Newton stepsize
%
%==============================================================================
%Output parameters:                                                    
%
%	w:		the normal vector of the classifier                  
%	gamma:		the threshold                                  
%	trainCorr:	training set correctness
%	testCorr:	test set correctness
%	cpu_time:	time elapsed
%       nu:             estimated value (or specified value) of nu
%==============================================================================
%Technical Notes:                                                     
%
%    1. In order to handle a massive dataset this code       
%       takes the advantage of sparsity of the Hessian        
%       matrix.                                              
%                                                            
%    2. We used the limit values of the sigmoid function      
%       and p-function as the smoothing parameter \alpha     
%       goes to infinity when we computer the Hessian        
%       matrix and the gradient of objective function.       
%                                                            
%    3. Decrease nu when the classifier is overfitting       
%       the training data                                     
%                                                            

if nargin<10
gamma0=0;
end
if nargin<9
s=size(C,2);
w0=zeros(s,1);
end
if nargin<8
maxIter=1000;
end
if nargin<7
tol=10e-8;
end
if nargin<6
step_size=1;
end
if nargin<5
output = 0;
end

if ((nargin<4)|(nu==0))
     nu = EstNuLong(C,d);  % default is hard estimation
elseif nu==-1  % easy estimation
  nu = EstNuShort(C,d);
end

if nargin<3
k = 0; 
end

r=randperm(size(d,1));d=d(r,:);C=C(r,:);    % random permutation  

tic;

%move one point in A a little if perfectly balanced
Cback=C;dback=d;   %  backup C and d
[sm sn]=size(C);
ma=C(find(d==1),:); mb=C(find(d==-1),:);
[s1 s2]=size(ma);
c1=sum(ma)/s1;
[s1 s2]=size(mb);
c2=sum(mb)/s1;
if c1==c2
   nu = 1;  % use 1 for perfectly balanced situation
   C(3,:)=C(3,:)+.001*norm(C(3,:)-c1,inf)*ones(1,sn);

end

trainCorr = 0;
testCorr = 0;


   % if k=0 no correctness is calculated, just run the algorithm
if k==0
  [w, gamma, iter] = core(C,d,nu,w0,gamma0,step_size,tol,maxIter,output);
  cpu_time = toc;
  if output==1
  fprintf(1,'\nNumber of Iterations: %d',iter);
   fprintf(1,'\nElapse time: %10.2f\n',cpu_time);
  end
  return
end

%if k==1 only training set correctness is calculated  
if k==1
  [w, gamma, iter] = core(C,d,nu,w0,gamma0,step_size,tol,maxIter,output);
  trainCorr = correctness(C,d,w,gamma);
  cpu_time = toc;
  if output == 1
    fprintf(1,'\nTraining set correctness: %3.2f%% \n',trainCorr);
fprintf(1,'\nNumber of Iterations: %d',iter);
    fprintf(1,'\nElapse time: %10.2f\n',cpu_time);
  end
  return
end

    accuIter = 0;
indx = [0:k];
indx = floor(sm*indx/k);    %last row numbers for all 'segments'
% split trainining set from test set
for i = 1:k
 Ctest = []; dtest = [];Ctrain = []; dtrain = [];

Ctest = C((indx(i)+1:indx(i+1)),:);
dtest = d(indx(i)+1:indx(i+1));

Ctrain = C(1:indx(i),:);
Ctrain = [Ctrain;C(indx(i+1)+1:sm,:)];
dtrain = [d(1:indx(i));d(indx(i+1)+1:sm,:)];

 [w, gamma, iter] = core(Ctrain,dtrain,nu,w0,gamma0,step_size,tol,maxIter,output);
tmpTrainCorr = correctness(Ctrain,dtrain,w,gamma);
tmpTestCorr = correctness(Ctest,dtest,w,gamma);

 if output==1
   fprintf(1,'________________________________________________\n');
fprintf(1,'Fold %d\n',i);
fprintf(1,'Training set correctness: %3.2f%%\n',tmpTrainCorr);
fprintf(1,'Testing set correctness: %3.2f%%\n',tmpTestCorr);    
fprintf(1,'Number of iterations: %d\n',iter);
fprintf(1,'Elapse time: %10.2f\n',toc);
end

trainCorr = trainCorr + tmpTrainCorr;
testCorr = testCorr + tmpTestCorr;
accuIter = accuIter + iter; % accumulative iterations

end % end of for (looping through test sets)

trainCorr = trainCorr/k;
testCorr = testCorr/k;
cpu_time=toc/k;

if output == 1
fprintf(1,'==============================================');
  fprintf(1,'\nTraining set correctness: %3.2f%%',trainCorr);
  fprintf(1,'\nTesting set correctness: %3.2f%%',testCorr);
fprintf(1,'\nAverage number of iterations: %d',accuIter/k);
fprintf(1,'\nAverage cpu_time: %10.2f\n',cpu_time);
end

%%%%%%%%%%%%%%%%%%%%% Core SSVM function %%%%%%%%%%%%%%%%%%%%%%%%%%%

function [w, gamma, iteration] = core(C,d,nu,w0,gamma0,step_size,tol,maxIter,output)

%separating the classes
iteration=0;
[ma, n] = size(C(find(d==1),:)); mb = size(C(find(d==-1),:),1);
C=[C(find(d==1),:);-C(find(d==-1),:)]; % equals "DA" in SSVM paper
d = [ones(ma,1); -ones(mb,1)]; % equal "De" is the paper

flag = 1;
H = zeros(ma+mb,1);
rv = zeros(ma+mb,1); e = ones(ma+mb,1);
while flag > tol & iteration< maxIter   %SZ
%tol 5flag is the optimality condition (smaller the better)
% put a counter to count iterations.
iteration=iteration+1;    %SZ
% Find a search direction!

  temp = C*w0 - gamma0*d; % D(Aw0 -e \gamma0)
  rv =  e -temp; % e - D(Aw0 - e \gamma0)
  % Compute the Hessian matrix              
  H = (e + sign(rv))/2;
  Ih= find(H ~= 0); ih = length(Ih); % We only consider the nonzero part
  Hs = H(Ih); T = speye(ih);
  SH = C(Ih,:)'*spdiags(Hs, 0, T);
  P = SH*C(Ih,:);  q = SH*d(Ih);
  clear SH;
  oneh = norm(Hs,1);
  Q = speye(n+1) +nu*[P,(-q); (-q'), oneh]; % Q is the Hessian matrix
  
  % Compute the gradient  
  prv = max(rv,0); % (e- D(Aw0 - e \gamma0))_+
  gradz = [(w0 - nu*C'*prv); gamma0+nu*d'*prv];
       
  if  gradz'*gradz > tol  % Check the First Order Opt. condition
      b =  - gradz;
      z = Q\b;  % z is the Newton direction

      % Compute the gap! (Only when you want to use the Armijo 's rule!)
  
      gap = z'*gradz;

      % Find the step size & Update to the new point !
       if step_size~=1
           w0 = w0+z(1:n);
           gamma0 = gamma0+z(n+1);
       else
           stepsize = armijo(C,d,w0,gamma0,nu,z, gap);
           w0 = w0 +stepsize*z(1:n);
           gamma0= gamma0 +stepsize*z(n+1);
       end   
      flag = z'*z;
  else
    flag = tol;     %SZ
  end;

% if output==1
%    if (((iteration/10)==floor(iteration/10))|(iteration==1))
%     fprintf(1,'__________________________________________________\n');
%     fprintf(1,'Iteration        Optimality         Elapse Time\n');
%    end
%     fprintf(1,'%d           %12.5f      %10.2f      \n',iteration,flag,toc);
% end


end;  %while


w = w0; gamma = gamma0;
return  % end of core function

%%%%%%%%%%%%%%%% correctness calculation %%%%%%%%%%%%%%%%

function corr = correctness(AA,dd,w,gamma)

p=sign(AA*w-gamma);
corr=length(find(p==dd))/size(AA,1)*100;
return

%%%%%%%%%%%%%Armijo stepsize function%%%%%%%%%%%%%%%%%%%%%%%

function stepsize = armijo(C,d,w,gamma,nu,zd, gap)

% Input
%   C = [A; -B]; equals "DA" in SSVM paper
%   d: equals "De" in SSVM paper (i.e. the diagonal of "D")
%   w, gamma: Current point
%   nu: weight parameter 
%   gap: defined in ssvm code
% Note:
%   You will need objf.m to evaluate the objective function value.
% 

temp =1; n = length(w);
obj1 =  objf(C,d, w,gamma,nu);
w2 = w+temp*zd(1:n); 
gamma2 = gamma +temp*zd(n+1);
obj2 = objf(C,d,w2,gamma2,nu);
diff = obj1 - obj2;
while diff  < -0.05*temp*gap

      temp = 0.5*temp;
      w2 = w+temp*zd(1:n); 
      gamma2 = gamma +temp*zd(n+1);
      obj2 = objf(C,d, w2,gamma2,nu);
      diff = obj1 - obj2;

end;
stepsize = temp;
return
%%%%%%%%%%%%%%%objf.m%%%%%%%%%%%%%%%%%%%%%%%%%%%
function value = objf(C,d,w,gamma,nu)
%
% Evaluate the function value
%

temp = abs(d)-(C*w - gamma*d);
v = max(temp,0);
value = 0.5*(nu*v'*v + w'*w + gamma^2);
return

%%%%%%%%%%%%%%EstNuLong%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hard way to estimate nu if not specified by the user
function value = EstNuLong(C,d)

[m,n]=size(C);e=ones(m,1);
H=[C -e];
if m<201
H2=H;d2=d;
else
r=rand(m,1);
[s1,s2]=sort(r);
H2=H(s2(1:200),:);
d2=d(s2(1:200));
end

lamda=1;
[vu,u]=eig(H2*H2');u=diag(u);p=length(u);
yt=d2'*vu;
lamdaO=lamda+1;

cnt=0;
while (abs(lamdaO-lamda)>10e-4)&(cnt<100)
     cnt=cnt+1;
     nu1=0;pr=0;ee=0;waw=0;
     lamdaO=lamda;
     for i=1:p
     nu1= nu1 + lamda/(u(i)+lamda);
pr= pr + u(i)/(u(i)+lamda)^2;
ee= ee + u(i)*yt(i)^2/(u(i)+lamda)^3;
waw= waw + lamda^2*yt(i)^2/(u(i)+lamda)^2;
   end
lamda=nu1*ee/(pr*waw);
end

value = lamda;
if cnt==100
    value=1;
end    
return

%%%%%%%%%%%%%%%%%EstNuShort%%%%%%%%%%%%%%%%%%%%%%%

% easy way to estimate nu if not specified by the user
function value = EstNuShort(C,d)

value = 1/(sum(sum(C.^2))/size(C,2));
return


