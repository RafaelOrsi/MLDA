function [P,K,V] = pca(Z,n)
% [P,K,V]=PCA(Z,n)
%
% P - Eigenvectors(sorted) of matrix Z. Each column represents an eigenvector.
% K - Eigenvalues (sorted) of matrix Z. It is a column vector.
% V - Variance explained by each eigenvalue. It is a column vector.
%
% Z - Matrix data with zero mean (each line points to a sample).
% n - Reduction of dimension, where n is within [1,min(size(Z,1)-1,size(Z,2))].
% 
% Carlos Thomaz, SPMMRC/Nottingham, 23/apr/2012.

%-------------------------------------------------------------------------------

if (n <= 0), error('Reduction of dimension (n) invalid.'),end
if (n >= size(Z,1)), error('Reduction of dimension (n) invalid.'),end
if (n > size(Z,2)), error('Reduction of dimension (n) invalid.'),end

h = waitbar(0,'Preallocating the PCA memory required...');
P = zeros(size(Z,2),n);

m = size(Z,1);                             % Number of lines or observations

if (m > size(Z,2))
   V = Z'*Z ./(m-1);
else
   V = Z* Z'./(m-1);
end

waitbar(0,h,'Calculating eigenvectors/values of the covariance matrix...');
[Qv,Qa] = eig(V, 'nobalance');             % Qv(eigenvects), Qa(eigenvals)

waitbar(0,h,'Sorting eigenvector/eigenvalues...');
[Ks,Ki] = sort(diag(Qa));
Ki = flipud(Ki);
Qv_s = Qv(:,Ki(1:n));
Qa_s = flipud(Ks);

if (m > size(Z,2))
   P = Qv_s;
else
   for i = 1:n
       waitbar(0,h,sprintf('Calculating eigenvectors... (%d/%d)',i,n));
       P(:,i)=(Z' * Qv_s(:,i))./(sqrt(Qa_s(i)*(m-1)));
       waitbar(i/n,h);
   end
%--Pt = (Z' * Qv * (Qa^(-0.5)))./(sqrt(m-1)); % Eigenvects of outer prod. of Z
%--P  = Pt(:,Ki(1:n));
end

close(h);

K = Qa_s(1:n);
V = K./sum(Qa_s);

%-------------------------------------------------------------------------------
