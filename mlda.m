function [L,K,V] = mlda(X,ns,nt,n)
% [L,K,V]=mlda(X,ns,nt,n)
%
% L - Eigenvectors(sorted) of matrix X. Each column represents an eigenvector.
% K - Eigenvalues (sorted) of matrix X. It is a column vector.
% V - Variance explained by each corresponding eigenvalue.
%
% X - Matrix containing the train. set, whose each line points to a sample data.
% ns- Number of distinct groups readen.
% nt- Vector of the number of samples of each group used as training sample.
% n - Reduction of dimension (n <= ns-1).
% 
% Carlos Thomaz, Doc-IC/London, 23/jan/2004.

%-------------------------------------------------------------------------------
% Validation
%-------------------------------------------------------------------------------

if ((n <= 0) | (n>=size(X,1))), error('Reduction of dimension (n) invalid.'),end
if (n>size(X,2)), error('Reduction of dimension (n) invalid.'),end

%-------------------------------------------------------------------------------
% Useful and auxiliar matrices
%-------------------------------------------------------------------------------

O = ones(size(X,1),1);               % Auxiliar Matrix (all ones)
H = repeatc(eye(ns),nt)';            % Auxiliar Matrix (ones and zeros)

M = mean(X);                         % Matrix of Total sample mean
Mg= meang(X,ns,nt);                  % Matrix of Group sample mean

W = X-(H*Mg);                        % Matrix of Within-Groups deviations
Sw= W'*W;                            % Within-Group scatter matrix

B = (H*Mg)-(O*M);                    % Matrix of Between-Groups deviations
Sb= B'*B;                            % Between-Group scatter matrix

%-------------------------------------------------------------------------------
% MDA with MECS approach on blending Sp and I
%-------------------------------------------------------------------------------

Sp= Sw./(size(X,1)-ns);              % Matrix of pooled Within-Groups
p = size(Sp,1);                      % Dimension

Ip = (trace(Sp)./p).*eye(p);         % Identity matrix scaled

Sm = mecs(Sp,Ip);                    % MECS approach
Sw = Sm.*(size(X,1)-ns);             % New Within-Group scatter matrix

[Qv,Qa] = eig((inv(Sw)*Sb));         % MDA - Qv eigenvectors and Qa eigenvalues

%-------------------------------------------------------------------------------
% Sort
% Sort in descending order the eigenvalues.
% Eigenvectors whose have the biggest eigenvalues at the beginning of the matrix
%-------------------------------------------------------------------------------

[Ks,Ki] = sort(diag(Qa));
Ki = flipud(Ki);

L = Qv(:,Ki(1:n));
K = flipud(Ks);
K = K(1:n);
V = K./sum(Ks);

%-------------------------------------------------------------------------------
