function [L,K,V] = lda(X,ns,nt,n)
% [L,K,V]=lda(X,ns,nt,n)
%
% L - Eigenvectors(sorted) of matrix Z. Each column represents an eigenvector.
% K - Eigenvalues (sorted) of matrix Z. It is a column vector.
% V - Variance explained by each corresponding eigenvalues.
%
% X - Matrix containing the train. set, whose each line points to a sample data.
% ns- Number of distinct subjects (groups) readen.
% nt- Vector of the number of photos of each subject used as training sample.
% n - Reduction of dimension (n <= ns-1).
% 
% Carlos Thomaz, DOC-IC/LONDON, 15/01/2004.

%-------------------------------------------------------------------------------
% Validation
%-------------------------------------------------------------------------------

if ((n <= 0) | (n>=size(X,1))), error('Reduction of dimension (n) invalid.'),end
if (n>size(X,2)), error('Reduction of dimension (n) invalid.'),end

%-------------------------------------------------------------------------------
% Computation of LDA
%-------------------------------------------------------------------------------

O = ones(size(X,1),1);               % Auxiliar Matrix (all ones)
H = repeatc(eye(ns),nt)';            % Auxiliar Matrix (ones and zeros)

M = mean(X);                         % Matrix of Total sample mean
Mg= meang(X,ns,nt);                  % Matrix of Group sample mean

W = X-(H*Mg);                        % Matrix of Within-Groups deviations
B = (H*Mg)-(O*M);                    % Matrix of Between-Groups deviations

Sw= W'*W;                            % Within-Group scatter matrix
Sb= B'*B;                            % Between-Group scatter matrix

[Qv,Qa] = eig((inv(Sw)*Sb));         % Qv eigenvectors and Qa eigenvalues

%-------------------------------------------------------------------------------
% Sort
% Sort in descending order the eigenvalues.
% Eigenvectors whose have the biggest eigenvalues at the beginning of the matrix
%-------------------------------------------------------------------------------

[Ks,Ki] = sort(diag(Qa));
Ki = flipud(Ki);

L  = Qv(:,Ki(1:n));
K  = flipud(Ks);
K  = K(1:n);                             % Return only n eigenvalues
V = K./sum(Ks);

%-------------------------------------------------------------------------------
