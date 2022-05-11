function [P,K,L,F] = fa(X,n)
% [P,K,L,F,]=FA(X,n)
%
% P - Principal components (sorted) of X. Columns are eigenvectors.
% K - Eigenvalues (sorted) of matrix X. It is a column vector.
% L - Factor loadings (sorted) of X. Columns are factors.
% F - Rotated L (sorted) of X. Columns are rotated factors.
%
% X - Matrix data (each line points to a sample).
% n - Reduction of dimension, where n is within [1,min(size(Z,1)-1,size(Z,2))].
% 
% Carlos Thomaz, FEI/SP, 01/oct/2014.

%-------------------------------------------------------------------------------

if (n <= 0), error('Reduction of dimension (n) invalid.'),end
if (n >= size(X,1)), error('Reduction of dimension (n) invalid.'),end
if (n > size(X,2)), error('Reduction of dimension (n) invalid.'),end

disp('Standardizing the variables of X...');
m = mean(X);
s = std(X);

for i = 1:size(X,2)
   X(:,i) = (X(:,i) - m(i))./s(i);
end

disp('Calculating the principal components of the correlation matrix...');
[P,K] = pca(X,n);

disp('Calculating the factor loadings...');
for i = 1:size(P,2)
    L(:,i) = P(:,i) .* sqrt(K(i));
end

disp('Rotating the factor loadings...');
F = rotatefactors(L(:,1:n),'method','varimax');

return

%-------------------------------------------------------------------------------
