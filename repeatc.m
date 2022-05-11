function Y = repeatc(X,n)
% Y=repeatc(X,n)
% Builds a matrix X by repeating each column i of matrix X n(i) times.
% The parameter n might be a vector or a scalar.
% If n is a scalar then all columns of X will be repeated equally n times.
%
% Carlos Thomaz, DoC-IC/London, 15/jan/2004.

%-------------------------------------------------------------------------------

if nargin~=2, error('Requires two input arguments.'), end

if (size(n,1)==size(n,2)) && (size(n,1)==1)                 % scalar
    N(1,1:size(X,2)) = n;                                   % transform to vector
else
    N = n;
end    
 
%-------------------------------------------------------------------------------

Y = zeros(size(X,1),sum(N));
cx = size(X,2);
cy = 1;

for i = 1:cx
    for j = 1:N(i)
        Y(:,cy)=X(:,i);
        cy=cy+1;
    end
end

%-------------------------------------------------------------------------------
