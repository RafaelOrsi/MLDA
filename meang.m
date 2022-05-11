function Y = meang(X,ns,n)
% Y=meang(X,ns,n)
% Mean's Group (cluster) values for unequal samples per class.
%
% X  - Matrix of samples, whose each line points to a sample data.
% ns - Number of subjects or groups.
% n  - Vector or scalar containing the number of samples per group.
%
% Obs : If n is a scalar then all groups have the same number of samples.
%	Samples of the same group must be placed together in X.
%
% Carlos Thomaz, DoC-IC/London, 16/jan/2004.

%-------------------------------------------------------------------------------
% Validation
%-------------------------------------------------------------------------------

if nargin~=3, error('Requires three input arguments.') , end

if (size(n,1)==size(n,2)) && (size(n,1)==1)                 % scalar
    N(1,1:size(X,2)) = n;                                              % transform to vector
else
    if (size(X,1)~=sum(n)), error('Total number of samples invalid.'),end
    N = n;
end  

%-------------------------------------------------------------------------------
% Computation of mean's (centroid's) group.
%-------------------------------------------------------------------------------

s = 1;

for i = 1 : ns
    k = 1;
    for j = s : (s+N(i)-1)
        Yaux(k,:) = X(j,:);
        k = k + 1;
    end
    s = s + N(i);
    Y(i,:) = mean(Yaux,1);
    clear Yaux;
end

%-------------------------------------------------------------------------------
