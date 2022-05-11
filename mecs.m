function [Se,tt] = mecs(Sp,Sg)
% [Se,tt] = mecs(Sp,Sg)
% A maximum entropy covariance selection (MECS) is performed so that the entropy
% of each class is maximised.
%
% Output
% Se - The Smix matrix based on the maximum entropy.
% tt - CPU total time in seconds of MECS calculation.
%
% Input 
% Sp - Spooled (common) covariance matrix.
% Sg - Sample group covariance matrix.
%
% Carlos Thomaz, DoC-IC/London, 29-oct-2001.

%-------------------------------------------------------------------------------
% Validation
%-------------------------------------------------------------------------------

if nargin ~= 2, error('Requires two input arguments.') , end

%-------------------------------------------------------------------------------
% MECS procedure
%-------------------------------------------------------------------------------

p  = size(Sp,1);                            % Number of variables (dimension)
t0 = cputime;                               % CPU time of MECS

Sm = Sp + Sg;                               % Linear combination of Sp and Sg
[Qm,Dm] = eig(Sm);                          % Eigenvectors/values of Sm

Dg = diag(Qm'*Sg*Qm);                       % Sg contribution on Sm eigenspace
Dp = diag(Qm'*Sp*Qm);                       % Sp contribution on Sm eigenspace

De = diag(max(Dg(1:p),Dp(1:p)));            % Maximum eigenvalues selection
Se = Qm*De*Qm';                             % New Smix based on maximum values

tt = cputime - t0;                          % CPU total time of MECS

%-------------------------------------------------------------------------------
