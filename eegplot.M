function eegplot(n1,n2,Zout)
% eegplot(n1x,n2x,Zout)
% EEG scatter plot. Type 'type eegplot' for details.
% 
% n1 - Number of samples of group 1 (CT: controls)
% n2 - Number of samples of group 2 (LD: learning disabilities)
% If Zout != 0 then LDA shifts are saved on Z files.
% 
% Carlos Thomaz, FEI/PS, 26/09/2014.

%-------------------------------------------------------------------------------

ng = 2;                                            % # of groups: CT and LD
%%%X1 = load('c:\personal\fabio\data\normais.txt');   % CT: controls
%%%X2 = load('c:\personal\fabio\data\disturbio.txt'); % LD: learning disabilities

%%%X1 = load('c:\personal\fabio\data\mp1.txt');   % Masculino Protocolo 1
%%%X2 = load('c:\personal\fabio\data\mp2.txt');   % Masculino Protocolo 2

%%%X1 = load('c:\personal\fabio\data\fp1.txt');      % Feminino Protocolo 1
%%%X2 = load('c:\personal\fabio\data\fp2.txt');      % Feminino Protocolo 2

%%%X1m = load('c:\personal\fabio\data\mp1.txt');      % Masculino Protocolo 1
%%%X2m = load('c:\personal\fabio\data\mp2.txt');      % Masculino Protocolo 2
%%%X1f = load('c:\personal\fabio\data\fp1.txt');      % Feminino Protocolo 1
%%%X2f = load('c:\personal\fabio\data\fp2.txt');      % Feminino Protocolo 2
%%%X1 = [X1m;X1f];
%%%X2 = [X2m;X2f];

%%%X1 = load('c:\personal\fabio\data\mp1.txt');   % Masculino Protocolo 1
%%%X2 = load('c:\personal\fabio\data\fp1.txt');   % Feminino Protocolo 1

X1 = load('c:\personal\fabio\data\mp2.txt');   % Masculino Protocolo 2
X2 = load('c:\personal\fabio\data\fp2.txt');   % Feminino Protocolo 2


%-------------------------------------------------------------------------------

X = [X1(1:n1,:);X2(1:n2,:)];

m = mean(X);
save c:/personal/fabio/results/m.txt m -ASCII -DOUBLE

X = X - ones(size(X,1),1)*m;
w = lda(X,ng,[n1 n2],ng-1);
%%%w = ssvm(X,[ones(n1,1);-ones(n2,1)]);

disp('Saving the LDA loadings...');
save c:/personal/fabio/results/w.txt w -ASCII -DOUBLE

Z = X * w;

%-------------------------------------------------------------------------------

figure

plot(Z(1:n1,1),1:n1,'bo')
hold on
plot(Z(n1+1:n1+n2,1),1:n2,'rx')
grid on
title('Most Discriminant Feature')
xlabel('LDA hyperplane')
ylabel('Subject Index in the Sample Set')
legend('1','2')
axis tight

%-------------------------------------------------------------------------------

Z1 = Z(1:n1,1);                       % Most discriminants of sample 1
Z2 = Z(n1+1:n1+n2,1);                 % Most discriminants of sample 2

mg(1) = mean(Z1);
mg(2) = mean(Z2);
vg(1) = (std(Z1))^2;
vg(2) = (std(Z2))^2;

[mmin nmin] = min(mg);
[mmax nmax] = max(mg);

xmin = mmin - 5*sqrt(vg(nmin));
xmax = mmax + 5*sqrt(vg(nmax));

xs = [xmin:(xmax-xmin)/100:xmax];

g1 = (1/(sqrt(2*pi*vg(1))))*exp((-1/(2*vg(1)))*((xs-mg(1)).^2));
g2 = (1/(sqrt(2*pi*vg(2))))*exp((-1/(2*vg(2)))*((xs-mg(2)).^2));

%-------------------------------------------------------------------------------

figure

plot(xs,g1./max(g1),'-b',xs,g2./max(g2),'-r');  % Corresponding Gaussian curves

title('Most Discriminant Feature')
xlabel('LDA hyperplane')
legend('1','2')
axis tight

%-------------------------------------------------------------------------------

if Zout == 0, return, end

b = (mg(1)+mg(2))/2;                            % Boundary (no priori)

disp('Creating Z-files...');

if (mg(1) < mg(2))
   from_mean = mg(1);
   from_std  = sqrt(vg(1));
   to_mean   = mg(2);
   to_std    = sqrt(vg(2));
else
   from_mean = mg(2);
   from_std  = sqrt(vg(2));
   to_mean   = mg(1);
   to_std    = sqrt(vg(1));
end

j = 1;

for i = -3:3
    new_shift = from_mean + from_std*i;
    if new_shift < b
       Z_shift(j) = new_shift;
       j = j + 1;
    end   
end    

Z_shift(j) = b;
j = j + 1;

for i = -3:3
    new_shift = to_mean + to_std*i;
    if new_shift > b
       Z_shift(j) = new_shift;
       j = j + 1;
    end   
end    

disp('Saving the shifts (feature space)...');
save c:/personal/fabio/results/z.txt Z_shift -ASCII -DOUBLE
Z_shift(:)

%-------------------------------------------------------------------------------

figure

plot(Z(1:n1,1),1:n1,'bo')
hold on
plot(Z(n1+1:n1+n2,1),1:n2,'rx')
plot(mg(1),0,'bs')
plot(b,0,'ks')
plot(mg(2),0,'rs')
plot(Z_shift,0,'k*')
grid on
title('Most Discriminant Feature')
xlabel('LDA hyperplane')
ylabel('Subject Index in the Sample Set')
legend('1','2')
axis tight

%-------------------------------------------------------------------------------

for i = 1:size(Z_shift,2)
    Y_shift(i,:) = m + (Z_shift(i) .* w)';    % with the mean
end

Y_shift = Y_shift';

disp('Saving the shifts (original space)...');
save c:/personal/fabio/results/y.txt Y_shift -ASCII -DOUBLE
return

%-------------------------------------------------------------------------------
