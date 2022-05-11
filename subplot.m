nt1=31;
nt2=21;

mds1 = Y(1:nt1,1);          % Most discriminants of sample1
mds2 = Y(nt1+1:nt1+nt2,1);  % Most discriminants of sample2

mg(1) = mean(mds1);
mg(2) = mean(mds2);
vg(1) = (std(mds1))^2;
vg(2) = (std(mds2))^2;

[mmin, nmin] = min(mg);
[mmax, nmax] = max(mg);

xmin = mmin - 5*sqrt(vg(nmin));
xmax = mmax + 5*sqrt(vg(nmax));

xs = [xmin:(xmax-xmin)/100:xmax];

g1 = (1/(sqrt(2*pi*vg(1))))*exp((-1/(2*vg(1)))*((xs-mg(1)).^2));
g2 = (1/(sqrt(2*pi*vg(2))))*exp((-1/(2*vg(2)))*((xs-mg(2)).^2));

%figure
subplot(2,1,1)
plot (xs,g1./max(g1),'r-',xs,g2./max(g2),'b-.', 'LineWidth', 1.5); % Corresponding Gaussian Curves
xlim([min(xs) max(xs)])

subplot(2,1,2)
p1 = plot(data1,0,'ro','MarkerSize',10); hold on   % Plot data set 1
p2 = plot(data2,0,'bo','MarkerSize',10); hold on   % Plot data set 2
%p5 = plot(data3,0,'gd','MarkerSize',10); hold on   % Plot data set 3
mRM = mean(data1); mRNM = mean(data2); mRt = [mRM mRNM]; % Media dos grupos
p3 = plot(mRt(1),0,'rx','MarkerSize',20, 'LineWidth', 1.5); hold on % Plot Media dos grupos
p4 = plot(mRt(2),0,'bx','MarkerSize',20, 'LineWidth', 1.5); hold on % Plot Media dos grupos
legend([p1(1), p2(1)],'Proficientes', 'Não Proficientes');

xlim([min(xs) max(xs)])