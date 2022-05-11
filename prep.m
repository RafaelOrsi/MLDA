function  Data = prep(database)
w = database;
x = 1:length(w);
% ETAPA 1 - Remo��o de outliers
w2 = w;
fator = 0.06;
for i = 1:(length(w)-1)
    if(abs(w(i,1)-w(i+1,1)) > fator)
        w2(i+1,1) = 0; 
    end
end
for i = (length(w)-1):1
    if(abs(w(i,1)-w(i+1,1)) > fator)
        w2(i,1) = 0; 
    end
end
w = w2;

% ETAPA 2 - Interpola��o para preencher dados perdidos
for i = 1:length(w)
    if(w(i,1)== 0)
        w(i,1) = NaN; %Converte sinal (zero) em NaN
    end
end
w = fillmissing(w,'linear'); %VER EXEMPLO DESTA FUN��O NO MATLAB PARA USAR NO TRABALHO

% ETAPA 3 - Filtro para remover ru�do
windowSize = 5;
b = (1 / windowSize) * ones (1, windowSize);
a = 1;
Data = filtfilt(b,a,w);
end