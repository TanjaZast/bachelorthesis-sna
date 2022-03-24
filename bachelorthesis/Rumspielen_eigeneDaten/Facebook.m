clear all
clearvars
T=readtable('/Users/tanjazast/Desktop/Bachelorarbeit/CSV/Interaction.csv');
T.data_name = str2double(T.data_name)
T.data_uri = str2double(T.data_uri)
A = table2array(T);
disp(A)
n=A(:,1);
n1 = n';
e=A(:,2);
e1 = e';
e2 = int64(e');
disp(n1)
disp(e2)
G = graph(n1,e2);
plot(G)
title('Facebook Friends')
