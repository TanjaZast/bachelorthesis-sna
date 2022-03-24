n = 30;
e = 100;

s = randi(n, e, 1);
t = randi(n, e, 1);
G = graph(s, t, [], n);

G = graph(true(n)); % Self-loops are possible
%G = graph(true(n), 'omitselfloops'); % Alternative without self-loops
p = randperm(numedges(G), e);
G = graph(G.Edges(p, :));
pg_ranks = centrality(G,'pagerank')

plot(G)