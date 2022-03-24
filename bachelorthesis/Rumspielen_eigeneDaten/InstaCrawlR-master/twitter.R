v = read.csv("/Users/tanjazast/Documents/tanjalein__insta.csv")
print(v)
names(v)

# read edges
e = read.csv("/Users/tanjazast/Documents/edge_tanjalein__.csv")
print(e)
names(e)

net = network(e, directed = TRUE)

# party affiliation
x = data.frame(Instagram = network.vertex.names(net))
x = merge(x, v, by = "Instagram_Name", sort = FALSE)$Groupe
net %v% "party" = as.character(x)

# color palette
y = RColorBrewer::brewer.pal(9, "Set1")[ c(3, 1, 9, 6, 8, 5, 2) ]
names(y) = levels(x)

# network plot
plot(net, color = "party", palette = y, alpha = 0.75, size = 4, edge.alpha = 0.5)

