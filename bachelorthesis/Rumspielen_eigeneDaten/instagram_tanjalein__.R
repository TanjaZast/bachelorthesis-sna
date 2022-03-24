install.packages("igraph", repos='http://cran.us.r-project.org')
library("igraph")

friends_list <- read.csv("/Users/tanjazast/Desktop/Bachelorarbeit/CSV/Combined6.csv")
class(friends_list)
friends_list

E(friends_list)
V(friends_list)
E(friends_list)$data.name
V(friends_list)$nameMee

ggnet2(friends_list, edge.arrow.size=.4,vertex.label=NA)
