install.packages("GGally")
library(GGally)
install.packages("igraph", repos='http://cran.us.r-project.org')
library("igraph")
install.packages('sna') 

friends_list <- read.csv("/Users/tanjazast/Desktop/Bachelorarbeit/CSV/Combined6.csv")
class(friends_list)
friends_list


ggnet2(friends_list, edge.arrow.size=.2, edge.color="orange",
     color="blue", edge.label=friends_list$name, label.color="black")

