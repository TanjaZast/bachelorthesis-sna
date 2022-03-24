
library(stringr)

#Import Table and Extract names
text <- list()
nametemp <- list()
names <- data.frame()
data <- read.csv("tanjalein__insta.csv", sep = ";")
data <- as.matrix(data[-1])

maxrows <- nrow(data)
for(i in 2:maxrows){
    text[i] <- as.character(data[2,11])
    nametemp <- str_extract_all(text[i], "#\\S+", TRUE)
    
    if(ncol(nametemp) != 0){
        for(j in 2:ncol(nametemp)){
            names[i,j] <- nametemp[2,j]
        }  
    }
} 

#Save Hashtags as csv for Excel
write.csv(names, "ht_unsort_follower.csv", fileEncoding = "UTF-8")
df_names <- as.data.frame(table(unlist(names)))
write.csv(df_names, "ht_sort_follower.csv", fileEncoding = "UTF-8")

