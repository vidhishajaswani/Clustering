library(mclust)
library(qgraph)


#read the data
df <- read.csv("vjaswan.csv")

#train model for k=6
gmm<-Mclust(df,G=6)

#see the summary
summary(gmm)

#plots for gmm
plot(gmm)

#plotting the distance graph
dist_m <- as.matrix(dist(df[1:20],method='euclidean'))
dist_mi <- 1/dist_m 
jpeg('example_forcedraw.jpg', width=1000, height=1000, unit='px')
qgraph(dist_mi, layout='spring', vsize=3)
dev.off()








