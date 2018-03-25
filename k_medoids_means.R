set.seed(123)
x <- rnorm(24, mean = rep(1:3, each=4), sd = 0.2)
y <- rnorm(24, mean=rep(c(1,2,1),each=4),sd=0.2)
data <- data.frame(x,y)

plot(x,y,col="blue",pch=19,cex=1)
text(x+0.05,y+0.05, labels = as.character(1:24))

#kmeans
km <- kmeans(data,centers = 3)
km
#variance within clusters
km$withinss
#clusters
km$cluster
plot(x,y,col=km$cluster, pch=19,cex=1)
points(km$centers,col = 1:3,pch=4,cex=3,lwd=2)

########################################

library(cluster)
k_medoids <- pam(x=data,k=3)
k_medoids
plot(x,y,col=k_medoids$clustering, pch=19,cex=1)
points(k_medoids$medoids, col=1:3,pch=4,cex=3,lwd=2)
