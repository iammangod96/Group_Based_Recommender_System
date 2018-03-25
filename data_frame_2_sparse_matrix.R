#loading libraries
library(stringi)
library(Matrix)
library(reshape2)

set.seed(123)

#user_item dataframe
df <- data.frame(item = stri_rand_strings(20,4))
df$user <- as.factor(1:nrow(df))
df$rating <- sample(1:10,nrow(df),T)

#create matrix
m <- acast(df, user~item, value.var="rating")
m[is.na(m)] <- 0
