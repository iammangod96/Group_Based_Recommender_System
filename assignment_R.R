rm(list = ls())

#notes
#As feedback types,
#we considered: whether a user tagged an item or not; and
#the number of times the user has visited a particular item.

#loading libraries


#loading datasets
artists <- read.delim("G:/BITS/4-2/Information retrieval/my_assignment/last_fm_dataset/artists.dat")
tags <- read.delim("G:/BITS/4-2/Information retrieval/my_assignment/last_fm_dataset/tags.dat")
user_artists <- read.delim("G:/BITS/4-2/Information retrieval/my_assignment/last_fm_dataset/user_artists.dat")
user_friends <- read.delim("G:/BITS/4-2/Information retrieval/my_assignment/last_fm_dataset/user_friends.dat")
user_taggedartists <- read.delim("G:/BITS/4-2/Information retrieval/my_assignment/last_fm_dataset/user_taggedartists.dat")
user_taggedartists_ts <- read.delim("G:/BITS/4-2/Information retrieval/my_assignment/last_fm_dataset/user_taggedartists-timestamps.dat")

#create matrix
m <- acast(user_artists, userID~artistID, value.var="weight")
m[is.na(m)] <- 0
head(m,1)
