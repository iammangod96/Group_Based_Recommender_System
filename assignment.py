import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import operator
import time

start_time = time.time()



#---------------------  kmedoids function ----------------------------
def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
    # return results
    return M,C

    
    
#---------------------  Data profiling ----------------------------

#load data
user_artists = pd.read_csv("G:/BITS/4-2/Information retrieval/my_assignment/last_fm_dataset/user_artists.csv")
user_friends = pd.read_csv("G:/BITS/4-2/Information retrieval/my_assignment/last_fm_dataset/user_friends.csv")


#data profiling
user_artists.describe()
#userID
unique_userID_arr = user_artists.userID.unique()
unique_userID_arr.sort()
unique_userID_arr.size #1892
for i in range(0,100):
    if(unique_userID_arr[i] != i+2):
        print("i:",i," / arr[i]:",unique_userID_arr[i])
#artistID
unique_artistID_arr = user_artists.artistID.unique()
unique_artistID_arr.sort()
unique_artistID_arr.size #17632

#17632 artists rangin from artist ID => 1 to 18745
#1892 users rangin from user ID => 2 to 2100



#---------------------  matrix computation ----------------------------

m = pd.pivot_table(user_artists, values='weight',index='userID',columns='artistID')
m = m.fillna(0)
m_index = m.index.values
m_columns = m.columns.values
m[8][9] #eureka m[column index][row index]

#---------------------  Clustering ----------------------------

#m = (m - m.min()) / (m.max() - m.min()) #normailizing m puts all in same cluster
D = pairwise_distances(m, metric='euclidean') #indexing from 0 to 1891
D_cosine = pairwise_distances(m,metric='cosine')
M , C= kMedoids(D, 4)
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, point_idx))

#testing clusters
#for i in range(C[0].size):
#    print(C[0][i])



#---------------------  fuctions to be used for userKNN ----------------------------

ua_list = user_artists.groupby('userID')['artistID'].apply(list) #indexing 2 to 2100
uf_list = user_friends.groupby('userID')['friendID'].apply(list) #indexing 2 to 2100

#ua_list.values[0] #userID starts at 2, so 0->2,1->3 and so on. #<---------- this is wrong

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def num_common(p, q): #put userID from 0 to 1891
    return len(intersection(ua_list.values[p],ua_list.values[q]))

#ua_list.values[0]
#print(num_common(0,2))

def euclideanDistance(user1, user2): #put userID from 0 to 1891
	return D[user1][user2]

#print(euclideanDistance(2,4))

def cosineDistance(user1, user2): #put userID from 0 to 1891
    return D_cosine[user1][user2]

def getClusterLabel(user): #put userID from 0 to 1891
    for i in range(len(C)):
        if(user in C[i]):
            return i
    return -1

#print(getClusterLabel(5))

def getUser(ix): #put userID from 0 to 1891
    return unique_userID_arr[ix]

#print(getUser(0))

def getArtist(ix):
    return unique_artistID_arr[ix]


num_users = m.shape[0]
num_artists = m.shape[1]
#print(num_artists)
#print(num_users)

def evaluate_common(arr1, arr2, n): #used in evaluation
    arr1_new = [] #only artists from the tuple array (artist + score)
    arr2_new = [] #only artists from the tuple array (artist + score)
    for i in range(0,n):
        arr1_new.append(arr1[i][0])
    for i in range(0,n):
        arr2_new.append(arr2[i][0])
    return intersection(arr1_new,arr2_new)

#---------------------  userKNN ----------------------------

def recommend_artists(user): #put userID from 0 to 1891
    cluster = getClusterLabel(user)
    cluster_size = C[cluster].size
    scores = [] #score for each artist predicted for that user
    for k in range(0,num_artists):
        s = 0
        num_people = 0
        for p in range(0,cluster_size):
            if(m[ getArtist(k) ][ getUser( C[cluster][p] ) ] > 0):
                if(user != C[cluster][p]):
                    n = num_common(user,C[cluster][p])
                    d = (n/(n+100))*cosineDistance(user,C[cluster][p])
                    if( getUser( C[cluster][p] ) in uf_list[ getUser(user) ] ):
                        s += 2*d #choosing this weight requires further research
                    else:
                        s+=d
                    num_people += 1
        if(num_people == 0):
            score=0
        else:
            score = s/num_people
        scores.append((getArtist(k),score))
    scores.sort(key=operator.itemgetter(1),reverse = True)
    return scores

def recommend_artists_print(scores):
    for i in range(0,50):
        print(scores[i][0]," ,score:",scores[i][1])
        

print ("Setup complete in ", time.time() - start_time, "time")
#takes around 20 seconds

#---------------------  recommendation ----------------------------

req_user = 73
start_time = time.time()
recommended_artists_arr = recommend_artists(req_user) 
recommend_artists_print(recommended_artists_arr)
print ("Recommendation complete in ", time.time() - start_time, "time")
#~10 secs - 1 items



#---------------------  Evaluation ----------------------------

#find decreasing sorted artist list of user actual
actual_arr = []
for i in range(0,num_artists):
    if(m[ getArtist(i) ][getUser(req_user)] > 0):
        actual_arr.append((getArtist(i),m[ getArtist(i) ][getUser(req_user)]))
actual_arr.sort(key=operator.itemgetter(1),reverse = True)
for i in range(0,50):
        print(actual_arr[i][0]," ,actual weight:",actual_arr[i][1])

print("Number of common:") #there's a problem here, the elements are tupple
print( evaluate_common(recommended_artists_arr,actual_arr,10) )

#---------------------  debugging ----------------------------
#
#if(-3):
#    print("manish")
#
##debugging
#cluster = getClusterLabel(72)
#cluster_size = C[cluster].size
#scores = [] #score for each artist predicted for that user
#for k in range(0,num_artists):
#    s = 0
#    num_people = 0
#    for p in range(0,cluster_size):
#        #if(m[C[cluster][p]][k] != 0):
#        s+=C[cluster][p];
#print("manish done")
#print(s)
#for i in range(0,10):
#    print(i)
#    
#print(m.iloc[7,7]) #another eureka
#llr = list(m.index)
#llc = list(m)
#m.columns
#m.rows
#print(user_artists.iloc[0,1]) #eureka
