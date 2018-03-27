import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import operator

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
 


user_artists = pd.read_csv("G:/BITS/4-2/Information retrieval/my_assignment/last_fm_dataset/user_artists.csv")
m = pd.pivot_table(user_artists, values='weight',index='userID',columns='artistID')
m = m.fillna(0)
D = pairwise_distances(m, metric='euclidean')
M , C= kMedoids(D, 4)
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, point_idx))

#testing clusters
#for i in range(C[0].size):
#    print(C[0][i])

ua_list = user_artists.groupby('userID')['artistID'].apply(list)

#ua_list.values[0] #userID starts at 2, so 0->2,1->3 and so on.

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def num_common(p, q): #put actual userID
    return len(intersection(ua_list.values[p-2],ua_list.values[q-2]))

#print(num_common(2,4))

def euclideanDistance(user1, user2): #put actual userID
	return D[user1 - 2][user2 - 2]

#print(euclideanDistance(2,4))

def getClusterLabel(user): #put actual userID
    for i in range(len(C)):
        if(user in C[i]):
            return i
    return -1

#print(getClusterLabel(5))

num_users = m.shape[0]
num_artists = m.shape[1]
#print(num_artists)

def recommend_artists(user):
    cluster = getClusterLabel(user)
    cluster_size = C[cluster].size
    scores = [] #score for each artist predicted for that user
    for k in range(0,num_artists):
        s = 0
        num_people = 0
        for p in range(0,cluster_size):
            if(m[C[cluster][p]][k] != 0):
                n = num_common(user,C[cluster][p])
                d = (n/(n+100))*euclideanDistance(user,C[cluster][p])
                s += d
                num_people += 1
        score = s/num_people
        scores.append((k,score))
    scores.sort(key=operator.itemgetter(1),reverse = True)
    #rec = []
    for i in range(0,5):
        print(scores[i][0])
        #rec.append(scores[i][0])
    
        
recommend_artists(3)
    