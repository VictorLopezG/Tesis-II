from numpy import *
from scipy import spatial


# Euclidean distance method
def dist_euclidean(vec_0, vec_1):
    return linalg.norm(vec_0 - vec_1)


# Euclidean distance method
def dist_hamming(vec_0, vec_1):
    hd = spatial.distance.hamming(array(vec_0).flatten(), array(vec_1).flatten())
    return hd


# Random k centroids from dataset
def rand_cent(template, k):
    centroids = zeros((k, shape(template)[1]))
    n = shape(template)[0]
    for i in range(k):
        centroids[i] = asarray(template[random.randint(0, n)][:])
    return centroids


def irand_cent(template, k):
    centroids = []
    n = shape(template)[1]
    for j in range(k):
        centr = random.randint(0, k, size=(n,))
        centroids.append(centr)
    return asarray(centroids)


# k-Means clustering method
def kmeans(dataset, k, cod='dct'):
    m = shape(dataset)[0]
    #Funcion de distancia
    cluster_assment = matrix(zeros((m, 2), dtype=int))
    if cod == 'gabor':
        dist = dist_hamming
        create_cent = irand_cent
    else:
        dist = dist_euclidean
        create_cent = rand_cent
    centroids = create_cent(dataset, k)
    
    cluster_changed = True
    while cluster_changed:
        #Calcular los centroides
        cluster_changed = False
        for i in range(m):  # cada plantilla
            min_dist = inf
            min_index = -2
            for j in range(k):  # 2 centroides
                dist_cent = dist(centroids[j][:], dataset[i][:])
                if dist_cent < min_dist:
                    min_dist = dist_cent
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            cluster_assment[i, :] = min_index, min_dist ** 2
            
        for i in range(k):
            nonzero_pos = nonzero((cluster_assment[:, 0].A == i))[0]
            pts_inclust = asarray(zeros((shape(nonzero_pos)[0], shape(dataset)[1]), dtype=int))
            icount = 0
            for dataAssment in nonzero_pos:
                pts_inclust[icount, :] = dataset[dataAssment, :]
                icount = icount + 1
            if pts_inclust.any():
                leng = len(centroids[i])
                for h in range(leng):
                    if cod == 'gabor':
                        centroids[i][h] = round(mean(pts_inclust[:, h]))
                    else:
                        centroids[i][h] = mean(pts_inclust[:, h])
    return centroids, cluster_assment


# Bisecting k-Means clustering method
'''
def bisecting_kmeans(dataset, limit_sse, cod, n_iterations=10):
    m, n = shape(dataset)
    centroid_list = []
    cluster_info = asarray(zeros((m, 2), dtype=int))  # decimal -> Int
    min_sse = inf

    for i in range(n_iterations):
        centroid, cluster_assment = kmeans(dataset, 2, cod)
        sse = sum(cluster_assment, axis=0)[0, 1]
        if sse < min_sse:
            min_sse = sse
            temp_centroid = centroid
            temp_cluster = cluster_assment
    centroid_list.append(temp_centroid[0][:].tolist())  # asarray?
    centroid_list.append(temp_centroid[1][:].tolist())
    cluster_info = temp_cluster
    max_sse = limit_sse + 1

    while max_sse > limit_sse:
        max_index = -2
        max_sse = -1
        for j in range(len(centroid_list)):
            sse = sum(cluster_info[nonzero(cluster_info[:, 0] == j)[0]])
            if sse > max_sse:
                max_index = j
                max_sse = sse

        min_sse = inf
        for k in range(n_iterations):
            points_incluster = asarray(dataset[nonzero(cluster_info[:, 0] == max_index)[0]])
            centroid, cluster_assment = kmeans(points_incluster, 2, cod)
            sse = sum(cluster_assment, axis=0)[0, 1]
            if sse < min_sse:
                min_sse = sse
                temp_centroid = centroid
                temp_cluster = cluster_assment
        temp_cluster[nonzero(temp_cluster[:, 0] == 1)[0], 0] = len(centroid_list)
        temp_cluster[nonzero(temp_cluster[:, 0] == 0)[0], 0] = max_index
        cluster_info[nonzero(cluster_info[:, 0] == max_index)[0], :] = temp_cluster
        centroid_list[max_index] = temp_centroid[0][:].tolist()
        centroid_list.append(temp_centroid[1][:].tolist())
    return centroid_list, cluster_info
    '''


def bisecting_kmeans(dataset, cod, n_clusters=3, n_iterations=10):
    m, n = shape(dataset)
    centroid_list = []
    cluster_info = asarray(zeros((m, 2), dtype=int))  # decimal -> Int
    min_sse = inf

    for i in range(n_iterations):
        centroid, cluster_assment = kmeans(dataset, 2, cod)
        sse = sum(cluster_assment, axis=0)[0, 1]
        if sse < min_sse:
            min_sse = sse
            temp_centroid = centroid
            temp_cluster = cluster_assment
    centroid_list.append(temp_centroid[0][:].tolist())  # asarray?
    centroid_list.append(temp_centroid[1][:].tolist())
    cluster_info = temp_cluster

    while n_clusters > len(centroid_list):
        max_index = -2
        max_sse = -1
        for j in range(len(centroid_list)):
            sse = sum(cluster_info[nonzero(cluster_info[:, 0] == j)[0]])
            if sse > max_sse:
                max_index = j
                max_sse = sse

        min_sse = inf
        for k in range(n_iterations):
            points_incluster = asarray(dataset[nonzero(cluster_info[:, 0] == max_index)[0]])
            centroid, cluster_assment = kmeans(points_incluster, 2, cod)
            sse = sum(cluster_assment, axis=0)[0, 1]
            if sse < min_sse:
                min_sse = sse
                temp_centroid = centroid
                temp_cluster = cluster_assment
        temp_cluster[nonzero(temp_cluster[:, 0] == 1)[0], 0] = len(centroid_list)
        temp_cluster[nonzero(temp_cluster[:, 0] == 0)[0], 0] = max_index
        cluster_info[nonzero(cluster_info[:, 0] == max_index)[0], :] = temp_cluster
        centroid_list[max_index] = temp_centroid[0][:].tolist()
        centroid_list.append(temp_centroid[1][:].tolist())
    return centroid_list, cluster_info
