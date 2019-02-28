# -*- coding: utf-8 -*-
"""
iDBSCAN: Iterative Density-Based Spatial Clustering of Applications with Noise
"""

import numpy as np
from sklearn.cluster import DBSCAN

def idbscan(X, iterative = 4):
    """
    Returns
    -------
    core_samples : array [n_core_samples]
        Indices of core samples.
    labels : array [n_samples]
        Cluster labels for each point.  Noisy samples are given the label -1.
    tag : array [n_samples]
        tag for each point.
            - Noisy samples are given the label 'n'
            - Long tracks samples are given the label 'l'
            - Medium tracks are given the label 'm'
            - Small tracks are given the label 's'

    """

    ## - - - - -
    Index              = np.arange(0,np.shape(X)[0],dtype=int)
    Fcluster           = (-1)+np.zeros(np.shape(X)[0],dtype=int)
    Flabel             = np.empty(np.shape(X)[0],dtype=str)
    Flabel[:]          = 'n'
    auxClu             = -1
    # - - - - - -
    vector_eps         = [2.26, 3.5, 2.8, 6]
    vector_min_samples = [2, 30, 6, 2]
    auxIti             = - 1
    ## - - - - -

    if iterative >= 0:

        auxIti += 1
        db      = DBSCAN(eps=vector_eps[auxIti], min_samples=vector_min_samples[auxIti]).fit(X)
        labels  = db.labels_
        indgood = db.labels_ != -1

        ## ----- Salve the clusters and labels
        Fcluster[db.labels_ == -1] = -1
        Flabel[db.labels_   == -1] = 'n' # 'n' = noise points

        if iterative == 0:
            ## ----- Salve the clusters and labels
            Fcluster        = labels
            Flabel[indgood] = 's' # 'n' = noise points

    if iterative >= 1:

        Xnew      = X[indgood,:]
        indicenew = np.where(indgood == True)[0]

        auxIti    += 1
        db         = DBSCAN(eps=vector_eps[auxIti], min_samples=vector_min_samples[auxIti]).fit(Xnew)
        labels     = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # Find the Long clusters

        clusters = [Xnew[labels == i] for i in range(n_clusters_)]

        lenClu = np.zeros(n_clusters_,)
        for i in range(0,n_clusters_):
            lenClu[i] = np.size(clusters[i])
        clusterI = (np.where(lenClu > 900))[0]

        if iterative == 1 or iterative == 4 or iterative == 12: # To salve ONLY the Long Clusters or 4 to all
            ## ----- Salve the clusters and labels
            for i in clusterI:
                auxClu+=1
                indice = Index[indicenew[labels == i]]
                Fcluster[indice] = auxClu
                Flabel[indice] = 'l' # 'l' = Long tracks

    if iterative >= 2:

        indgood2 = ~np.in1d(db.labels_, clusterI)
        Xnew2 = Xnew[indgood2,:]
        indicenew2 = np.where(indgood2 == True)[0]

        auxIti+=1
        db = DBSCAN(eps=vector_eps[auxIti], min_samples=vector_min_samples[auxIti]).fit(Xnew2)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        clusters = [Xnew2[labels == i] for i in range(n_clusters_)]

        lenClu = np.zeros(n_clusters_,)
        for i in range(0,n_clusters_):
            lenClu[i] = np.size(clusters[i])
        clusterI = (np.where(lenClu > 150))[0]

        if iterative == 2 or iterative == 4 or iterative == 12: # To salve ONLY the Medium Clusters or 4 to all
            ## ----- Salve the clusters and labels
            for i in clusterI:
                auxClu+=1
                indice = Index[indicenew[indicenew2[labels == i]]]
                Fcluster[indice] = auxClu
                Flabel[indice] = 'm' # 'c' = Curly tracks


    if iterative >= 3:

        indgood3 = ~np.in1d(db.labels_, clusterI)
        Xnew3 = Xnew2[indgood3,:]        
        indicenew3 = np.where(indgood3 == True)[0]

        auxIti+=1
        db = DBSCAN(eps=vector_eps[auxIti], min_samples=vector_min_samples[auxIti]).fit(Xnew3)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        clusters = [Xnew3[labels == i] for i in range(n_clusters_)]
        
        if iterative == 3 or iterative == 4: # To salve ONLY the Small Clusters or 4 to all
            ## ----- Salve the clusters and labels
            for j in range(0,n_clusters_):
                auxClu+=1
                indice = Index[indicenew[indicenew2[indicenew3[labels == j]]]]
                Fcluster[indice] = auxClu
                Flabel[indice] = 's' # 'c' = others tracks


    return Fcluster, np.where(Fcluster != -1)[0], Flabel


class iDBSCAN:
    
    def __init__(self, iterative = 3):
        self.iterative = iterative

    def fit(self, X):
        
        clust = idbscan(X, self.iterative)
        self.labels_, self.core_sample_indices_, self.tag_  = clust
        
        return self