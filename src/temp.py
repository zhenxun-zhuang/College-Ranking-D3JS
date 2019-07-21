import os
import numpy as np
from sklearn import preprocessing
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def pca_dr(data):
    data = preprocessing.scale(data) #standardization
    pca = PCA().fit(data)
    
    eig = pca.explained_variance_
    np.savetxt("pca_eig.csv", eig, delimiter=",", fmt='%.4f')
    
    two_comp = pca.components_[0:2, :].transpose()
    data_pca = np.dot(data, two_comp)
    np.savetxt("pca_2_comp.csv", data_pca, delimiter=",", fmt = "%.4f")
    
    intri_dim = np.where(eig>1)[0] #intrinsic dimensions
    loadings = pca.components_[intri_dim, :].transpose()    
    significances = np.sum(np.square(loadings),axis=1)
    np.savetxt("loadings.csv", np.vstack((loadings.transpose(), significances)).transpose(), delimiter=",", fmt = "%.4f")
    
    sig_vari = np.argsort(significances)[::-1][:3]
    data_vari = data[:, sig_vari]
    np.savetxt("three_attri.csv", data_vari, delimiter=",", fmt = "%.4f")

    np.savetxt("attri_name.csv", sig_vari, delimiter=",", fmt="%s")

    return


def mds_dr(data):
    data = preprocessing.scale(data) #standardization
    
    #Euclidean metric
    mds = MDS(n_components=2, metric=True, n_init=4, max_iter=300, eps=0.001, dissimilarity='euclidean').fit(data)
    data_mds_euc = mds.embedding_
    np.savetxt("mds_euc.csv", data_mds_euc, delimiter=",", fmt = "%.4f")
    
    #Pearson correlation metric
    dist_mtx = abs(1-np.corrcoef(data, rowvar=True))
    mds = MDS(n_components=2, metric=True, n_init=4, max_iter=300, eps=0.001, dissimilarity='precomputed').fit(dist_mtx)
    data_mds_corr = mds.embedding_
    np.savetxt("mds_corr.csv", data_mds_corr, delimiter=",", fmt = "%.4f")
	
    #Cosine similarity metric
    dist_cos = 1-cosine_similarity(data);
    mds = MDS(n_components=2, metric=True, n_init=4, max_iter=300, eps=0.001, dissimilarity='precomputed').fit(dist_cos)
    data_mds_cos = mds.embedding_
    np.savetxt("mds_cos.csv", data_mds_cos, delimiter=",", fmt = "%.4f")    
    
    return

def comp(data):
    data = preprocessing.scale(data) #standardization
    
    n_neighbors = 10
    n_components = 2
 
    methods = ['standard', 'modified']
    labels = ['std', 'modified']    
    for i, method in enumerate(methods):    
        lle = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                            eigen_solver='auto',
                                            method=method).fit_transform(data)
        np.savetxt("lle_"+labels[i]+".csv", lle, delimiter=",", fmt = "%.4f")
        
    iso = manifold.Isomap(n_neighbors, n_components).fit_transform(data)
    np.savetxt("iso.csv", iso, delimiter=",", fmt = "%.4f")
    
    se = manifold.SpectralEmbedding(n_components=n_components,
                                    n_neighbors=n_neighbors).fit_transform(data)
    np.savetxt("se.csv", se, delimiter=",", fmt = "%.4f")

    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0).fit_transform(data)
    np.savetxt("tsne.csv", tsne, delimiter=",", fmt = "%.4f")

def ufun(year):
    main_folder = "C:\Study\SUNY Stony Brook\Class\CSE564\project\code\static\data"    
    folder = os.path.join(main_folder, str(year))
    os.chdir(folder)

    df = pd.read_csv(str(year)+'_data.csv', sep=',',header=None)
    data = df.values

    #pca_dr(data)
    mds_dr(data)
    
    #comp(data)


for year in range(2011,2017):
    ufun(year)