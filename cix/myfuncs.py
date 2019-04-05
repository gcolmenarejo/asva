import chemfp
from chemfp import search
import pandas as pd
import time
import sys
import subprocess as sp
from rdkit import Chem
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import kde
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import pylab
import random
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold as ms
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools as pt
from rdkit.Chem import Descriptors
from mpl_toolkits.axes_grid.inset_locator import inset_axes


### The results of the Taylor-Butina clustering
class ClusterResults(object):
    def __init__(self, true_singletons, false_singletons, clusters):
        self.true_singletons = true_singletons
        self.false_singletons = false_singletons
        self.clusters = clusters



### The clustering implementation
def taylor_butina_cluster(similarity_table):
    # Sort the results so that fingerprints with more hits come
    # first. This is more likely to be a cluster centroid. Break ties
    # arbitrarily by the fingerprint id; since fingerprints are
    # ordered by the number of bits this likely makes larger
    # structures appear first.:

    # Reorder so the centroid with the most hits comes first.  (That's why I do
    # a reverse search.)  Ignore the arbitrariness of breaking ties by
    # fingerprint index

    centroid_table = sorted(((len(indices), i, indices)
                                 for (i,indices) in enumerate(similarity_table.iter_indices())),
                            reverse=True)

    # Apply the leader algorithm to determine the cluster centroids
    # and the singletons:

    # Determine the true/false singletons and the clusters
    true_singletons = []
    false_singletons = []
    clusters = []

    seen = set()
    for (size, fp_idx, members) in centroid_table:
        if fp_idx in seen:
            # Can't use a centroid which is already assigned
            continue
        seen.add(fp_idx)

        # Figure out which ones haven't yet been assigned
        unassigned = set(members) - seen

        if not unassigned:
            false_singletons.append(fp_idx)
            continue

        # this is a new cluster
        clusters.append((fp_idx, unassigned))
        seen.update(unassigned)

    # Return the results:
    return ClusterResults(true_singletons, false_singletons, clusters)



### Calculate distance matrix for hierarchical clustering
def distance_matrix(arena):
    n = len(arena)

    # Start off a similarity matrix with 1.0s along the diagonal
    similarities = np.identity(n, "d")

    ## Compute the full similarity matrix.
    # The implementation computes the upper-triangle then copies
    # the upper-triangle into lower-triangle. It does not include
    # terms for the diagonal.
    results = search.threshold_tanimoto_search_symmetric(arena, threshold=0.0)

    # Copy the results into the NumPy array.
    for row_index, row in enumerate(results.iter_indices_and_scores()):
        for target_index, target_score in row:
            similarities[row_index, target_index] = target_score

    # Return the distance matrix using the similarity matrix
    return 1.0 - similarities



### Create a smis list from a smiles file (just smiles)
def smif2smis(name):
    smidf = pd.read_csv(name, delim_whitespace = True, names = ['smiles'], header = None)
    return list(smidf['smiles'])



### Find the correct smiles in a list of smiles
def corrsmis(smis):
    n = len(smis)
    corr_smi_yn = [x != None for x in [Chem.MolFromSmiles(s) for s in smis]]
    ncorr = sum(corr_smi_yn)
    smis = [smi for i, smi in enumerate(smis) if corr_smi_yn[i] == True]
    wrongsmis = [smi for i, smi in enumerate(smis) if corr_smi_yn[i] == False]
    return ncorr, n, smis, wrongsmis



### Create a dataframe of smiles, id from smiles list
def smis2smidf(smis):
    return pd.DataFrame({'smiles': smis, 'id': ['s' + str(x) for x in range(1, len(smis)+1)]}, columns = ['smiles','id'])



### Create a dataframe of smiles, id from smiles file
def smisf2smidf(smisf, noid = True, random = False, seed = 1234):
    
    if noid:
        smidf = pd.read_csv(smisf, delim_whitespace = True, names = ['smiles'], header = None)
    else:
        smidf = pd.read_csv(smisf, delim_whitespace = True, names = ['smiles','id'], header = None)
        
    if random == True:
        smidf = smidf.sample(frac=1, random_state = seed)
    return smidf



### Create arena from smiles df
def smidf2arena(smidf, reorder = True):

    # Write df of smiles, id
    smidf.to_csv('smidf.smi', header = False, sep = ' ', index = False)
    
    # Generate fps file
    sp.call(['rdkit2fps', './smidf.smi', '-o', 'smidf.fps'])
    
    ## Load the FPs into an arena
    try:
        arena = chemfp.load_fingerprints('./smidf.fps', reorder = reorder)
    except IOError as err:
        sys.stderr.write("Cannot open fingerprint file: %s" % (err,))
        raise SystemExit(2)
    
    # Remove files
    sp.call(['rm', './smidf.smi', './smidf.fps'])
    
    # Return arena
    return arena



### Create an fps file from a smiles df
def smidf2fps(smidf, name):
    # Write df of smiles, id
    smidf.to_csv('smidf.smi', header = False, sep = ' ', index = False)

    # Generate fps file
    sp.call(['rdkit2fps', './smidf.smi', '-o', name + ".fps"])	

    # Remove files
    sp.call(['rm', './smidf.smi'])



### Remove fps
def remfps(name):
    sp.call(['rm', './' + name + '.fps'])



### Cluster from smiles df
def clusmidf(smidf, th = 0.8, method = 'butina', arena = None):
    
    if method != 'butina' and method != 'cl':
        print('Please select butina or cl')
        return None
        
    # Init time counter
    start = time.time()
    
    # Get the arena
    if arena is None:
        arena = smidf2arena(smidf)

    # Do the clustering
    if method == 'butina':
        # Generate the similarity table
        similarity_table = search.threshold_tanimoto_search_symmetric(arena, threshold = th)
    
        # Cluster the data
        clus_res = taylor_butina_cluster(similarity_table)
        
        # Output
        out = []
        # We need to re-sort the clusters as the creation of them does not generate a monotonously decreasing list
        cs_sorted = sorted([(len(c[1]), c[1], c[0]) for c in clus_res.clusters], reverse = True)
        for i in range(len(cs_sorted)):
            cl = []
            c = cs_sorted[i]
            cl.append(arena.ids[c[2]]) # Retrieve the arenaid of the centroid and add to the cluster
            cl.extend([arena.ids[x] for x in c[1]]) # Retrieve the arenaid of the neighbors and add to cluster
            out.append(cl)
        for i in range(len(clus_res.false_singletons)):
            cl = [arena.ids[clus_res.false_singletons[i]]]
            out.append(cl)
        for i in range(len(clus_res.true_singletons)):
            cl = [arena.ids[clus_res.true_singletons[i]]]
            out.append(cl)
        
    elif method == 'cl':
        # Generate the condensed distance table
        distances  = ssd.squareform(distance_matrix(arena))
        
        # Cluster the data
        clus_res = fcluster(linkage(distances, method='complete'), th, 'distance')
        
        # Ouptut
        aids = arena.ids
        out = []
        for i in np.unique(clus_res):
            cl = [aids[i] for i in list(np.where(clus_res == i)[0])]
            out.append(cl)
        out = [x[2] for x in sorted([(len(x), i, x) for (i, x) in enumerate(out)], reverse = True)]


    # End time count and report
    end = time.time()
    elapsed_time = end - start
    print('Clustering time: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # Return cluster results
    return out
    
    

### Draw a set of molecules from smiles list
def paintmols(smis, molsPerRow = 5, subImgSize=(150,150)):
    ms = [Chem.MolFromSmiles(s) for s in smis]
    return Draw.MolsToGridImage(ms,molsPerRow=molsPerRow,subImgSize=subImgSize)



### Generate framework for a SMILES, handling for errors
def framecheck(s):
    try:
        return Chem.MolToSmiles(ms.GetScaffoldForMol(Chem.MolFromSmiles(s)))
    except:
        pass



### Generate generic framework for a SMILES, handling for errors
def gframecheck(s):
    try:
        return Chem.MolToSmiles(ms.MakeScaffoldGeneric(Chem.MolFromSmiles(s)))
    except:
        pass



### Diversity analysis
def divan(smidf, summ = False, OnlyBu = False, arena = None):
    
    start = time.time()
    
    # Get the arena
    if arena is None:
        arena = smidf2arena(smidf)
    
    # Cluster by butina and cl
    clr_bu = clusmidf(smidf, arena = arena)
    if(not OnlyBu):
        clr_cl = clusmidf(smidf, method = 'cl', th = 0.55, arena = arena)
    
    # Count the number of clusters in each method
    ncl_bu = len(clr_bu)
    if(not OnlyBu):
        ncl_cl = len(clr_cl)
    
    # Count Murko frameworks
    fras = list(set([framecheck(s) for s in smidf.smiles]))
    nfra = len(fras)
    
    # Count generic Murko frameworks
    frasg = list(set([gframecheck(s) for s in fras]))
    nfrag = len(frasg)
    
    # Calculate aggregated distance
    results = chemfp.search.knearest_tanimoto_search_symmetric(arena, k=1, threshold=0.0)
    ag_d = sum([1-hits.get_ids_and_scores()[0][1] for hits in results])
    
    end = time.time()
    eltime = end - start
    print('Diversity analysis time: ' + time.strftime("%H:%M:%S", time.gmtime(eltime)))
    
    if(summ):
        if(OnlyBu):
            return ncl_bu, nfra, nfrag, ag_d
        else:
            return ncl_bu, ncl_cl, nfra, nfrag, ag_d
    else:
        if(OnlyBu):
            return clr_bu, fras, frasg, ag_d
        else:
            return clr_bu, clr_cl, fras, frasg, ag_d



### Novelty analysis
def novan(smidfq, smidft, th = 0.7, arq = None, art = None):
    
    start = time.time()
    
    # Get the arenas
    if arq is None:
        arq = smidf2arena(smidfq)
    if art is None:
        art = smidf2arena(smidft)
    
    end = time.time()
    eltime = end - start
    print('Arenas creation time: ' + time.strftime("%H:%M:%S", time.gmtime(eltime)))
    
    # Find hits
    results = chemfp.search.threshold_tanimoto_search_arena(arq, art, threshold=th)
    
    # Generate list with new guys (no neighbors in target arena) and calculate its length
    news = []
    for query_id, query_hits in zip(arq.ids, results):
        if len(query_hits) == 0:
            news.append(query_id)
    
    
    # Generate list of frameworks for query and target
    fraq = [framecheck(s) for s in smidfq.smiles]
    fraq = list(np.unique(fraq))
    frat = [framecheck(s) for s in smidft.smiles]
    frat = list(np.unique(frat))
    
    newfraqs = [f for f in fraq if f not in frat]
    
    # Generate list of generic frameworks for query and target
    gfraq = [gframecheck(s) for s in fraq]
    gfraq = list(np.unique(gfraq))
    gfrat = [gframecheck(s) for s in frat]
    gfrat = list(np.unique(gfrat))
    
    newgfraqs = [f for f in gfraq if f not in gfrat]
    
    end = time.time()
    eltime = end - start
    print('Novelty analysis time: ' + time.strftime("%H:%M:%S", time.gmtime(eltime)))
    
    return news, fraq, newfraqs, gfraq, newgfraqs



### Plot clusters
def plotclus(d, xlab, ylab, xloglab, yloglab):

    ax1 = plt.axes()  # standard axes
    ax2 = plt.axes([0.45, 0.45, 0.4, 0.4])
    ax1.scatter(d.iloc[:,0], d.iloc[:,1], marker = '.', linewidth = 0)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(xloglab)
    ax2.set_ylabel(yloglab)
    ax2.scatter(d.iloc[:,0], d.iloc[:,1], marker = '.', linewidth = 0)



### Plot list of clusters
def plotmulticlus(cls, sizex, sizey):
    
    ncl = len(cls)
    
    fig, ax = plt.subplots(ncl, 2, figsize=[sizex, sizey], squeeze = False)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
    
    ni = 0
    for i in range(ncl):
        cl = cls[i]
            
        # Left column: cluster distribution
        a0 = ax[i][0]
        d = pd.DataFrame({'clid':range(1, len(cl)+1), 'n':map(len, cl)})
        a0.scatter(d.iloc[:,0], d.iloc[:,1], marker = '.', linewidth = 0)
        a0.set_xlabel("Cluster ID")
        a0.set_ylabel("# Elements")
        ia0 = inset_axes(a0, width=1.3, height=0.9)
        ia0.set_xscale("log")
        ia0.set_yscale("log")
        ia0.set_xlabel("log Cluster ID")
        ia0.set_ylabel("log #Elements")
        ia0.scatter(d.iloc[:,0], d.iloc[:,1], marker = '.', linewidth = 0)
            
        # Right column: neighbor distribution
        a1 = ax[i][1]
        d2 = pd.DataFrame({"clsize":d.n.groupby(d.n).unique(), "n":d.n.groupby(d.n).sum()})
        a1.scatter(d2.iloc[:,0], d2.iloc[:,1], marker = '.', linewidth = 0)
        a1.set_xlabel("Cluster Size")
        a1.set_ylabel("# Elements")
        ia1 = inset_axes(a1, width=1.3, height=0.9)
        ia1.set_xscale("log")
        ia1.set_yscale("log")
        ia1.set_xlabel("log Cluster Size")
        ia1.set_ylabel("log # Elements")
        ia1.scatter(d.iloc[:,0], d.iloc[:,1], marker = '.', linewidth = 0)
    return
    


### Paint property histogram
def painthis(smidf, prop):

    pt.AddMoleculeColumnToFrame(smidf,"smiles")
    smidf['pr'] = smidf['ROMol'].map('Descriptors.' + prop)
    del smidf["ROMol"]
    ax = smidf['pr'].hist(bins = 50)
    ax.set_xlabel(prop)



### Paint a bunch of histograms
def paintmultihist(prs, xlab, nrow, ncol, xtxt, ytxt, sizex, sizey, legx, legy, leg):
    
    mes = map(np.mean, prs)
    sds = map(np.std, prs)
    fig, ax = plt.subplots(nrow, ncol, figsize=[sizex, sizey], squeeze = False)
    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

    ni = 0

    for row in range(nrow):
        for col in range(ncol):
            if ni < len(prs):
                ax[row][col].hist(prs[ni], bins = 40)
                ax[row][col].set_xlabel(xlab)
                ax[row][col].text(xtxt,ytxt, "Mean=" + str(round(mes[ni],2)))
                ax[row][col].text(xtxt,ytxt-40, "SD=" + str(round(sds[ni],2)))
                ax[row][col].text(legx, legy, leg[ni])
                ni = ni+1

    plt.show()



### Whole diversity and novelty analysis for iterations
def wholean(it, name_train = "train", name_pref = "unc", th = 0.7):

    nit = len(it)
    df = pd.DataFrame(np.nan, index = range(1, nit+1),\
                      columns =\
                     ["# train","%corr inp","# un train","# clus inp","# fram inp","# gen fram inp",\
                     "# out","%corr out","# un out","# clus out","# fram out","# gen fram out",\
                     "% new str","% new fram","% new gen fram"])
    
    cls = [] # List with lists of clusters
    
    for i in range(len(it)):
    
        # Find corrects and unique in input and fill ntrain, nuntrain and pcorr inp
        smis = smif2smis('./' + name_train + str(it[i]) + '.smi')
        ncorr, n, smis, wrongsmis = corrsmis(smis)
        smis = list(set(smis))
        nuntrain = len(smis)
        smidft = smis2smidf(smis)
        del smis
        df["# train"].iloc[i] = n
        df["%corr inp"].iloc[i] = round(ncorr/float(n)*100,2)
        df["# un train"].iloc[i] = nuntrain
    
        # Find corrects and unique in output 
        smis = smif2smis('./' + name_pref +str(it[i]) + '.smi')
        ncorr, n, smis, wrongsmis = corrsmis(smis)
        smis = list(set(smis))
        nunout = len(smis)
        smidfq = smis2smidf(smis)
        del smis
        df["# out"].iloc[i] = n
        df["%corr out"].iloc[i] = round(ncorr/float(n)*100,2)
        df["# un out"].iloc[i] = nunout
        
        # Generate arenas
        art = smidf2arena(smidft)
        arq = smidf2arena(smidfq)
    
        # Diversity analysis of input and fill nclus inp, nfram inp, ngenfram inp
        clb, fs, fg = divan(smidft, OnlyBu = True, arena = art)
        df["# clus inp"].iloc[i] = len(clb)
        df["# fram inp"].iloc[i] = len(fs)
        df["# gen fram inp"].iloc[i] = len(fg)
        cls.append(clb)
    
        # Diversity analysis of output and fill nclus out, nfram out, ngenfram out
        clb, fs, fg = divan(smidfq, OnlyBu = True, arena = arq)
        df["# clus out"].iloc[i] = len(clb)
        df["# fram out"].iloc[i] = len(fs)
        df["# gen fram out"].iloc[i] = len(fg)
    
        # Novelty analysis
        news, fraq, newfraqs, gfraq, newgfraqs = novan(smidfq, smidft, th = th, arq = arq, art = art)
        df["% new str"].iloc[i] = round(100*len(news)/float(smidfq.shape[0]),2)
        df["% new fram"].iloc[i] = round(100*len(newfraqs)/float(len(fraq)),2)
        df["% new gen fram"].iloc[i] = round(100*len(newgfraqs)/float(len(gfraq)),2)
                            
    # Return dataframe with output
    return df, cls


### Whole diversity and novelty analysis for iterations, returning in the output also the clusters of the outputs
def wholean2(it, name_train = "train", name_pref = "unc", th = 0.7):

    nit = len(it)
    df = pd.DataFrame(np.nan, index = range(1, nit+1),\
                      columns =\
                     ["# train","%corr inp","# un train","# clus inp","# fram inp","# gen fram inp",\
                     "# out","%corr out","# un out","# clus out","# fram out","# gen fram out",\
                     "% new str","% new fram","% new gen fram"])
    
    clsi = [] # List with lists of clusters input
    clso = [] # List with lists of clusters output

    
    for i in range(len(it)):
    
        # Find corrects and unique in input and fill ntrain, nuntrain and pcorr inp
        smis = smif2smis('./' + name_train + str(it[i]) + '.smi')
        ncorr, n, smis, wrongsmis = corrsmis(smis)
        smis = list(set(smis))
        nuntrain = len(smis)
        smidft = smis2smidf(smis)
        del smis
        df["# train"].iloc[i] = n
        df["%corr inp"].iloc[i] = round(ncorr/float(n)*100,2)
        df["# un train"].iloc[i] = nuntrain
    
        # Find corrects and unique in output 
        smis = smif2smis('./' + name_pref +str(it[i]) + '.smi')
        ncorr, n, smis, wrongsmis = corrsmis(smis)
        smis = list(set(smis))
        nunout = len(smis)
        smidfq = smis2smidf(smis)
        del smis
        df["# out"].iloc[i] = n
        df["%corr out"].iloc[i] = round(ncorr/float(n)*100,2)
        df["# un out"].iloc[i] = nunout
        
        # Generate arenas
        art = smidf2arena(smidft)
        arq = smidf2arena(smidfq)
    
        # Diversity analysis of input and fill nclus inp, nfram inp, ngenfram inp
        clb, fs, fg = divan(smidft, OnlyBu = True, arena = art)
        df["# clus inp"].iloc[i] = len(clb)
        df["# fram inp"].iloc[i] = len(fs)
        df["# gen fram inp"].iloc[i] = len(fg)
        clsi.append(clb)
    
        # Diversity analysis of output and fill nclus out, nfram out, ngenfram out
        clb, fs, fg = divan(smidfq, OnlyBu = True, arena = arq)
        df["# clus out"].iloc[i] = len(clb)
        df["# fram out"].iloc[i] = len(fs)
        df["# gen fram out"].iloc[i] = len(fg)
        clso.append(clb)
    
        # Novelty analysis
        news, fraq, newfraqs, gfraq, newgfraqs = novan(smidfq, smidft, th = th, arq = arq, art = art)
        df["% new str"].iloc[i] = round(100*len(news)/float(smidfq.shape[0]),2)
        df["% new fram"].iloc[i] = round(100*len(newfraqs)/float(len(fraq)),2)
        df["% new gen fram"].iloc[i] = round(100*len(newgfraqs)/float(len(gfraq)),2)
                            
    # Return dataframe with output
    return df, clsi, clso




### Diversity sampler 0
def divsamp0(ar, th = 0.7, nlimit = 300000, seed = 1234):
    start = time.time()

    np.random.seed(seed)
    rnin = np.arange(len(ar))
    np.random.shuffle(rnin)

    neig = set()
    sel = []

    for i in range(len(ar)):
        if(len(sel) < nlimit):
            if i == 0:
                sel.append(rnin[i])
                fp = ar[rnin[i]][1]
                res = chemfp.search.threshold_tanimoto_search_fp(fp, ar, threshold= th)
                neig.update(res.get_indices())
            else:
                if rnin[i] not in neig:
                    sel.append(rnin[i])
                    fp = ar[rnin[i]][1]
                    res = chemfp.search.threshold_tanimoto_search_fp(fp, ar, threshold=th)
                    neig.update(res.get_indices())
            print "i=" + str(i) + "; nsel=" + str(len(sel)) + "; nneig=" + str(len(neig)) + "\r",

    end = time.time()
    eltime = end - start

    print "i=" + str(i) + "; nsel=" + str(len(sel)) + "; nneig=" + str(len(neig)) + "; Sampling time: " + time.strftime("%H:%M:%S", time.gmtime
(eltime))
    
    return sel



# Paint a bidimensional scatter/contour-density plot
def bidiplot(data, xlab, ylab, alpha = 1, s = 2, nbins = 20, d = False):

    if d is False:
        ax = plt.axes()  
        ax.scatter(data[:,0],data[:,1], alpha = alpha, s = 2)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
    else:
        ax = plt.axes() 
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        x, y = data.T
        nbins = 20
        k = kde.gaussian_kde(data.T)
        xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.afmhot_r)
        plt.contour(xi, yi, zi.reshape(xi.shape))
        plt.colorbar()
        plt.show()
