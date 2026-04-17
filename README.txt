C:\Jack\Extracted NIST\FragPath
    Location of folders that contain the .MOL files (.MOL)
    Location of folders that contain windows installer files of spectra (.MSP)

Plan is as follows:

1. build SQL DB to organize .mol files and .MSP spectra DONE
2. use Morgan fingerprints to generate embeddings for the .mol files and place in a matrix 
3. cluster matrix (HDBSCAN) and find strongest cluster to use as our dataest
    umap to reduce dimensionality
    vary min_cluster_size and see what works best
    cluster metrics:
        Denisty-Based Clustering Validation (DBCV) - primary metric (maximize) specific to density based clustering methods
        Silhouette score - how similar each point is to its own cluster vs neightbors (maximize)
        Noise Fraction - fraction of points passed in that are classified as noise (minimize, <30% ideally)
        Cluster Persistance - stability across density thresholds (maximize)
        Adjusted Rand Index (ARI) - how well cluster assignment agree with CAS no groupings (-1 bad 1 perfect), sensitive to number of clusters
        Adjusted Manual Information (AMI) - same as ARI but normalizes for cluster size distribution

test on clusters 0,1, and 5 

we can add 2 as well just to see what happens
