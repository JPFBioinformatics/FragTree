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

After clustering we will train model on three clusters - 2,5,25 for 3 views.  cluster 2 is large (416) with bad SSE but otherwise good scores, clusetr 5 is small (86) with 
really good scores across the board, and cluster 25 is inbetween (163) with scores generally better than 5 and very close to 2.  We can also train on the full set and
measure how good it did using something like top 10/5/3/1 recall where recall is based on cos similarity between generated spectra and true spectra (10 most simliar for
top-10 recall, 5 for top-5 etc...)

