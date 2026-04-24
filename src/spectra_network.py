"""

Generates a weighted network based on pairwise cosine similarity between all 
spectra stored in the spectra.h5 file, on duplicate spectra (share casNO and 
formula) it avearges the spectra and treats it as a single entry.

Uses K-NN to ensure min_edges (default 10) coming from each spectra and adds  
additional edges between any spectra whose cos similarity is above add_edge_t 
(top 20% of cos similarities).  Weights of edges in the network will be 
cosine siilarity, but we can convert to 1-weight for any distance based 
algorithms.

Once data is loaded (_load_spectra()) the spectra are converted to 1001 degree
vectors where index is the m/z and the value is the

Workflow
--------
1. init 
    give path to medatada.db, spectra.h5, and structure.h5
2. run generate_network(n_neighbors, add_edge_t)
    takes full dataset of spectra from spectra.h5 and creates a KNN
    network where each spectra is a node with at least n_neighbors 
    edges (weight = cos simliarity) and additional edges added for each
    pariwise cos similarity > add_edge_t, edges labelled with casNO
3. run initailize_fingerprints()


"""
# region Imports

from pathlib import Path
from collections import defaultdict, namedtuple
from torch_geometric.data import Data
from torch_geometric.utils import degree
import h5py, sqlite3, torch, re, logging, csv
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# endregion

# region Data Structures

ConsensusSpectrum = namedtuple("ConsensusSpectrum", ["casNO", "h5ids", "vec"])           # [str,list(str),np.array]
DroppedSpectrum = namedtuple("DroppedSpectrum", ["casNO", "h5id", "n_origonal", "mean_sim"])    # [str,str,int,float]

# endregion

class SpectraNetwork:
    
    def __init__(self, name: str, 
                 metadata: Path, 
                 spectra: Path, 
                 structure: Path, 
                 rep_threshold: float,
                 device: str = "cpu",
                 data_dir = Path(__file__).resolve().parent.parent / "network"):

        self.name = name
        self.metadata_path = Path(metadata)
        self.spectra_path = Path(spectra)
        self.structure_path = Path(structure)
        self.rep_threshold = rep_threshold
        self.device = device
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.node_map = {}
        self.net_file = None
        log.info(f"Initialized {name} network")

    # Fully generates network and plots
    def generate_network(self, threshold: float = 0.5, n_neighbors: int = 10):
        """
        Takes the list of consensnus spectra information and builds
        a node_map dict of tensor_idx : (casNO, h5ids) where casNO is int and 
        h5ids is a list of the h5ids that were used to create the consensus
        spectra.  Consensus spectra (vec) are stored as data.x, the node feature
        matrix

        Params
        ------
        threshold                   min cosine similarity to add additional edges
        n_neighbors                 min edges for each node in network for knn
                                    network building
        """

        consensus_list = self.average_casno_replicates()

        # create a stacked tensor/matrix (x) of consensus vectors and node_map
        vecs = []
        for i,entry in enumerate(consensus_list):
            self.node_map[i] = (entry.casNO, entry.h5ids)
            vecs.append(torch.tensor(entry.vec, dtype=torch.float32))
        x = torch.stack(vecs)

        # create PyG Data graph container to manage network
        data = Data(x=x)
        n_nodes = x.shape[0]
        data.num_nodes = n_nodes

        # calcualte similarity matrix (cosine simliarity)
        sim_path = self._compute_sim_matrix(x)

        # calculate t_val by sampling similarity matrix
        sample_idx = np.random.choice(n_nodes, size=min(2000,n_nodes), replace=False)
        sample_rows = np.memmap(sim_path, dtype=np.float32, mode="r", shape=(n_nodes,n_nodes))[sample_idx]
        mask = np.ones(sample_rows.shape, dtype=bool)   # mask removes self-similarity
        for i,idx in enumerate(sample_idx):
            mask[i,idx] = False
        sample_vals = torch.tensor(sample_rows[mask].flatten(), dtype=torch.float32)
        sample_vals = sample_vals[torch.randperm(len(sample_vals))[:1_000_000]]
        self.t_val_hist("Pariwise Cosine Similarity Histogram", sample_vals, threshold)
        log.info("Built Pairwise Cosine Similarity Histogram")

        # build edges for network
        sources, targets, edge_weights = self._build_edges(sim_path,threshold,n_nodes,n_neighbors)

        # save edge indexes and weights to tensors, and calculate edge distances (1-cos similarity)
        edge_index = torch.stack([torch.cat([sources,targets]), torch.cat([targets,sources])])
        edge_attr = torch.cat([edge_weights, edge_weights])
        edge_dist = 1 - edge_attr

        # save to network Data object
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.edge_dist = edge_dist

        # plot degree histogram
        self.degree_hist(data)

        # save network
        net_file = self.data_dir / f"{self.name}_network.pt"
        torch.save(data, net_file)
        self.net_file = net_file
        log.info(f"Generated {self.name} network")

    # region SQLite3 helpers

    def connect(self):
        """
        Opens a connection to the SQLite database.

        Returns
        -------
        (conn, cursor)      open connection and cursor, or (None, None) on failure
        """
        try:
            conn   = sqlite3.connect(self.metadata_path)
            cursor = conn.cursor()
            return conn, cursor
        except sqlite3.Error as e:
            log.warning(f"Failed to connect to database at {self.metadata_path}\nError: {e}")
            return None, None

    def run_query(self, query, params=(), fetch=True):
        """
        Executes a SQL query and optionally returns results.

        Parameters
        ----------
        query               SQL string with ? as parameter placeholders
        params              tuple of values to substitute for placeholders
        fetch               if True, return rows for SELECT/PRAGMA queries

        Returns
        -------
        results             list of row tuples, or empty list
        """
        conn, cursor = self.connect()
        if not conn or not cursor:
            return []

        try:
            cursor.execute(query, params)
            command = query.strip().split()[0].upper()

            if command in {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP"}:
                conn.commit()
                log.info(f"Executed: {query}")
            elif fetch and command in {"SELECT", "PRAGMA"}:
                log.info(f"Executed: {query}")
                return cursor.fetchall()

            return []

        except sqlite3.Error as e:
            conn.rollback()
            log.warning(f"Query failed — {e}\nSQL: {query}")
            return []

        finally:
            conn.close()

    # endregion

    # region Data prep

    def average_casno_replicates(self):
        """
        takes the linked_spectra list from link_sp_cas and finds duplicate
        casNO, then drops spectra for those casNO that are different from the 
        rest based on a thershold and averages the rest to produce a consensus
        vector (averaged spectra)

        Returns
        -------
        consensus                   named tuple of casNO h5ids (list) and vec (consensus vector)
        """

        threshold = self.rep_threshold

        linked_spectra = self._link_sp_cas()
        groups = defaultdict(list)
        
        # build dict to group spectra by casno
        for id,cas,arr in linked_spectra:
            groups[cas].append((id,arr))
        log.info("Grouped spectra by casNO")

        # drop low similarity spectra and average the rest
        dropped= []
        consensus = []
        sp_count = 0

        for key,value in groups.items():
            if len(value) <= 1:
                raw = value[0][1]
                norm = np.linalg.norm(raw)
                if norm == 0:
                    log.warning(f"\nWarning: zero vector after averaging {key}, skipping")
                    continue
                consensus.append(ConsensusSpectrum(key, [value[0][0]], raw/norm))
                sp_count += len(value)
                continue
            n_origonal = len(value)
            sp_count += n_origonal

            ids, mat, sim_mat = self._rep_sim_mat(value)

            while mat.shape[0] > 1:
                upper = sim_mat[np.triu_indices(mat.shape[0], k=1)]
                if upper.min() >= threshold:
                    break

                n = mat.shape[0]
                row_means = ((sim_mat.sum(axis=1) - 1) / (n-1))
                worst = np.argmin(row_means)
                dropped.append(DroppedSpectrum(key,ids[worst],n_origonal,row_means[worst].item()))

                ids = [id for i,id in enumerate(ids) if i != worst]
                mat = np.delete(mat,worst,axis=0)
                sim_mat = mat @ mat.T
            
            # L2 normalize consensus spectrum and save 
            avg = np.mean(mat,axis=0)
            norm = np.linalg.norm(avg)
            if norm == 0:
                log.warning(f"\nWarning: zero vector after averaging {key}, skipping")
                continue
            avg = avg / norm
            consensus.append(ConsensusSpectrum(key,ids,avg))

        log.info(f"Generated Consensus Spectra\
                 \nTotal Queried: {sp_count}\
                 \nDropped: {len(dropped)}\
                 \nUnique casNO: {len(consensus)}")
        
        # save dropped list to csv
        if dropped:
            name = re.sub(r'[^\w]', '_', self.name)
            csv_file = self.data_dir / f"{name}_dropped_spectra.csv"
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["casNO", "H5ID", "n_origonal", "mean_sim"])
                for d in dropped:
                    writer.writerow([d.casNO, d.h5id, d.n_origonal, d.mean_sim])
            log.info(f"Wrote dropped replicates to {csv_file}")
        else:
            log.info(f"No dropped replicates for network {self.name}")

        # plot replicate histogram
        if groups:
            self.replicate_hist(f"{self.name} Replicate Histogram",groups)
        else:
            log.warning(f"No groups dictionary detected for {self.name}")

        return consensus

    def _load_mol(self):
        """
        Loads all mol files from the structures.h5 file

        Returns
        -------
        structures                  list of mol files 
        """

    def _load_spectra(self):
        """
        Loads all spectra to a list, removes any m/z < 100 or in the set
        of tms canonical ions 73, 147, 149

        Returns
        -------
            spectra             list of (key, spectrum) where spectrum is
                                a 1001 degree np array where the index is
                                the m/z value for the intensity stored in the
                                cell
        """
        spectra = []
        skipped = 0
        with h5py.File(self.spectra_path, "r") as f:
            for key in f.keys():
                spectrum = np.array(f[key])

                mz_col = spectrum[:,0]
                mask = (mz_col >= 100) & (mz_col <= 1000) & ~np.isin(mz_col, [73,147,149])

                spectrum = spectrum[mask]

                if spectrum.shape[0] == 0:
                    log.warning(f"Warning: {key} has no peaks after flitering, skipping")
                    skipped += 1
                    continue

                sp_vector = np.zeros(1001)
                for mz,intensity in spectrum:
                    sp_vector[int(mz)] = intensity

                spectra.append((key,sp_vector))

        log.info(f"Loaded {len(spectra)} spectra, skipped {skipped} entries")

        return spectra
    
    def _load_casNO(self):
        """
        loads casNO, h5ID from spectra table of metadata.db

        Returns
        -------
            id_to_cas           dict of h5ID : casNO
        """
        rows = self.run_query("SELECT h5ID, casNO FROM spectra")
        id_to_cas = {h5ID:cas for h5ID,cas in rows}
        
        log.info(f"Loaded {len(id_to_cas)} casNO from {self.metadata_path}")
        return id_to_cas
    
    def _link_sp_cas(self):
        """
        Genretes a new list of h5ID, casNO, spectra joining spectra and 
        id_to_cas on h5ID

        Returns
        -------
        linked_spectra          list of (h5ID, casNO, spectra)
        """
        linked_spectra = []
        spectra = self._load_spectra()
        id_to_cas = self._load_casNO()

        for h5id,value in spectra:
            casNO = id_to_cas.get(h5id)
            linked_spectra.append((h5id, casNO, value))

        log.info("Linnked Spectra")
        return linked_spectra

    def _rep_sim_mat(self, value: list):
        """
        Calculates cosine simimlarity matrix for a list of (h5ID, spectra)
        values (spectra is 1001 degree vector indexed to m/z) that is L2 
        normalized

        Params
        ------
        value                   list of (h5ID, spectra)

        Returns
        -------
        ids                     list of h5IDs where idx = idx in matrix
        mat                     matrix of spectra
        sim_mat                 similarity matrix for mat
        """
        vecs = []
        ids = []
        for h5id,array in value:
            norm = np.linalg.norm(array)
            if norm > 0:
                vecs.append(array / norm)
                ids.append(h5id)
        mat = np.stack(vecs)
            
        sim_mat = mat @ mat.T

        return ids, mat, sim_mat
    
    def _compute_sim_matrix(self, mat: torch.Tensor, batch_size: int = 500):
        """
        takes a matrix and computes a cosine similarity matrix using either a cpu
        or a gpu depending on what device is specified

        Params
        ------
        mat                         matrix to compute simliariy matrix for
        batch_size                  how many rows to compute at once

        Returns
        -------
        sim_path                    path to sim_matrix.dat
        """
        device = self.device
        n_nodes = mat.shape[0]
        sim_path = self.data_dir / "sim_matrix.dat"
        sim_mat = np.memmap(sim_path, dtype=np.float32, mode="w+", shape=(n_nodes,n_nodes))

        if device == "cpu":
            for start in range(0,n_nodes,batch_size):
                end = min(start+batch_size, n_nodes)
                batch = mat[start:end]
                sim_mat[start:end] = (batch @ mat.T).numpy()
                sim_mat.flush()
        
        elif device == "cuda":
            x = mat.to("cuda")
            for start in range(0, n_nodes, batch_size):
                end = min(start+batch_size, n_nodes)
                sim_mat[start:end] = (x[start:end] @ x.T).cpu().numpy()
                sim_mat.flush()
        
        log.info(f"Computed sim matrix on {device} mode")

        return sim_path
    
    def _build_edges(self, sim_path: Path, t_val: float, n_nodes: int, n_neighbors: int = 10):
        """
        Builds edges for the network based on cosine similarity values above t_val
        and ensuring the t_val most similar spectra are connected to each node

        Params
        ------
        sim_path                        path to the similarity matrix
        t_val                           threshold value above which add edge
        n_nodes                         how many nodes are in the matrix
        n_neighbors                     min edges for each node

        Returns
        -------
        sources                         tensor of source nodes for edges
        targets                         tensor of target nodes for edges
        weights                         tensor of edge weights (cos similaritites)
        """
        sim_mat = np.memmap(sim_path, dtype=np.float32, mode="r", shape=(n_nodes,n_nodes))
        sources = []
        targets = []
        weights = []

        for i in range(n_nodes):
            row = sim_mat[i].copy()
            row[i] = -1
            row_t = torch.tensor(row, dtype=torch.float32)

            # add top n_neighbors edges
            vals, idx = torch.topk(row_t,n_neighbors)
            for j,v in zip(idx.tolist(), vals.tolist()):
                sources.append(i)
                targets.append(j)
                weights.append(v)

            """
            # add additional edges for all that are above threshold
            above = np.where(row > t_val)[0]
            for j in above:
                sources.append(i)
                targets.append(int(j))
                weights.append(float(row[j]))
            """

        # convert to tensors
        t_sources = torch.tensor(sources, dtype=torch.long)
        t_targets = torch.tensor(targets, dtype=torch.long)
        t_weights = torch.tensor(weights, dtype=torch.float32)
        log.info("Built edges")
        
        # delete the similarity matrix
        del sim_mat
        sim_path.unlink()
        log.info("Deleted sim matrix file")

        return t_sources, t_targets, t_weights
    
    # endregion

    # region Plotting

    def t_val_hist(self, title: str, upper_tri: torch.Tensor, t_val: float = -1):
        """
        Plots a histogram of all values included in upper triangle of matrix 
        with a dotted red line vertically at t_val

        Params
        ------
        title                       title for the plot
        upper_tri                   upper triangle of sim matrix to plot
        t_val                       threshold value to plot
        """

        vals = upper_tri.numpy()
        max = upper_tri.max().item()
        min = upper_tri.min().item()

        plt.hist(vals,bins=50)
        if t_val != -1:
            plt.axvline(t_val, color="red",linestyle="--",linewidth=1.2)
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Count")
        plt.title(f"{title}\nMax: {max:.2f}     Min: {min:.2f}     Threshold: {t_val:.2f}")
        safe_title = re.sub(r'[^\w]', '_', title)
        plt.savefig(self.data_dir / f"{safe_title}.pdf")
        plt.close()
        log.info(f"Generated similarity histogram {safe_title}.pdf")

    def replicate_hist(self, title: str, groups: defaultdict):
        """
        Plots a histogram of all cosine similarities for each group of replicate spectra for 
        a given casNO as well as a table of outliers

        Params
        ------
        title                       title for the plot
        groups                      defaultdict(list) of casNO: list(h5ID, spectra)
        t_val                       threshold value for plotting
        """
        t_val = self.rep_threshold

        sims = []
        include_threshold = False

        for _, replicate_list in groups.items():
            
            # if replicates exist get similarity matrix
            if len(replicate_list) > 1:
                _, _, sim_mat = self._rep_sim_mat(replicate_list)
            else:
                continue

            rows,cols = np.triu_indices(sim_mat.shape[0], k=1)
            for sim in sim_mat[rows,cols]:
                sims.append(sim)
                if include_threshold == False and sim < t_val:
                    include_threshold = True
            
        plt.hist(sims,bins=50)
        if include_threshold == True:
            plt.axvline(t_val, color="red",linestyle="--",linewidth=1.2)
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Count")
        plt.title(f"{title}\nThreshold: {t_val}")
        safe_title = re.sub(r'[^\w]', '_', title)
        plt.savefig(self.data_dir / f"{safe_title}.pdf")
        plt.close()
        log.info(f"Generated repolicate similiarity histogram {safe_title}.pdf")

    def degree_hist(self, data: Data):
        """
        Geneartes degree distributions both log-log (for power law) and untransformed for the 
        network

        Params
        ------
        data                    pytorch Data object
        """

        if data.edge_index is None:
            log.warning(f"No edge indexes found for network {self.name}")
            return
        
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        deg = deg.numpy()

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
        ax1.hist(deg,bins=50)
        ax1.set_xlabel("Degree")
        ax1.set_ylabel("Count")
        ax1.set_title("Degree Distribution")

        ax2.hist(deg,bins=50,log=True)
        ax2.set_xscale("log")
        ax2.set_xlabel("Log Degree")
        ax2.set_ylabel("Log Count")
        ax2.set_title("Log-Log Degree Distribution")

        fig.suptitle(f"{self.name} Network Degree Distribution\nn_nodes: {data.num_nodes}   n_edges: {data.num_edges}")
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(self.data_dir / f"{self.name}_Network_Degree_Distribution.pdf")
        plt.close()

        log.info(f"Plotted degree histogram for network {self.name}")

    # endregion
