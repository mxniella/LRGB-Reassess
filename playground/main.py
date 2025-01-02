import logging
import time
import torch
import tqdm

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import set_dataset_attr
from torch_geometric.graphgym.models.encoder import AtomEncoder
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_undirected


class LapPENodeEncoder(torch.nn.Module):
    """Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be linearly
    projected to (dim_emb - dim_pe) size and concatenated with LapPE.
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in  # Expected original input node features dim
        dim_pe = 16
        self.model_type = "DeepSet"
        n_layers = 2  # Num. layers in PE encoder model
        max_freqs = 10  # Num. eigenvectors (frequencies)
        norm_type = None  # Raw PE normalization layer type

        pecfg = cfg.posenc_LapPE
        n_heads = pecfg.n_heads  # Num. attention heads in Trf PE encoder
        post_n_layers = pecfg.post_layers  # Num. layers to apply after pooling
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"LapPE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        # Initial projection of eigenvalue and the node's eigenvector value
        self.linear_A = nn.Linear(2, dim_pe)
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        # Build DeepSet model
        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        layers = []
        self.linear_A = nn.Linear(2, 2 * dim_pe)
        layers.append(activation())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
            layers.append(activation())
        layers.append(nn.Linear(2 * dim_pe, dim_pe))
        layers.append(activation())
        self.pe_encoder = nn.Sequential(*layers)

        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(activation())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.post_mlp = nn.Sequential(*layers)


    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        EigVals = batch.EigVals
        EigVecs = batch.EigVecs

        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe
        pos_enc = self.pe_encoder(pos_enc)

        # Remove masked sequences; must clone before overwriting masked elements
        pos_enc = pos_enc.clone().masked_fill_(empty_mask[:, :, 0].unsqueeze(2),
                                               0.)

        # Sum pooling
        pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_LapPE = pos_enc
        return batch


class AtomLapPENodeEncoder(torch.nn.Module):
    """Encoder that combines Atom and Laplace Positional Encoding."""
    def __init__(self, dim_emb):
        super().__init__()
        # Retrieve positional encoding dimension from config
        dim_pe = 16
        self.atom_encoder = AtomEncoder(dim_emb - dim_pe)
        self.lap_pe_encoder = LapPENodeEncoder(dim_emb, expand_x=False)

    def forward(self, batch):
        # Encode node features using AtomEncoder
        batch = self.atom_encoder(batch)
        # Add positional encodings using LapPENodeEncoder
        batch = self.lap_pe_encoder(batch)
        return batch


def l2_eigvec_normalizer(EigVecs, EigVals, eps=1e-12):
    """
    Implements L2 eigenvector normalization.
    """
    EigVals = EigVals.unsqueeze(0)

    # L2 normalization: eigvec / sqrt(sum(eigvec^2))
    denom = EigVecs.norm(p=2, dim=0, keepdim=True)
    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = l2_eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def compute_posenc_stats(data, is_undirected):
    """Precompute Laplacian positional encodings (LapPE) for the given graph."""
    # Determine the number of nodes in the graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    laplacian_norm_type = "l2"

    # Ensure the graph is undirected if needed.
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Compute Laplacian eigenvalues and eigenvectors.
    L = to_scipy_sparse_matrix(
        *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                        num_nodes=N)
    )
    evals, evects = np.linalg.eigh(L.toarray())
        
    # Select the top max_freqs eigenvalues and normalize eigenvectors.
    max_freqs=10
    eigvec_norm="L2"
    data.EigVals, data.EigVecs = get_lap_decomp_stats(
        evals=evals, evects=evects,
        max_freqs=max_freqs,
        eigvec_norm=eigvec_norm)

    return data


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def setup_standard_split(dataset):
    """Ensure that standard graph-level splits exist in the dataset."""
    for split_name in 'train_graph_index', 'val_graph_index', 'test_graph_index':
        if not hasattr(dataset.data, split_name):
            raise ValueError(f"Missing '{split_name}' for standard split")
        

def set_dataset_splits(dataset, splits):
    """Set given graph-level splits to the dataset object.

    Args:
        dataset: PyG dataset object
        splits: List of train/val/test split indices.

    Raises:
        ValueError: If any pair of splits has intersecting indices.
    """
    # First check whether splits intersect and raise error if so.
    for i in range(len(splits) - 1):
        for j in range(i + 1, len(splits)):
            n_intersect = len(set(splits[i]) & set(splits[j]))
            if n_intersect != 0:
                raise ValueError(
                    f"Splits must not have intersecting indices: "
                    f"split #{i} (n = {len(splits[i])}) and "
                    f"split #{j} (n = {len(splits[j])}) have "
                    f"{n_intersect} intersecting indices"
                )

    # Set graph-level splits
    split_names = [
        'train_graph_index', 'val_graph_index', 'test_graph_index'
    ]
    for split_name, split_index in zip(split_names, splits):
        set_dataset_attr(dataset, split_name, split_index, len(split_index))


def preformat_Peptides(dataset_dir, name):
    """Load Peptides dataset, functional or structural.

    Note: This dataset requires RDKit dependency!

    Args:
        dataset_dir: path where to store the cached dataset
        name: the type of dataset split:
            - 'peptides-functional' (10-task classification)
            - 'peptides-structural' (11-task regression)

    Returns:
        PyG dataset object
    """
    try:
        # Load locally to avoid RDKit dependency until necessary.
        from graphgps.loader.dataset.peptides_functional import \
            PeptidesFunctionalDataset
        from graphgps.loader.dataset.peptides_structural import \
            PeptidesStructuralDataset
    except Exception as e:
        logging.error('ERROR: Failed to import Peptides dataset class, '
                      'make sure RDKit is installed.')
        raise e

    dataset_type = name.split('-', 1)[1]
    if dataset_type == 'functional':
        dataset = PeptidesFunctionalDataset(dataset_dir)
    elif dataset_type == 'structural':
        dataset = PeptidesStructuralDataset(dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'val', 'test']]
    return dataset


def load_dataset_master(name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """

    dataset = preformat_Peptides(dataset_dir, name)

    # Precompute necessary statistics for positional encodings.        
    pe_name = "LapPE"
    pe_enabled_list=[pe_name]
    
    start = time.perf_counter()
    logging.info(f"Precomputing LapPE statistics...")

    # Estimate directedness based on 10 graphs to save time.
    is_undirected = all(d.is_undirected() for d in dataset[:10])
    logging.info(f"  Estimated to be undirected: {is_undirected}")

    pre_transform_in_memory(dataset,
                            partial(compute_posenc_stats,
                                    pe_types=pe_enabled_list,
                                    is_undirected=is_undirected,
                                    ),
                            show_progress=True
                            )
    elapsed = time.perf_counter() - start
    logging.info(f"Done! Took {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    setup_standard_split(dataset)

    return dataset


def main():
    # Load dataset
    name = "peptides-functional"
    dataset_dir = "./loader/dataset/"
    output_dir = f"{name}-transformed"
    dim_emb = 128  # Set desired embedding dimension

    # Load dataset
    try:
        dataset = load_dataset_master(name, dataset_dir)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return

    # initialize encoder
    encoder = AtomLapPENodeEncoder(dim_emb)

    # encode dataset
    transformed_dataset = []
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for data in data_loader:
        try:
            data = encoder(data)
            transformed_dataset.append(data)
        except Exception as e:
            logging.warning(f"Failed to encode a graph: {e}")

    # Save the transformed dataset
    torch.save(transformed_dataset, output_dir)
    print("Finished Transformation")

if __name__ == "__main__":
    main()
