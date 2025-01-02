import logging
import time
from functools import partial

import torch
from torch_geometric.graphgym.register import register_loader

from loader.split_generator import (setup_standard_split,
                                    set_dataset_splits)
from transform.posenc_stats import compute_posenc_stats
from transform.transforms import pre_transform_in_memory


@register_loader('custom_master_loader')
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