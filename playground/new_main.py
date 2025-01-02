import torch

import numpy as np

from torch_geometric.datasets import LRGBDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


class LaplacianPEWithAtomEncoding(BaseTransform):
    def __init__(self, num_eigenvectors=10, emb_dim=128):
        self.num_eigenvectors = num_eigenvectors
        self.emb_dim = emb_dim

    def __call__(self, data):
        # Compute Laplacian eigenvalues and eigenvectors
        laplacian_norm_type = "sym"
        L = to_scipy_sparse_matrix(
            *get_laplacian(data.edge_index, normalization=laplacian_norm_type, num_nodes=data.num_nodes)
        )
        evals, evects = np.linalg.eigh(L.toarray())
        
        # Select top-k eigenvectors and normalize
        idx = evals.argsort()[:self.num_eigenvectors]
        evals, evects = evals[idx], np.real(evects[:, idx])
        evals = torch.from_numpy(np.real(evals)).clamp_min(0)

        evects = torch.from_numpy(evects).float()
        evects = self._l2_normalize(evects)

        # Pad and save eigenvalues and eigenvectors
        data.lap_pe = evects
        if data.x is not None:
            data.x = torch.cat([data.x, evects], dim=1)  # Concatenate LapPE to node features
        else:
            data.x = evects
        return data

    def _l2_normalize(self, eigenvectors, eps=1e-12):
        denom = eigenvectors.norm(p=2, dim=0, keepdim=True).clamp_min(eps)
        return eigenvectors / denom


def main():
    name = "Peptides-struct"

    # Define the transform
    num_eigenvectors = 10
    emb_dim = 128
    transform = LaplacianPEWithAtomEncoding(num_eigenvectors=num_eigenvectors, emb_dim=emb_dim)

    # Load dataset with the pre-transform
    dataset = LRGBDataset(root="data/LRGBDataset", name=name, pre_transform=transform)
    print(f"Dataset loaded with {len(dataset)} graphs.")

    # Save the transformed dataset
    output_path = "Peptides-struct-transformed.pt"
    torch.save(dataset, output_path)
    print(f"Transformed dataset saved to: {output_path}")

if __name__ == "__main__":
    main()