import torch
from torch_geometric.graphgym.models.encoder import AtomEncoder
from torch_geometric.graphgym.register import register_node_encoder

from encoder.laplace_pos_encoder import LapPENodeEncoder


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


# Register the Atom + LapPE encoder
register_node_encoder("Atom+LapPE", AtomLapPENodeEncoder)