import os
import sys

import numpy as np
import scanpy as sc
import torch
from scdataloader import Collator
from scdataloader.data import SimpleAnnDataset
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

# Adjust these imports to your project structure
from scprint import scPrint


def extract_gene_embeddings(
    model,
    adata,
    batch_size: int = 64,
    num_workers: int = 8,
    device: str = "cuda",
    gene_list: list = None,
):
    """
    Extract gene embeddings from a scPrint model for a given AnnData object.

    Args:
        model (scPrint): A loaded and trained scPrint model.
        adata (AnnData): AnnData containing cell x gene matrix.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for DataLoader.
        device (str): Device to run on ("cuda" or "cpu").
        gene_list (list): Optional list of genes to restrict to. If None, uses model.genes.

    Returns:
        embeddings (np.ndarray): A numpy array of shape (n_cells, n_genes, embedding_dim)
        genes_processed (list): The list of genes processed in order.
    """
    model.eval()
    model.to(device)
    model.pred_log_adata = False

    # Determine which genes to use
    if gene_list is None:
        gene_list = model.genes
    else:
        gene_list = [g for g in gene_list if g in model.genes]
        if len(gene_list) == 0:
            raise ValueError("No overlap between provided gene_list and model.genes")

    # Set up dataset and dataloader
    # If needed, ensure adata.obs contains 'organism_ontology_term_id' or adapt Collator arguments
    if "organism_ontology_term_id" not in adata.obs:
        # Assign a default organism if needed
        adata.obs[
            "organism_ontology_term_id"
        ] = "NCBITaxon:9606"  # or your relevant organism ID

    adataset = SimpleAnnDataset(adata, obs_to_output=["organism_ontology_term_id"])
    col = Collator(
        organisms=model.organisms,
        valid_genes=model.genes,
        max_len=0,
        how="some",
        genelist=gene_list,
    )
    dataloader = DataLoader(
        adataset,
        collate_fn=col,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    all_embeddings = []

    # Use autocast to ensure half precision if required by the model
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
        for batch in dataloader:
            gene_pos, expression, depth = (
                batch["genes"].to(device),
                batch["x"].to(device),
                batch["depth"].to(device),
            )

            # Run encode_only to get transformer outputs
            output = model(
                gene_pos=gene_pos,
                expression=expression,
                req_depth=depth,
                get_gene_emb=True,
            )
            # transformer_output shape: (B, cell_embs_count + num_genes, d_model)
            # Extract gene embeddings:
            gene_embeddings = output["gene_embedding"]
            # shape: (B, num_genes, d_model)
            all_embeddings.append(gene_embeddings.cpu().numpy())
            del output
            torch.cuda.empty_cache()

    # Concatenate all the embeddings for all cells
    embeddings = np.concatenate(
        all_embeddings, axis=0
    )  # shape: (n_cells, n_genes, d_model)
    return embeddings, gene_list
