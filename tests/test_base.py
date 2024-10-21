import os
import urllib.request

import numpy as np
import pytest
import scanpy as sc
import torch
from scdataloader import Preprocessor
from scdataloader.utils import populate_my_ontology

from scprint import scPrint
from scprint.base import NAME
from scprint.tasks import Denoiser, Embedder, GNInfer


def test_base():
    assert NAME == "scprint"
    populate_my_ontology(
        organisms=["NCBITaxon:10090", "NCBITaxon:9606"],
        sex=["PATO:0000384", "PATO:0000383"],
        # celltypes=None,
        # ethnicities=None,
        # assays=None,
        # tissues=None,
        # diseases=None,
        # dev_stages=None,
    )
    filepath = os.path.join(os.path.dirname(__file__), "test.h5ad")
    ckpt_path = os.path.join(os.path.dirname(__file__), "small.ckpt")
    if not os.path.exists(ckpt_path):
        url = "https://huggingface.co/jkobject/scPRINT/resolve/main/small.ckpt"
        urllib.request.urlretrieve(url, ckpt_path)

    adata = sc.read_h5ad(filepath)
    adata.obs.drop(columns="is_primary_data", inplace=True, errors="ignore")
    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    preprocessor = Preprocessor(
        do_postp=False,
        force_preprocess=True,
    )
    adata = preprocessor(adata)
    # conf = dict(self.config_init[subcommand])
    try:
        model = scPrint.load_from_checkpoint(
            ckpt_path,
            precpt_gene_emb=None,
            # triton gets installed so it must think it has cuda enabled
            transformer="normal",
        )
        dn = Denoiser(
            max_cells=10,
            batch_size=2,
            num_workers=1,
            max_len=300,
            downsample=0.3,
            predict_depth_mult=3,
            dtype=torch.float32,
        )
        metrics, random_indices, genes, expr_pred = dn(
            model=model,
            adata=adata,
        )
        assert (
            metrics["reco2full"] - metrics["noisy2full"] > 0
        ), "Model is not denoising"
        # emb, class, grn inf and fit function for scPRINT
        # Cell embedding
        cell_embedder = Embedder(
            batch_size=2,
            num_workers=1,
            how="random expr",
            max_len=300,
            doclass=True,
            pred_embedding=[
                "cell_type_ontology_term_id",
                "disease_ontology_term_id",
                "self_reported_ethnicity_ontology_term_id",
                "sex_ontology_term_id",
            ],
            plot_corr_size=8,
            doplot=True,
            keep_all_cls_pred=False,
            dtype=torch.float32,
        )
        adata_emb, metrics = cell_embedder(model, adata[:10, :])
        assert "scprint" in adata_emb.obsm, "Cell embedding failed"

        assert any(
            col.startswith("pred_") for col in adata_emb.obs.columns
        ), "Classification failed"

        # GRN inference
        grn_inferer = GNInfer(
            layer=[0, 1],
            batch_size=2,
            how="random expr",
            preprocess="softmax",
            head_agg="mean",
            filtration="none",
            forward_mode="none",
            num_genes=100,
            max_cells=10,
            doplot=False,
        )
        grn_adata = grn_inferer(model, adata)
        assert "GRN" in grn_adata.varp, "GRN inference failed"
        # fit scprint
        # cli
        # get_Seq
        # sinkhorn
        # knn_smooth
        # layernorm
        # encoders
        # model
        # cell emb
        # denoise
        # grn
    except Exception as e:
        pytest.fail(f"An exception occurred: {str(e)}")
