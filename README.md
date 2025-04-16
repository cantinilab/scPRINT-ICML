
# XPressor and ESM2-fine-tuning version of scPRINT

scPRINT is a large transformer model built for the inference of gene networks (connections between genes explaining the cell's expression profile) from scRNAseq data.

It uses novel encoding and decoding of the cell expression profile and new pre-training methodologies to learn a cell model.

scPRINT can be used to perform the following analyses:

- __expression denoising__: increase the resolution of your scRNAseq data
- __cell embedding__: generate a low-dimensional representation of your dataset
- __label prediction__: predict the cell type, disease, sequencer, sex, and ethnicity of your cells
- __gene network inference__: generate a gene network from any cell or cell cluster in your scRNAseq dataset

[Read the manuscript!](https://www.biorxiv.org/content/10.1101/2024.07.29.605556v1) if you would like to know more about XPressor.

![figure1](docs/figure1.png)

🎊 test scPRINT and scDataloader on this simple [google collab](https://colab.research.google.com/drive/1CacoQDAwJn86tq2sBhUoZ6M-xAqsYFDI#scrollTo=Lb4E9IhQ7NK8)

## Reproducibility 

To reproduce the results in the XPressor paper:
- Follow the installation instruction from regular scPRINT (kept in this README)
- run the fit-loop of scPRINT (full train/val/test program) for each of the 3 different versions using these commands:
   - scprint fit --config ablation_study.yml --model.cell_specific_blocks True
   - scprint fit --config ablation_study.yml --model.finetune_gene_emb True
   - scprint fit --config ablation_study.yml

## Changes made:

- we have updated the Flash-Attention package from scPRINT to integrate the ability to use the XPressor architecture.
- we have updated the model.py file to add ESM2 fine tuning and the XPressor architecture.

## Install `scPRINT`

For the moment scPRINT has been tested on MacOS and Linux (Ubuntu 20.04) with Python 3.10. Its instalation takes on average 10 minutes.

If you want to be using flashattention2, know that it only supports triton 2.0 MLIR's version and torch==2.0.0 for now.

### lamin.ai

To use scPRINT, you will need to use [lamin.ai](https://lamin.ai/). This is needed to load biological informations like genes, cell types, organisms etc...

### install

To start you will need to do:

```bash
uv venv -n <env-name> python==3.10 #scprint might work with python >3.10, but it is not tested
#one of
uv pip install scprint # OR
uv pip install scprint[dev] # for the dev dependencies (building etc..) OR
uv pip install scprint[flash] # to use flashattention2 with triton: only if you have a compatible gpu (e.g. not available for apple GPUs for now, see https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility)
#OR pip install scPRINT[dev,flash]

lamin init --storage ./testdb --name test --schema bionty
```

if you start with lamin and had to do a `lamin init`, you will also need to populate your ontologies. This is because scPRINT is using ontologies to define its cell types, diseases, sexes, ethnicities, etc.

you can do it manually or with our function:

```python
from scdataloader.utils import populate_my_ontology

populate_my_ontology() #to populate everything (recommended) (can take 2-10mns)

populate_my_ontology( #the minimum for scprint to run some inferences (denoising, grn inference)
organisms: List[str] = ["NCBITaxon:10090", "NCBITaxon:9606"],
    sex: List[str] = ["PATO:0000384", "PATO:0000383"],
    celltypes = None,
    ethnicities = None,
    assays = None,
    tissues = None,
    diseases = None,
    dev_stages = None,
)
```

We make use of some additional packages we developed alongside scPRint.

Please refer to their documentation for more information:

- [scDataLoader](https://github.com/jkobject/scDataLoader): a dataloader for training large cell models.
- [GRnnData](https://github.com/cantinilab/GRnnData): a package to work with gene networks from single cell data.
- [benGRN](https://github.com/jkobject/benGRN): a package to benchmark gene network inference methods from single cell data.

### pytorch and GPUs

scPRINT can run on machines without GPUs, but it will be slow. It is highly recommended to use a GPU for inference.

Once you have a GPU, and installed the required drivers, you might need to install a specific version of pytorch that is compatible with your drivers (e.g. nvidia 550 drivers will lead to a nvidia toolkit 11.7 or 11.8 which might mean you need to re-install a different flavor of pytorch for things to work. e.g. using the command:
`pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118` on my case on linux
 ).

I was able to test it with nvidia 11.7, 11.8, 12.2.

### dev install

If you want to use the latest version of scPRINT and work on the code yourself use `git clone` and `pip -e` instead of `pip install`.

```bash
git clone https://github.com/cantinilab/scPRINT
git clone https://github.com/jkobject/scDataLoader
git clone https://github.com/cantinilab/GRnnData
git clone https://github.com/jkobject/benGRN
pip install -e scPRINT[dev]
pip install -e scDataLoader[dev]
pip install -e GRnnData[dev]
pip install -e benGRN[dev]
```

## Usage

### scPRINT's basic commands

This is the most minimal example of how scPRINT works:

```py
from lightning.pytorch import Trainer
from scprint import scPrint
from scdataloader import DataModule

datamodule = DataModule(...)
model = scPrint(...)
# to train / fit / test the model
trainer = Trainer(...)
trainer.fit(model, datamodule=datamodule)
# to do predictions Denoiser, Embedder, GNInfer
denoiser = Denoiser(...)
adata = sc.read_h5ad(...)
denoiser(model, adata=adata)
...
```

or, from a bash command line

```bash
$ scprint fit/train/predict/test/denoise/embed/gninfer --config config/[medium|large|vlarge] ...
```

find out more about the commands by running `scprint --help` or `scprint [command] --help`.

more examples of using the command line are available in the [docs](./docs/usage.md).

### Notes on GPU/CPU usage with triton

If you do not have [triton](https://triton-lang.org/main/python-api/triton.html) installed you will not be able to take advantage of GPU acceleration, but you can still use the model on the CPU.

In that case, if loading from a checkpoint that was trained with flashattention, you will need to specify `transformer="normal"` in the `load_from_checkpoint` function like so:

```python
model = scPrint.load_from_checkpoint(
    '../data/temp/last.ckpt', precpt_gene_emb=None,
    transformer="normal")
```
