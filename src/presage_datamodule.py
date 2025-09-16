import os
from zipfile import ZipFile
import scanpy as sc
import json
import re
import pandas as pd
import numpy as np
from scipy import sparse

import pytorch_lightning as pl
from datamodule import scPerturbDataModule, ReplogleDataModule, compute_pseudobulk
from torch.utils.data import DataLoader, Dataset


class PRESAGEDataModule(scPerturbDataModule):
    urls = {
        "adamson": "https://dataverse.harvard.edu/api/access/datafile/6154417",
        "dixit": "https://dataverse.harvard.edu/api/access/datafile/6154416",
        "replogle_k562_essential": "https://dataverse.harvard.edu/api/access/datafile/7458695",
        "replogle_rpe1_essential": "https://dataverse.harvard.edu/api/access/datafile/7458694",
        "wessels_2023":"perturb_processed.h5ad",
        "replogle_2020":"perturb_processed.h5ad",
        "replogle_k562_essential_unfiltered":"perturb_processed.h5ad",
        "replogle_rpe1_essential_unfiltered":"perturb_processed.h5ad",
        "replogle_k562_gw":"perturb_processed.h5ad",
        "nadig_hepg2":"perturb_processed.h5ad",
        "nadig_jurkat":"perturb_processed.h5ad",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print(self.data_dir)
        os.makedirs(self.dataset_dir, exist_ok=True)

    @property
    def preprocessed_path(self) -> str:
        return os.path.join(self.dataset_dir, f"{self.dataset}_processed.h5ad")

    @property
    def dataset_dir(self) -> str:
        return os.path.join(self.data_dir, self.dataset)

    @property
    def deg_dir(self) -> str:
        return os.path.join(self.dataset_dir, "degs")

    @property
    def download_path(self) -> str:
        url = self.urls[self.dataset]
        filename = url.split("/")[-1]
        return os.path.join(self.dataset_dir, filename)

    @property
    def raw_path(self) -> str:
        return os.path.join(self.dataset_dir, "perturb_processed.h5ad")

    def prepare_data(self) -> None:
        # Download archive
        print(self.download_path)
        if not os.path.exists(self.download_path):
            if self.url == "perturb_processed.h5ad":     
                print(f"{self.dataset} path is not downloadable. This must be downloaded separately")
                quit()
            print(f"Downloading from {self.url}")
            self.download(self.url, self.download_path)
        else:
            print(f"Found local data file {self.download_path}")

        # Extract pre-processed data
        if not os.path.exists(self.raw_path):
            with ZipFile(self.download_path, "r") as f:
                f.extractall(path=self.data_dir)
        else:
            print(f"Found local extracted data file {self.raw_path}")
        
        

        if not os.path.exists(self.preprocessed_path):
            adata = sc.read_h5ad(self.raw_path)
            adata.obs.index = np.arange(adata.shape[0]).astype(str)

            # remove perturbation column if it exists
            adata.obs = adata.obs.loc[:,adata.obs.columns != 'perturbation']
            
            # Harmonize to scPerturb keys
            adata.obs.rename(columns=dict(condition="perturbation"), inplace=True)
            
            adata.obs["perturbation"] = adata.obs["perturbation"].apply(
                lambda x: x.replace("+", "_").replace("ctrl", "").strip("_")
            )
            
            adata.obs["perturbation"].replace("", "control", inplace=True)
            # Add nperts series
            adata.obs["nperts"] = adata.obs["perturbation"].apply(
                lambda x: 1 + ("_" in x)
            )
            adata.obs.loc[adata.obs["perturbation"] == "control", "nperts"] = 0
            
            if 'gene_name' not in adata.var.columns:
                adata.var.index.name = None
                adata.var['gene_name'] = adata.var.index.values
            
            adata.var["gene_name"] = adata.var["gene_name"].astype(str)
            adata.var = adata.var.reset_index().set_index("gene_name")
            adata.var_names_make_unique()
            
            # this is hacky to add perturbations that were not measured by gex
            # we will remove these later before eval
            missing_perturbations_from_gex = np.asarray(adata.obs.perturbation[np.isin(adata.obs.perturbation,
                                                                                       adata.var.index,
                                                                                       invert=True)].unique())
                                                                                       
                        
            adata.var = pd.DataFrame(adata.var.index)

            adata.var.index = adata.var.gene_name
            
            adata.var['measured_gene'] = True

            saved_metadata = adata.obs.copy()
            
            adata_missing_perturbations = sc.AnnData(X=np.zeros((adata.shape[0],
                                                                len(missing_perturbations_from_gex))))

            adata_missing_perturbations.obs = saved_metadata.copy()

            adata_missing_perturbations.var['measured_gene'] = False
            adata_missing_perturbations.var.index = missing_perturbations_from_gex
            adata_missing_perturbations.var['gene_name'] = adata_missing_perturbations.var.index.copy()

            adata.var.index.name = None
            adata_missing_perturbations.var.index.name = None

            adata.obs.index = adata.obs.index.astype('object')
            adata_missing_perturbations.obs.index = adata_missing_perturbations.obs.index.astype('object')
            adata.var.index = adata.var.index.astype('object')
            adata_missing_perturbations.var.index = adata_missing_perturbations.var.index.astype('object')

            adata.var = adata.var.loc[:,['gene_name','measured_gene']]
            adata_missing_perturbations.var = adata_missing_perturbations.var.loc[:,['gene_name','measured_gene']]
            
            if isinstance(adata.X,sparse._csr.csr_matrix):
                adata.X = adata.X.toarray()

            if adata_missing_perturbations.shape[1] > 0:
                # im not sure why the .T works, but it doesn't work without it
                adata = sc.concat([adata.T,adata_missing_perturbations.T],
                                axis=0,
                                join='outer').T
            adata.X = adata.X.astype(np.float32)
            adata.obs = saved_metadata.copy()
            
            adata.write(self.preprocessed_path)
        else:
            print(f"Found local preprocessed data file {self.preprocessed_path}")

        # Compute DEGs
        if not os.path.exists(self.merged_deg_file):
            print("Computing DEGs...")
            print(self.raw_path)
            adata = sc.read_h5ad(self.raw_path)
            adata.obs.index = np.arange(adata.shape[0]).astype(str)
            
            if 'gene_name' in adata.var.columns:
                id2name_table = adata.var.reset_index().set_index("index")['gene_name']
                #adata.var.columns = ['gene_name']
            else:
                id2name_table = adata.var.reset_index().set_index("gene_id")["gene_name"]
                #adata.var.columns = ['gene_name']

            
            degs = adata.uns["rank_genes_groups_cov_all"]
            
            # Harmonize to scPerturb perturbation keys
            degs = {
                re.sub(
                    r"^.*?_(.*)",
                    r"\1",
                    k.replace("_1+1", "").replace("+", "_").replace("ctrl", ""),
                )
                .strip("_"): id2name_table[v]
                .tolist()
                for k, v in degs.items()
            }
            with open(self.merged_deg_file, "w") as f:
                json.dump(degs, f)
        else:
            print(f"Found local preprocessed data file {self.merged_deg_file}")


class ReploglePRESAGEDataModule(PRESAGEDataModule, ReplogleDataModule):
    pass
