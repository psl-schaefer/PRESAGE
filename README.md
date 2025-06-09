# PRESAGE (Perturbation Response EStimation with Aggregated Gene Embeddings)

This repository contains the code accompanying the PRESAGE manuscript (https://www.biorxiv.org/content/10.1101/2025.06.03.657653v1):

**Gene-embedding-based prediction and functional evaluation of perturbation expression responses with PRESAGE**

Understanding the impact of genetic perturbations on cellular behavior is crucial for biological research, but comprehensive experimental mapping remains infeasible. We introduce PRESAGE (Perturbation Response EStimation with Aggregated Gene Embeddings), a simple, modular, and interpretable framework that predicts perturbation-induced expression changes by integrating diverse knowledge sources via gene embeddings. PRESAGE transforms gene embeddings through an attention-based model to predict perturbation expression outcomes. To assess model performance, we introduce a comprehensive evaluation suite with novel functional metrics that move beyond traditional regression tasks, including measures of accuracy in effect size prediction, in identifying perturbations with similar expression profiles (phenocopy), and in prediction of perturbations with the strongest impact on specific gene set scores.  PRESAGE outperforms existing methods in both classical regression metrics and our novel functional evaluations. Through ablation studies, we demonstrate that knowledge source selection is more critical for predictive performance than architectural complexity, with cross-system Perturb-seq data providing particularly strong predictive power. We also find that performance saturates quickly with training set size, suggesting that experimental design strategies might benefit from collecting sparse perturbation data across multiple biological systems rather than exhaustive profiling of individual systems. Overall, PRESAGE establishes a robust framework for advancing perturbation response prediction and facilitating the design of targeted biological experiments, significantly improving our ability to predict cellular responses across diverse biological systems.

## Cite Us

## Installation
### Clone this repo  
```sh
git clone https://github.com/genentech/PRESAGE
cd PRESAGE
```

### Create environment for PRESAGE
```sh
conda env create --channel-priority flexible -f environment.yml
```  
Note: This also works with mamba or micromamba
  
### Download and unpack cached files
To run PRESAGE, you will need to download the cached gene embeddings from https://zenodo.org/records/15587986.  
```sh
wget -c -O cache.tar.gz https://zenodo.org/records/15587986/files/cache.tar.gz?download=1
```
Then, run:

```sh
./src/prep_dataset_utils/unpack_cache.sh
```

### Download datasets
To download and prepare the datasets used in this study:  
```sh
./src/prep_dataset_utils/download_datasets.sh
```   
Note: This reads in and cleans the data so an interactive node or cpu with memory is required.

## Tutorials
An example of how to run PRESAGE can be found in `notebooks/PRESAGE_example.ipynb`.
  
An example of how to run the PRESAGE evaluation suite from a set of predictions can be found in `notebooks/EvaluateFromPredictions.ipynb`.

## Data attribution and license

This codebase is licensed under the Genentech Non-Commercial Software License Version 1.0.
For more information, please see the attached LICENSE.txt file.

The download script `./src/prep_dataset_utils/download_datasets.sh` downloads the following datasets:

* replogle_k562_essential_unfiltered, replogle_k562_gw, replogle_rpe1_essential_unfiltered
  * Replogle, Joseph M., et al.  Cell 185.14 (2022). DOI: 10.1016/j.cell.2022.05.013 [(link)](https://www.cell.com/cell/pdf/S0092-8674(22)00597-9.pdf)
  * License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* nadig_hepg2, nadig_jurkat
  * Nadig, Ajay, et al. bioRxiv (2024). DOI: 10.1101/2024.07.03.601903 [(link)](https://www.biorxiv.org/content/10.1101/2024.07.03.601903v1)
  * License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Note that the files in the `cache/data` and `cache/splits` folders of the data distribution are derived from these datasets as well.

The knowledge source embeddings in the data distribution (`cache/pathway_embeddings/*` and `cache/other_embeddings/*`) are licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) license and derived from the following sources:

* [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/)
  * Subramanian, Aravind, et al. Proceedings of the National Academy of Sciences 102.43 (2005). DOI: 10.1073/pnas.0506580102 [(link)](https://doi.org/10.1073/pnas.0506580102)
  * Liberzon, Arthur, et al. Cell systems 1.6 (2015). DOI: 10.1016/j.cels.2015.12.004 [(link)](https://doi.org/10.1016/j.cels.2015.12.004)
  * License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* [STRINGDB](https://string-db.org/)
  * Szklarczyk et al. Nucleic acids research 51.D1 (2023). DOI: 10.1093/nar/gkac1000 [(link)](https://doi.org/10.1093/nar/gkac1000)
  * License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* Periscope data
  * Ramezani, Meraj, et al. Nature Methods (2025). DOI: 10.1038/s41592-024-02537-7 [(link)](https://doi.org/10.1038/s41592-024-02537-7)
  * License: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/deed.en)
* BioGPT embeddings
  * Luo, Renqian, et al. Briefings in Bioinformatics 23.6 (2022). DOI: 10.1093/bib/bbac409 [(link)](https://doi.org/10.1093/bib/bbac409)
  * License (model): [MIT](https://github.com/microsoft/BioGPT?tab=MIT-1-ov-file)
* DepMap CRISPR Gene Effect
  * DepMap, Broad (2024). DepMap 24Q2 Public. Figshare+. Dataset. DOI: 10.25452/figshare.plus.25880521.v1 [(link)](https://doi.org/10.25452/figshare.plus.25880521.v1)
  * License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* ESM2 embeddings
  * Lin, Zeming, et al. Science 379.6637 (2023). DOI: 10.1126/science.ade2574 [(link)](https://doi.org/10.1126/science.ade2574)
  * License (model): [MIT](https://github.com/facebookresearch/esm?tab=MIT-1-ov-file)
* Funk et al. OPS data
  * Funk, Luke, et al. Cell 185.24 (2022). [(link)](https://doi.org/10.1016/j.cell.2022.10.017)
  * License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
* GenePT
  * Chen, Yiqun, and James Zou. bioRxiv (2024): 2023-10. DOI: 10.1101/2024.10.27.620513 [(link)](https://doi.org/10.1101/2024.10.27.620513)
  * License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
