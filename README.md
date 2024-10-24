# ChronoGauge ensemble
Connor Reynolds<sub>1</sub>, Joshua Colmer<sub>1</sub>, Hannah Rees<sub>2</sub>, Ehsan Khajouei<sub>1</sub>, Rachel Rusholme-Pilcher<sub>1</sub>, Hiroshi Kudoh<sub>3</sub>, Antony Dodd<sub>4</sub>, Anthony Hall<sub>1,5</sub>

<sub>1</sub>Earlham Institute, Norwich Research Park  
<sub>2</sub>Institue of Biological, Environmental & Rural Sciences (IBERS)  
<sub>3</sub>Centre for Ecological Research, Kyoto University  
<sub>4</sub>John Innes Centre, Norwich Research Park  
<sub>5</sub>School of Biological Sciences, University of East Anglia  

## Overview
ChronoGauge is a bagging-like ensemble model for circadian time (CT) estimation from transcriptome samples (e.g. RNA-seq, microarray). The model was developed specifically for use in the context of plant transcriptome data.

ChronoGauge is trained using _Arabidopsis_ RNA-seq data and has been applied across various contexts including:
* Testing hypotheses related to the circadian clock in RNA-seq data (e.g. between control samples and those exposed to experimental pertubations)
* Microarray samples
* Non-model species using gene orthologs

The ensemble of subpredictors can be generated by running a custom sequential feature selection (SFS) wrapper multiple times using different seed values. Each SFS run outputs a unique feature set that can be used to train a subpredictor. These subpredictors can be applied to test datasets that include these feature sets and their predictions aggregated to provide a CT estimate that is robust in spite of technical variation/batch effects.

## System requiremnets
ChronoGauge has been tested locally on MacOS Sequoia 15.0.1 and Windows 10 Home 22H2 with systems including at least 8Gb RAM.

ChronoGauge requires the following dependencies:
* Python v3.9.5
* tensorflow
* scikit-learn
* numpy
* pandas
* tqdm

## Installation
A working environment for running ChronoGauge can be installed via Anaconda. Details on installing an Anaconda distribution can be found [here](https://www.anaconda.com/download/).

### For MacOS/Linux users
In command terminal:
```
git clone https://github.com/ConnorReynoldsUK/ChronoGauge
cd ChronoGauge
source install_current.sh
```

The environment installation should take ~2 minutes.

**Note:** The original script in our paper uses a conda environment installed specifically in AlmaLinux 5.14.0 OS on the HPC, which is not reproducible across other OS. We provide 'env/chronogauge_original.yml' which can be used to install the environment in the respective OS using:

```
conda env create PATH_TO_REPO/ChronoGauge/env/chronogauge_alma.yml -n chronogauge_alma
conda activate chronogauge_alma
```

## Sequential feature selection (SFS)
To build a set of predictive gene features, we provide the script `sfs_main_git.py` to execute a single SFS run. This script can be run using the following command, using a seed value of 0 as an example:

```
python3 sfs_main_git.py --seed 0 --max_genes 40 --n_gene_balance 25 --n_iterations 10000 & PID=$!; sleep 21600 && kill $PID
```
We add a kill command to terminate the script after 21600 seconds (6 hours), as the algorithm will not finish in a scalable time-frame.


A detailed and more simple example is shown in the notebook `example_sfs_wrapper.ipynb`.


The custom SFS wrapper takes:
1. A training expression matrix (e.g. `data/expression_matrices/x_training.csv`)
2. Time-point labels for each sample (e.g. `data/targets/target_training.csv`)
3. Prior circadian information regarding each gene feture's phase (in 4-hour bins ranging 0-24) & Q-value determined using MetaCycle [1] (meta2d method) (e.g. `data/sfs_input/sfs_gene_info.csv`). We provide the notebook `example_prior_information.ipynb` to demonstrate how prior information is obtained from the MetaCycle results. 
4. An ID integer value to initialize the random state of the script. When generating an ensemble of feature sets, the ID should be unique for each SFS run. 

Pre-processing follows these steps:
1. Only gene features with a circadian rhyhtmicity meta2d Q < 0.05 are selected
2. A bootstrap is applied to randomly select a determined proportion of genes (default 50%).
3. The top _N_ gene features from each phase bin are selected (default _N_ = 25).
4. A gene is selected at random to initialize the feature set.

The SFS algorithm iteratively searches for genes which optimize the mean-absolute-error (MAE) within 5-fold cross-validation, whilst also mantaining a balance across gene phase bins. The algorithm outputs any feature set giving a MAE under 60 mins. We recommended running the script for a specified duration (e.g. 6 CPU hours), then selecting the feature set giving the minimum MAE. The notebook `example_sfs_results.ipynb` demontrates how the outputs can be analyzed to select the optimal feature set.

![Sequential Feature Selection run](notebooks/sfs_example.png)

We note that the SFS is intended to generate multiple feature sets for building an ensemble of sub-predictors. While a single predictor can be generated, it is unlikely the model will be reliable across unseen data. To generate multiple featrure sets, we suggest running the SFS algorithm across a parellelized job array using high performance computing (HPC).

## Model training
To train a model with a specified feature set and list of hyperparameters, we provide the script `train_model.py`. By default, the script will train a model using 17 cannonical circadain clock genes as features.

Multiple models each with a unique ID value can be trained using the following command:

```
for i in {0..10}; 
do python3 train_model.py --x_test data/expression_matrices/x_test_rna.csv --target_test data/targets/target_test_rna.csv --out_model results/saved_model --model_id $i; 
done
```
Each script should take < 1 minute to complete.

Additionally, we provide the notebook `example_model_training.ipynb` as a more detailed walkthrough.

## Saved models
The ensemble of sub-models that were fit to the `data/expression_matrices/x_training.csv` RNA-seq expression matrix can be found within the following Hugging Face repositories:
* [ChronoGauge ensemble for _Arabidopsis_ RNA-seq data](https://huggingface.co/conjr94/ChronoGauge_RNAseq)
* [ChronoGauge ensemble for _Arabidopsis_ ATH1 microarray data](https://huggingface.co/conjr94/ChronoGauge_ATH1_microarray)
* [ChronoGauge ensemble for _Arabidopsis_ AraGene microarray data](https://huggingface.co/conjr94/ChronoGauge_AraGene_microarray)

The notebook `example_test_trained.ipynb` will demonstrate how to use saved sub-models to make predictions across test data.

The notebook `example_ensemble_aggregation.ipynb` will demonstrate how to aggregate the CT predictions across multiple sub-models using a circular mean. Based on cross-validation results, we expect that combining the CT predictions of multiple sub-models will give more reliable results compared with using any individual model.

![Single model errors vs. Ensemble errors](notebooks/ensemble_example.png)

While all models were trained using only RNA-seq data, ensembles generated for microarray experiments included only gene features that are present within each platforms gene set. Each ensemble includes 100 models using feature sets geenrated by SFS. 

## Applicability to mammalian transcriptome data
ChronoGauge was specifically developed for the purpose of investigating circadian clock function in plants, thus it has not been tested for mammalian/animal contexts. 

While there is no theoretical reason ChronoGauge should not be applicable to mammalian data, we note that there are multiple methods already developed for these context:
* MolecularTimetable [2]
* ZeitZeiger [3]
* TimesignatR [4]
* Partial-least-squares-regression [5]
* Taufisher [6]

We would therefore recommend researchers to evaluate multiple different approaches and to carefully consider the suitability and reliability of each method before deciding which one to use within their context.

## Datasets
The following datasets are included in this repository:
* Training _Arabidopsis_ data (_N_ samples = 56) includes experiments from _Cortijo et al._[7], _Yang et al._[8] and _Romanowski et al._[9].

* A collection of RNA-seq test datasets (_N_ samples = 58) were selected for benchmarking ChronoGauge including _Rugnone et al._[10], _Miller et al._[11], _Takahashi et al._[12], _Ezer et al._[13], _Graf et al._[14] and _Dubois et al._[15].

* A collection of ATH1 micorarray test datasets (_N_ samples = 73) were also selected for benchmarking including _Edwards et al._[16], _Covington et al._[17], _Michael et al._[18] and _Espinoza et al._[19].

* An AraGene microarray test dataset (_N_ samples = 72) was also selected for benchmarking by _Endo et al._[20].

* Samples for testing hypotheses related to clock include those by _Rugnone et al._[10], _Ezer et al._[13], _Graf et al._[14], _Blair et al._[21] and _Dubin et al._[22].


## References
1. Wu, G., Anafi, R. C., Hughes, M. E., Kornacker, K. & Hogenesch, J. B. MetaCycle: an integrated R package to evaluate periodicity in large scale data. Bioinformatics 32, 3351–3353 (2016).
2. Ueda, H. R., Chen, W., Minami, Y., Honma, S., Honma, K., Iino, M. & Hashimoto, S. Molecular-timetable methods for detection of body time and rhythm disorders from single-time-point genome-wide expression profiles. Proc. Natl. Acad. Sci. 101, 11227–11232 (2004).
3. Hughey, J. J., Hastie, T. & Butte, A. J. ZeitZeiger: supervised learning for high-dimensional data from an oscillatory system. Nucleic Acids Res. 44, e80 (2016).
4. Braun, R., Kath, W. L., Iwanaszko, M., Kula-Eversole, E., Abbott, S. M., Reid, K. J., Zee, P. C. & Allada, R. Universal method for robust detection of circadian state from gene expression. Proc. Natl. Acad. Sci. U. S. A. 115, E9247–E9256 (2018).
5. Laing, E. E., Möller-Levet, C. S., Poh, N., Santhi, N., Archer, S. N. & Dijk, D.-J. Blood transcriptome based biomarkers for human circadian phase. eLife 6, e20214 (2017).
6. Duan, J., Ngo, M. N., Karri, S. S., Tsoi, L. C., Gudjonsson, J. E., Shahbaba, B., Lowengrub, J. & Andersen, B. tauFisher predicts circadian time from a single sample of bulk and single-cell pseudobulk transcriptomic data. Nat. Commun. 15, 3840 (2024).
7. Cortijo, S., Aydin, Z., Ahnert, S. & Locke, J. C. Widespread inter‐individual gene expression variability in Arabidopsis thaliana. Mol. Syst. Biol. 15, (2019).
8. Yang, Y., Li, Y., Sancar, A. & Oztas, O. The circadian clock shapes the Arabidopsis transcriptome by regulating alternative splicing and alternative polyadenylation. J. Biol. Chem. 295, 7608 (2020).
9. Romanowski, A., Schlaen, R. G., Perez-Santangelo, S., Mancini, E. & Yanovsky, M. J. Global transcriptome analysis reveals circadian control of splicing events in Arabidopsis thaliana. Plant J. 103, 889–902 (2020).
10. Rugnone, M. L., Soverna, A. F., Sanchez, S. E., Schlaen, R. G., Hernando, C. E., Seymour, D. K., Mancini, E., Chernomoretz, A., Weigel, D., Más, P. & Yanovsky, M. J. LNK genes integrate light and clock signaling networks at the core of the Arabidopsis oscillator. Proc. Natl. Acad. Sci. U. S. A. 110, 12120 (2013).
11. Miller, M., Song, Q., Shi, X., Juenger, T. E. & Chen, Z. J. Natural variation in timing of stress-responsive gene expression predicts heterosis in intraspecific hybrids of Arabidopsis. Nat. Commun. 6, 7453 (2015).
12. Takahashi, N., Hirata, Y., Aihara, K. & Mas, P. A Hierarchical Multi-oscillator Network Orchestrates the Arabidopsis Circadian System. Cell 163, 148–159 (2015).
13. Ezer, D., Jung, J.-H., Lan, H., Biswas, S., Gregoire, L., Box, M. S., Charoensawan, V., Cortijo, S., Lai, X., Stöckle, D., Zubieta, C., Jaeger, K. E. & Wigge, P. A. The Evening Complex coordinates environmental and endogenous signals in Arabidopsis. Nat. Plants 3, 17087 (2017).
14. Graf, A., Coman, D., Uhrig, R. G., Walsh, S., Flis, A., Stitt, M. & Gruissem, W. Parallel analysis of Arabidopsis circadian clock mutants reveals different scales of transcriptome and proteome regulation. Open Biol. 7, 160333 (2017).
15. Dubois, M., Claeys, H., Van den Broeck, L. & Inzé, D. Time of day determines Arabidopsis transcriptome and growth dynamics under mild drought. Plant Cell Environ. 40, 180–189 (2017).
16. Edwards, K. D., Anderson, P. E., Hall, A., Salathia, N. S., Locke, J. C. W., Lynn, J. R., Straume, M., Smith, J. Q. & Millar, A. J. FLOWERING LOCUS C Mediates Natural Variation in the High-Temperature Response of the Arabidopsis Circadian Clock. Plant Cell 18, 639–650 (2006).
17. Covington, M. F. & Harmer, S. L. The Circadian Clock Regulates Auxin Signaling and Responses in Arabidopsis. PLoS Biol. 5, e222 (2007).
18. Michael, T. P., Breton, G., Hazen, S. P., Priest, H., Mockler, T. C., Kay, S. A. & Chory, J. A Morning-Specific Phytohormone Gene Expression Program underlying Rhythmic Plant Growth. PLOS Biol. 6, e225 (2008).
19. Espinoza, C., Degenkolbe, T., Caldana, C., Zuther, E., Leisse, A., Willmitzer, L., Hincha, D. K. & Hannah, M. A. Interaction with Diurnal and Circadian Regulation Results in Dynamic Metabolic and Transcriptional Changes during Cold Acclimation in Arabidopsis. PLOS ONE 5, e14101 (2010).
20. Endo, M., Shimizu, H., Nohales, M. A., Araki, T. & Kay, S. A. Tissue-specific clocks in Arabidopsis show asymmetric coupling. Nature 515, 419–422 (2014).
21. Blair, E. J., Bonnot, T., Hummel, M., Hay, E., Marzolino, J. M., Quijada, I. A. & Nagel, D. H. Contribution of time of day and the circadian clock to the heat stress responsive transcriptome in Arabidopsis. Sci. Rep. 9, 4814 (2019).
22. Dubin, M. J., Zhang, P., Meng, D., Remigereau, M.-S., Osborne, E. J., Paolo Casale, F., Drewe, P., Kahles, A., Jean, G., Vilhjálmsson, B., Jagoda, J., Irez, S., Voronin, V., Song, Q., Long, Q., Rätsch, G., Stegle, O., Clark, R. M. & Nordborg, M. DNA methylation in Arabidopsis has a genetic basis and shows evidence of local adaptation. eLife 4, e05255 (2015).
