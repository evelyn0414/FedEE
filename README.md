environment for using the dataset:

```
cd FLamby
make install
conda activate flamby
```

downloading new datasets:

```
cd FLamby
cd flamby/datasets/fed_heart_disease/dataset_creation_scripts
python download.py --output-folder ./heart_disease_dataset
```

Main code for the evaluation is in `PFL/my_main.py`. The uncertainty evaluation is in `PFL/uncertainty.py`. Main code for early exit ensembles is in `PFL/earlyexit.py`.

Experiment code can be run by simply executing our the main python file,

````bash
python my_main.py --alg fedavg
````

with only the strategy to be parsed and other parameters defined in the code.

To switch between datasets, only the import statements and definitions in the beginning need to be changed (see the annotation).

The trained models and training logs of server&client performance in each round are stored in `./cks/`. However PAMAP2 and ISIC2019 model checkpoints are not included in the submission as they are too large.

The config of Heart-Disease and ISIC2019 is imported from the FLamby benchmark. That of PAMAP2 is defined in `har_config.py`.

