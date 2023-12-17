# NLP_Fewshot_project
## Description
The goal of this project is to utilize few shot learning for NER on the ChEMU dataset.


**ONLY TESTED ON A SLURM MACHINE. RUNNING ON A NON SLURM MACHINE MAY HAVE UNEXPECTED BEHAVIORS.
## Recreating the k=5 experiments:
1. Run ```sbatch train.sh``` until completion (You don't have to do this, because we have the best model saved in model_dumps already) **YOU HAVE TO CHANGE THE PATHS TO THE DATASETS IN THE SH FILE
2. Run ```sbatch test.sh``` until completion **YOU HAVE TO CHANGE THE PATHS TO THE DATASETS IN THE SH FILE

You can do the same with k=25, just replace the k argument in the sh files with 25. If you are only testing, you may want to specify our best model for model_path for 25-shot which will be in model_dumps
