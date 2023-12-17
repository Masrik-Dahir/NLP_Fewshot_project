# NLP_Fewshot_project
## Description
The goal of this project is to utilize few shot learning for NER on the ChEMU dataset. See main.py header comment for more information :)


**ONLY TESTED ON A SLURM MACHINE. RUNNING ON A NON SLURM MACHINE MAY HAVE UNEXPECTED BEHAVIORS.
## Training & testing from scratch k=5
1. replace data paths in train.sh to where your data is located (ALL OF THEM ARE NEEDED)
2. run ```sbatch train.sh``` to completion
3. check 5-shotlog.txt. It's in the project root directory. At the bottom there will be a printed line that says what model is the best (which epoch)
4. replace data paths in test.sh to where your data is located (ALL OF THEM ARE NEEDED)
5. replace model_dumps/5-shot-78.pth with model_dumps/5-shot-{best_epoch_num}.pth in test.sh
6. run ```sbatch test.sh``` to completion
7. Check out reports per task in project root directory. They will look sort of like the report format in the reports directory

## Testing using pretrained k=5 model
0. Get pretrained model from Char, put it in model_dumps
1. replace data paths in test.sh to where your data is located (ALL OF THEM ARE NEEDED)
2. run ```sbatch test.sh``` to completion
3. Check out reports per task in project root directory. They will look sort of like the report format in the reports directory

## Training & testing from scratch k=25
1. replace data paths in train.sh to where your data is located (ALL OF THEM ARE NEEDED)
2. run ```sbatch train.sh``` to completion
3. check 25-shotlog.txt. It's in the project root directory. At the bottom there will be a printed line that says what model is the best (which epoch)
4. replace data paths in test.sh to where your data is located (ALL OF THEM ARE NEEDED)
5. replace model_dumps/5-shot-78.pth with model_dumps/25-shot-{best_epoch_num}.pth in test.sh
6. run ```sbatch test.sh``` to completion
7. Check out reports per task in project root directory. They will look sort of like the report format in the reports directory

## Testing using pretrained k=25 model
0. Get pretrained model from Char, put it in model_dumps
1. replace data paths in test.sh to where your data is located (ALL OF THEM ARE NEEDED)
2. replace model_dumps/5-shot-78.pth with model_dumps/25-shot-12.pth in test.sh
3. replace k value with 25
4. run ```sbatch test.sh``` to completion
5. Check out reports per task in project root directory, They will look sort of like the report format in the reports directory
