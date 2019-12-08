#! /bin/bash
#$ -N c-6-4
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
##  pido la cola gpu.q  ---- Esto es un comentario
#$ -q gpu@compute-6-4.local
## pido una placa  ---- Esto es un comentario
## Cantidad de gpu que voy a usar:
#$ -l gpu=1
## Memoria RAM que voy a usar:
#$ -l memoria_a_usar=1G
#
# Load gpu drivers and conda  -- Esto es un comentario
module load miniconda

source activate deep_learning

# Execute the script  ---- Esto es un comentario
hostname

## Esto es para que solamente se ejecute en la gpu 0 si es 1 y 1 si es 0
## CUDA_VISIBLE_DEVICES=1 python testkeras.py
path_train='/home/andres/Documents/DataScience/DataSets/DigitRecognizer/train.csv'
path_test='/home/andres/Documents/DataScience/DataSets/DigitRecognizer/test.csv'
saveto='./results'
nfilters='64'
batch_size='128'
epochs='1'
gpus='1'

python train.py --path_train=$path_train --path_test=$path_test  --saveto=$saveto --nfilters=$nfilters --batch_size=$batch_size --epochs=$epochs --gpus=$gpus