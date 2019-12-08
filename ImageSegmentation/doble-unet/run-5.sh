#! /bin/bash
#$ -N c-6-5
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -V
##  pido la cola gpu.q  ---- Esto es un comentario
#$ -q gpu@compute-6-5.local
## pido una placa  ---- Esto es un comentario
## Cantidad de gpu que voy a usar:
#$ -l gpu=2
## Memoria RAM que voy a usar:
#$ -l memoria_a_usar=1G
#
# Load gpu drivers and conda  -- Esto es un comentario
module load miniconda

source activate tensorflow

# Execute the script  ---- Esto es un comentario
hostname

## Esto es para que solamente se ejecute en la gpu 0 si es 1 y 1 si es 0
## CUDA_VISIBLE_DEVICES=1 python testkeras.py

python train.py --path=/home/andres/Documents/TESIS/nuclei-segmentation-in-microscope-cell-images/Dataset_64x64_divided_in_5 --saveto= --nfilters=64 --batch_size=128 --epochs=0 --input_shape=64x64 --gpus=1 --shuffle=True --seed=0 --lr=1e-4 --fold=1
