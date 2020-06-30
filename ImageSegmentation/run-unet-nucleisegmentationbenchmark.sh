#! /bin/bash
#$ -N Unet-nucleisegmentationbenchmark
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

function train() {
    #path='/share/apps/DeepLearning/Datos/nuclei-segmentation-in-microscope-cell-images/Dataset_64x64_5_folds'
    path='/home/andres.ziemecki/nuclei-segmentation-in-microscope-cell-images/Dataset_64x64_5_folds'
    #path='/home/andres/Documents/TESIS/nuclei-segmentation-in-microscope-cell-images/Dataset_64x64_divided_in_5'
    saveto='results-unet-nucleisegmentationbenchmark'
    nfilters='64'
    batch_size='128'
    epochs='200'
    input_shape='64x64'
    gpus='1'
    shuffle='True'
    seed='0'
    lr='1e-4'
    fold=$1 #From 1 to 5
    dataset_train='nucleisegmentationbenchmark'
    dataset_test='nucleisegmentationbenchmark' 

    echo "--------------------------------------------------------------------"
    echo "------------------------ Iniciando Fold $1 --------------------------"
    echo "--------------------------------------------------------------------"
    echo "Command: python train_unet.py --path=$path --saveto=$saveto --nfilters=$nfilters --batch_size=$batch_size --epochs=$epochs --input_shape=$input_shape --gpus=$gpus --shuffle=$shuffle --seed=$seed --lr=$lr --fold=$fold --dataset_train=$dataset_train --dataset_test=$dataset_test"
    echo "--------------------------------------------------------------------"
    python train_unet.py --path=$path --saveto=$saveto --nfilters=$nfilters --batch_size=$batch_size --epochs=$epochs --input_shape=$input_shape --gpus=$gpus --shuffle=$shuffle --seed=$seed --lr=$lr --fold=$fold --dataset_train=$dataset_train --dataset_test=$dataset_test
}

train 1
train 2
train 3
train 4
train 5
