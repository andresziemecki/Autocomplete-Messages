#! /bin/bash
#$ -N everything-gpu-6-4
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
# /home/andres/Documents/TESIS/nuclei-segmentation-in-microscope-cell-images/Dataset_64x64_divided_in_5
# '/share/apps/DeepLearning/Datos/nuclei-segmentation-in-microscope-cell-images/Dataset_64x64_5_folds'
# /home/andres.ziemecki/nuclei-segmentation-in-microscope-cell-images/Dataset_64x64_5_folds
function train() {
    path='/home/andres.ziemecki/nuclei-segmentation-in-microscope-cell-images/Dataset_64x64_5_folds'
    saveto=''
    nfilters='64'
    batch_size='128'
    epochs='100'
    input_shape='64x64'
    gpus='1'
    shuffle='True'
    seed='0'
    lr='1e-3'
    fold=$1 # From 1 to 5

    echo "--------------------------------------------------------------------"
    echo "------------------------ Iniciando Fold $1 --------------------------"
    echo "--------------------------------------------------------------------"
    echo "Command: python train_encoder_and_acnn.py --path=$path --saveto=$saveto --nfilters=$nfilters --batch_size=$batch_size --epochs=$epochs --input_shape=$input_shape --gpus=$gpus --shuffle=$shuffle --seed=$seed --lr=$lr --fold=$fold"
    echo "--------------------------------------------------------------------"

    python train_encoder_and_acnn.py --path=$path --saveto=$saveto --nfilters=$nfilters --batch_size=$batch_size --epochs=$epochs --input_shape=$input_shape --gpus=$gpus --shuffle=$shuffle --seed=$seed --lr=$lr --fold=$fold
}

train 1
train 2
train 3
train 4
train 5

