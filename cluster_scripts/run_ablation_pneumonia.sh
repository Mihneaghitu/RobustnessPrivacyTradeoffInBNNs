#!/bin/bash
cd ..
EPSILONS=(0.01 0.05 0.07 0.09 0.11 0.13 0.15 0.17 0.19 0.21)
TAU_GS=(0.1 0.2 0.3 0.4 0.5 0.6)
PRIOR_STDS=(1 3 5 7 9 11)
DSET_SIZES=(1 0.9 0.8 0.7 0.6 0.5)
MODEL_SIZE_RATIOS=(0.25 0.5 1 2 4)

for eps in ${EPSILONS[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pneumonia_mnist_runner.py eps $eps
done
for tau in ${TAU_GS[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pneumonia_mnist_runner.py tau $tau
done
for priorstd in ${PRIOR_STDS[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pneumonia_mnist_runner.py prior $priorstd
done
for dsetsizes in ${DSET_SIZES[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pneumonia_mnist_runner.py dset $dsetsizes
done
for modelsizes in ${MODEL_SIZE_RATIOS[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pneumonia_mnist_runner.py model_size $modelsizes
done