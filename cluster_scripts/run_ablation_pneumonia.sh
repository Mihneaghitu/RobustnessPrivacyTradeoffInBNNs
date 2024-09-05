#!/bin/bash
cd ..
EPSILONS=(0.01 0.05 0.07 0.09 0.11 0.13 0.15 0.17 0.19 0.21)
TAU_GS=(0.1 0.2 0.3 0.4 0.5 0.6)
PRIOR_STDS=(1 3 5 7 9 11)

# for eps in ${EPSILONS[@]}; do
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pneumonia_mnist_runner.py eps $eps
# done
for tau in ${TAU_GS[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pneumonia_mnist_runner.py tau $tau
done
for priorstd in ${PRIOR_STDS[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pneumonia_mnist_runner.py prior $priorstd
done