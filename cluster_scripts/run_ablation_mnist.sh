
#!/bin/bash
cd ..
EPSILONS=(0.075 0.1 0.2 0.3 0.4 0.5)
TAU_GS=(0.05 0.1 0.15 0.2 0.25 0.3)
PRIOR_STDS=(6 9 12 15 18 21)
DSET_SIZES=(1 0.9 0.8 0.7 0.6 0.5)
MODEL_SIZES=(128 256 512 1024 2048)

for eps in ${EPSILONS[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python mnist_runner.py eps $eps
done
for tau in ${TAU_GS[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python mnist_runner.py tau $tau
done
for priorstd in ${PRIOR_STDS[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python mnist_runner.py prior $priorstd
done
for dsetsizes in ${DSET_SIZES[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python mnist_runner.py dset $dsetsizes
done
for modelsizes in ${MODEL_SIZES[@]}; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python mnist_runner.py model_size $modelsizes
done