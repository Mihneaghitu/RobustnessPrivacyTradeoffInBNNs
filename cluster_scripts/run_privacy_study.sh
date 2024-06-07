cd ..
for i in `seq 1 2 21`
do
    if [ "$1" -eq "0" ]; then
        echo "Running MNIST"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python mnist_runner.py $i
    else
        echo "Running PNEUMONIA_MNIST"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python pneumonia_mnist_runner.py $i
    fi
done