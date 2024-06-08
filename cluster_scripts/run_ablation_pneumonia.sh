cd ..
EPSILONS=("0.01" "0.05" "0.07" "0.09" "0.11" "0.13" "0.15" "0.17" "0.19" "0.21")
CLIPS=("0.5" "0.5" "0.3" "0.2" "0.1" "0.05" "0.025" "0.01")

for eps in ${EPSILONS[@]}; do
    python pneumonia_mnist_runner.py eps $eps
done
for bg in ${CLIPS[@]}; do
    python pneumonia_mnist_runner.py dp $bg
done