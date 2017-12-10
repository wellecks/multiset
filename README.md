# multiset
Loss Functions for Multiset Prediction

### Running

1. Generate an MNIST-multi dataset

2. Run `train.py` with suitable cmd line arguments e.g:
```bash
python train.py --dataset-path data/mnist_multi_70000_min20_max50_4 --mnist-multi \
                --max-objects 4 --loss multiset_loss --use-cuda
```
Run `train.py -h` for cmd line argument details. 

Note that `--dataset-path` and `--max-objects` vary based on the MNIST Multi dataset used.

Choose the loss with `--loss`.

When using the sequential loss (`--loss ce_loss`), choose an ordering strategy with `--label-order`.
