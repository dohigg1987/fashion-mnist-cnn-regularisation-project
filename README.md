# Fashion-MNIST CNN Regularisation Study

A PyTorch convolutional neural network is trained on Fashion-MNIST (70,000 real 28x28 greyscale images of clothing, 10 classes). The project compares a baseline model with one that adds dropout (0.3) and L2 weight decay (0.0001) as a single regularisation change, keeping architecture, optimiser, data order, seed and epochs identical.

## Contents

- `deep_learning.ipynb`: the complete, executed notebook (data, baseline, experiment, evaluation, error analysis, summary)
- `Deep_Learning_Systems_Analysis_Report.pdf`: the written analysis report
- `requirements.txt`: pinned Python dependencies
- `figures/`: figures written by the notebook when it runs (also embedded in the notebook outputs and included in the zip version)
- `data/`: dataset access instructions (the notebook downloads the data and verifies checksums)

## Reproduce

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute deep_learning.ipynb
```

Training uses the CPU and takes about 25 minutes for six epochs per model and the three-seed check.

## Headline result

Regularisation cut the generalisation gap from 0.0214 to 0.0086 but test accuracy fell from 0.9146 to 0.9036 over six epochs, and a three-seed check showed the same direction. The loss was concentrated in the Shirt, Coat and T-shirt/top classes.

## Data source

Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747.
