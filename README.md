# Improved Variational Online Newton (IVON)

We provide code of the IVON optimizer to train deep neural networks, along with a usage guide and small-scale examples.

**Variational Learning is Effective for Large Deep Networks**  
*Y. Shen\*, N. Daheim\*, B. Cong, P. Nickl, G.M. Marconi, C. Bazan, R. Yokota, I. Gurevych, D. Cremers, M.E. Khan, T. Möllenhoff*\
Paper: https://arxiv.org/abs/2402.17641

## Installation of IVON

To install the IVON optimizer run:
`pip install ivon-opt` 

**Dependencies**

Install PyTorch as described [here](https://pytorch.org/get-started/locally/): `pip3 install torch --index-url https://download.pytorch.org/whl/cu118`

## Usage guide

**Training loop**

In the code snippet below we demonstrate the difference in the implementation of the training loop of the IVON optimizer compared to standard optimizers like SGD or Adam.
The standard setting for weight sampling during training is to use one MC sample (`train_samples=1`).

```diff
import torch
+import ivon

train_loader = torch.utils.data.DataLoader(train_dataset) 
test_loader = torch.utils.data.DataLoader(test_dataset) 
model = MLP()

-optimizer = torch.optim.Adam(model.parameters())
+optimizer = ivon.IVON(model.parameters())

for X, y in train_loader:

+    for _ in range(train_samples):
+       with optimizer.sampled_params(train=True)
            optimizer.zero_grad()
            logit = model(X)
            loss = torch.nn.CrossEntropyLoss(logit, y)
            loss.backward()

    optimizer.step()
```

**Prediction**

There are two different ways of using the variational posterior of IVON for prediction:

(1) Predicting at the mean of the variational posterior (fast)

```
for X, y in test_loader:
    logit = model(X)
    _, prediction = logit.max(1)
```
(2) Predicting with Bayesian model averaging obtained by drawing a total of `test_samples` weight samples from the variational posterior (slower, but better inference with multiple weight samples)

```
for X, y in test_loader:
    sampled_probs = []
    for i in range(test_samples):
        with optimizer.sampled_params():
            sampled_logit = model(X)
            sampled_probs.append(F.softmax(sampled_logit, dim=1))
    prob = torch.mean(torch.stack(sampled_probs), dim=0)
    _, prediction = prob.max(1)
```

## Examples

We include three Google Colab notebooks to demonstrate the usage of the IVON optimizers on small-scale problems.
1. [2-D Logistic Regression](https://colab.research.google.com/drive/1o2XFJA8UbCiAUEKbiGFsNCwuvhZdFFfg?usp=sharing)
    - SGD finds the mode of the weight posterior, while IVON converges to a region that is more robust to perturbation.
2. [1-D Regression](https://colab.research.google.com/drive/1GcCCRfiZ6u7OwkYS46LGIAQKLnGL8Ae7?usp=sharing)
    - IVON captures uncertainty in regions with little data. AdamW fails at this task.
3. [MNIST image classification](https://colab.research.google.com/drive/1Q6MdLxmvR5Q1I2NbVXLCgGTDuP1m79tV?usp=sharing)
    - We compare IVON to an SGD baseline.

## How to cite

```
@article{shen2024variational,
      title={Variational Learning is Effective for Large Deep Networks}, 
      author={Yuesong Shen and Nico Daheim and Bai Cong and Peter Nickl and Gian Maria Marconi and Clement Bazan and Rio Yokota and Iryna Gurevych and Daniel Cremers and Mohammad Emtiyaz Khan and Thomas Möllenhoff},
      journal={arXiv:2402.17641},
      year={2024},
      url={https://arxiv.org/abs/2402.17641}
}
```
