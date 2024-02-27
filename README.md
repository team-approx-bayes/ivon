# Improved Variational Online Newton (IVON)

We provide code of the IVON optimizer to train deep neural networks, along with a usage guide and small-scale examples.

**Variational Learning is Effective for Large Deep Networks**  
*Y. Shen\*, N. Daheim\*, B. Cong, P. Nickl, G.M. Marconi, C. Bazan, R. Yokota, I. Gurevych, D. Cremers, M.E. Khan, T. Möllenhoff*\
(ArXiv link coming soon)

## Installation of IVON

- To install the **ivon** package locally run `pip install -e .` in the current directory
- IVON will soon be added to the Python Package Index

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

Coming soon.

## How to cite

Coming soon.
