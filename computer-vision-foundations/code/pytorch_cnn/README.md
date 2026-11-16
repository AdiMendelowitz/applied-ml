# ResNet-18 from scratch in PyTorch

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)

ResNet-18 written from `nn.Module` primitives and trained from scratch on CIFAR-10, reaching **93.43% test accuracy** in 50 epochs. The folder also holds a small CNN for MNIST that mirrors the NumPy network in [`cnn_scratch/`](../cnn_scratch/README.md).

![Training and test curves for ResNet-18 on CIFAR-10](resnet_cifar10_training.png)

## Results

| Metric | Value | Source |
|---|---:|---|
| Best test accuracy | 93.43% (epoch 46) | `training_history.json` |
| Test accuracy at the final epoch | 93.38% | `training_history.json` |
| Training accuracy at the final epoch | 99.40% | `training_history.json` |
| Test accuracy after epoch 1 | 35.93% | `training_history.json` |
| Trainable parameters | 11,173,962 | `resnet.py` |
| Epochs | 50 | `03_resnet_training.ipynb` |

The gap between training and test accuracy at the final epoch is 6.0 points. Test accuracy moves by 0.05 points between the best epoch and the last one, so the reported figure does not depend on picking a lucky checkpoint.

### Per-class accuracy

![Per-class accuracy](resnet_per_class.png)

### Sample predictions

![Sample predictions](resnet_predictions.png)

## Architecture

### ResNet-18 (`resnet.py`)

An implementation of [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He et al., 2016). No `torchvision.models` code is used.

```
Input (32×32×3) → Conv 3×3, stride 1 (no MaxPool) →
                  [BasicBlock ×2] (64 channels) →
                  [BasicBlock ×2] (128 channels) →
                  [BasicBlock ×2] (256 channels) →
                  [BasicBlock ×2] (512 channels) →
                  GlobalAvgPool → FC(512→10) → Output
```

> [!NOTE]
> The stem is adapted for CIFAR-10. The first convolution is 3×3 with stride 1 and no max-pool, in place of the 7×7 stride-2 convolution and max-pool used for 224×224 ImageNet inputs, which keeps spatial resolution on 32×32 images.

Each `BasicBlock` computes `out = F(x) + x`, where `F` is Conv → BatchNorm → ReLU → Conv → BatchNorm. The shortcut is the identity when shapes match and a 1×1 convolution when the channel count or stride changes.

The identity path is what lets the gradient reach early layers.

```
∂L/∂x = ∂L/∂y × (∂F/∂x + 1)
      = ∂L/∂y × ∂F/∂x + ∂L/∂y
```

Even when `∂F/∂x` is close to zero, the second term carries the gradient through unchanged.

### Basic CNN (`cnn_pytotch.py`)

A two-convolution network for MNIST with about 100K parameters, built to match the NumPy CNN layer for layer.

```
Input (28×28×1) → Conv2d(1→8, 3×3) → ReLU → MaxPool(2×2) →
                  Conv2d(8→16, 3×3) → ReLU → MaxPool(2×2) →
                  Flatten → Linear(784→128) → ReLU →
                  Linear(128→10) → Output
```

`compare_implementations.py` trains this network and the NumPy network on the same data and prints the differences in accuracy and loss.

## Training configuration

The 93.43% run is the notebook `03_resnet_training.ipynb`, with the settings below.

| Setting | Value |
|---|---|
| Optimiser | SGD, momentum 0.9, weight decay 5e-4 |
| Learning rate | 0.1, multiplied by 0.1 at epochs 20, 35 and 45 (`MultiStepLR`) |
| Epochs | 50 |
| Batch size | 128 |
| Augmentation | Random crop (32, padding 4), random horizontal flip |

The logged learning rate in `training_history.json` follows this schedule, ending at 0.0001.

## Usage

Shape and parameter checks for the two models:

```bash
python resnet.py
python cnn_pytotch.py
```

A short CPU training run, which defaults to 10 epochs with the learning rate reduced at epochs 5 and 8:

```bash
python train_cifar.py
```

The full 50-epoch run is in `03_resnet_training.ipynb`. Unit tests for ResNet-18 live in [`../tests/test_core.py`](../tests/test_core.py).

## Files

| File | Contents |
|---|---|
| `resnet.py` | `BasicBlock`, `_make_layer` and the `resnet18` factory |
| `cnn_pytotch.py` | Basic CNN for MNIST and its training function |
| `train_cifar.py` | CIFAR-10 training script with a `CONFIG` dict |
| `compare_implementations.py` | NumPy CNN against the PyTorch CNN on the same data |
| `03_resnet_training.ipynb` | The 50-epoch training run and its plots |
| `training_history.json` | Loss, accuracy and learning rate for every epoch |

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016, pp. 770-778. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML 2015, pp. 448-456. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

Part of the [applied-ml portfolio](../../../README.md).
