"""
Finite-difference gradient checks for the NumPy CNN in cnn_scratch/.

Every analytical gradient produced by a backward pass is compared with a
central-difference estimate of the same derivative, in float64. A layer
passes when the relative error, ||a - n|| / (||a|| + ||n||), is below 1e-6.

The last two tests check the loss and the whole network together. They are
the ones that would have caught the cross-entropy gradient being passed
through the Softmax layer's Jacobian a second time.

Run from the project root:
    python -m pytest computer-vision-foundations/code/tests/test_gradcheck.py -v
"""

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "cnn_scratch"))

from layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU, Softmax  # noqa: E402
from network import CrossEntropyLoss, Network  # noqa: E402

EPS = 1e-6
TOL = 1e-6


def _rel_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    assert analytical.shape == numerical.shape
    assert np.linalg.norm(numerical) > 1e-8, "a zero gradient would pass vacuously"
    denom = np.linalg.norm(analytical) + np.linalg.norm(numerical)
    return float(np.linalg.norm(analytical - numerical) / max(denom, 1e-12))


def _numerical_grad(f: Callable[[], float], x: np.ndarray) -> np.ndarray:
    """Central differences of the scalar f() with respect to x, perturbing x in place."""
    grad = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(x.shape):
        original = x[idx]
        x[idx] = original + EPS
        f_plus = f()
        x[idx] = original - EPS
        f_minus = f()
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * EPS)
    return grad


def _check_layer(layer, x: np.ndarray, rng: np.random.Generator) -> None:
    """Check d(sum(out * R))/d(input) and d/d(params) for a random projection R."""
    out = layer.forward(x)
    projection = rng.standard_normal(out.shape)
    analytical_input = layer.backward(projection).copy()
    analytical_params = [grad.copy() for _, grad in layer.get_params()]

    def loss() -> float:
        return float(np.sum(layer.forward(x) * projection))

    assert _rel_error(analytical_input, _numerical_grad(loss, x)) < TOL
    for (param, _), analytical in zip(layer.get_params(), analytical_params):
        assert _rel_error(analytical, _numerical_grad(loss, param)) < TOL


@pytest.mark.parametrize(
    "in_ch, out_ch, kernel, stride, padding, size",
    [(2, 3, 3, 1, 1, 5), (2, 2, 3, 2, 0, 7)],
    ids=["pad1-stride1", "pad0-stride2"],
)
def test_conv2d(in_ch: int, out_ch: int, kernel: int, stride: int, padding: int, size: int) -> None:
    np.random.seed(0)
    rng = np.random.default_rng(0)
    layer = Conv2D(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
    layer.bias[...] = rng.standard_normal(layer.bias.shape)
    x = rng.standard_normal((2, in_ch, size, size))
    _check_layer(layer, x, rng)


def test_maxpool2d() -> None:
    rng = np.random.default_rng(1)
    # Distinct values 0.1 apart, so no window has a near-tie that a perturbation of EPS could flip
    x = rng.permutation(2 * 2 * 4 * 4).reshape(2, 2, 4, 4).astype(np.float64) * 0.1
    _check_layer(MaxPool2D(pool_size=2), x, rng)


def test_dense() -> None:
    np.random.seed(2)
    rng = np.random.default_rng(2)
    layer = Dense(6, 5)
    layer.bias[...] = rng.standard_normal(layer.bias.shape)
    _check_layer(layer, rng.standard_normal((4, 6)), rng)


def test_relu() -> None:
    rng = np.random.default_rng(3)
    # Magnitudes of at least 0.1 keep every input away from the kink at zero
    x = rng.uniform(0.1, 1.0, (3, 7)) * rng.choice([-1.0, 1.0], (3, 7))
    _check_layer(ReLU(), x, rng)


def test_softmax() -> None:
    rng = np.random.default_rng(4)
    _check_layer(Softmax(), rng.standard_normal((3, 5)), rng)


def test_softmax_cross_entropy_logit_gradient() -> None:
    """Softmax followed by the loss must give (p - y) / batch for the logits."""
    rng = np.random.default_rng(5)
    logits = rng.standard_normal((4, 6))
    labels = np.array([0, 3, 5, 1])
    softmax, loss_fn = Softmax(), CrossEntropyLoss()

    probs = softmax.forward(logits)
    loss_fn.forward(probs, labels)
    analytical = softmax.backward(loss_fn.backward())

    one_hot = np.eye(6)[labels]
    expected = (probs - one_hot) / len(labels)
    np.testing.assert_allclose(analytical, expected, rtol=1e-10, atol=1e-12)

    def loss() -> float:
        return float(loss_fn.forward(softmax.forward(logits), labels))

    assert _rel_error(analytical, _numerical_grad(loss, logits)) < TOL


def test_network_end_to_end() -> None:
    """Every parameter gradient of a small Conv-ReLU-Pool-Dense-Softmax network under cross-entropy."""
    np.random.seed(6)
    rng = np.random.default_rng(6)
    model = Network(
        [
            Conv2D(1, 2, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(2 * 3 * 3, 4),
            Softmax(),
        ]
    )
    x = rng.standard_normal((2, 1, 6, 6))
    labels = np.array([1, 3])
    loss_fn = CrossEntropyLoss()

    loss_fn.forward(model.forward(x), labels)
    model.backward(loss_fn.backward())
    analytical = [grad.copy() for _, grad in model.get_params()]

    def loss() -> float:
        return float(loss_fn.forward(model.forward(x), labels))

    params = [param for param, _ in model.get_params()]
    assert len(params) == len(analytical) == 4
    for param, grad in zip(params, analytical):
        assert _rel_error(grad, _numerical_grad(loss, param)) < TOL
