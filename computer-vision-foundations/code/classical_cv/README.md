# Classical computer vision in NumPy

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-green.svg)](https://numpy.org/)

Gaussian blur, Sobel and Canny edge detection, and affine and perspective warps written in NumPy alone and compared side by side with OpenCV. Where the output differs from OpenCV, the difference and its cause are documented below.

![Sobel gradients and magnitude on a test image](outputs/sobel_visualization.png)

## Comparison with OpenCV

| Algorithm | Implementation | Result against OpenCV |
|---|---|---|
| Gaussian blur | 2D kernel convolution | Visually identical in the side-by-side output |
| Sobel | Gx and Gy gradient operators | Gradient structure as expected on all three test images |
| Canny | Five-stage pipeline | Edge positions agree, edges are thicker, and a smooth gradient produces false positives |
| Rotation | Inverse mapping | Agrees in the side-by-side output |
| Affine warp | 2×3 matrix, inverse mapping | Agrees in the side-by-side output |
| Perspective warp | 3×3 homography, inverse mapping | Agrees in the side-by-side output |

The comparisons are images in [`outputs/`](outputs/). The numerical checks are the unit tests in `test_unit.py`, which cover kernel normalisation, gradient direction on synthetic edges, threshold behaviour, transform matrices and identity warps.

## Gaussian blur

![Gaussian blur](outputs/gaussian_test.png)

A 2D Gaussian kernel with σ=3, normalised to sum to one so that brightness is preserved.

```python
# G(x,y) = (1/2πσ²) exp(-(x²+y²)/2σ²)
kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
kernel /= kernel.sum()
```

The inner grey square blurs more slowly than the outer white one, which shows how σ sets the blur radius relative to feature size. The blur is the same in every direction, as a correct 2D kernel requires.

![Blur on the checkerboard](outputs/output_checkerboard.png)
![Blur on the circle](outputs/output_circle.png)
![Blur on the gradient](outputs/output_gradient.png)

## Sobel edge detection

Convolution with the 3×3 Sobel kernels for Gx and Gy, then magnitude √(Gx² + Gy²).

- Gx responds to vertical boundaries and Gy to horizontal ones.
- A 45° edge gives equal Gx and Gy contributions, and both peak together at a corner.
- The response is three pixels wide, which is what a 3×3 kernel produces.

| Checkerboard | Circle | Horizontal gradient |
|---|---|---|
| ![Sobel on the checkerboard](outputs/output_sobel_checkerboard.png) | ![Sobel on the circle](outputs/output_sobel_circle.png) | ![Sobel on the gradient](outputs/output_sobel_gradient.png) |
| Symmetric banding along the diagonal stripes | Gx dominates left and right, Gy top and bottom | Uniform Gx and near-zero Gy |

## Canny edge detection

The pipeline is Gaussian blur, Sobel gradients, non-maximum suppression, double thresholding and hysteresis. Each test image is compared with `cv2.Canny()`.

### Checkerboard

![Canny on the checkerboard](outputs/canny_edge_comparisoncheckerboard.png)

This implementation marks both edges of each stripe as thick bands, where OpenCV returns edges one pixel wide. The edge positions agree. The non-maximum suppression here keeps several adjacent pixels on diagonal edges, and OpenCV keeps only the local maximum.

### Circle

![Canny on the circle](outputs/canny_edge_comparisoncircle.png)

Both versions find the circle. This one leaves thick blobs at the top, bottom, left and right, where the gradient angle crosses from one quantisation bin to the next and more than one pixel survives suppression.

### Smooth gradient

![Canny on the gradient](outputs/canny_edge_comparisongradient.png)

> [!WARNING]
> This implementation reports false edges across a smooth gradient, and OpenCV correctly returns none. The thresholds are ratios of the image's own maximum gradient, so a uniformly low gradient still clears them. An absolute or automatically chosen threshold, such as Otsu's method, would remove these.

## Geometric transformations

Every warp uses inverse mapping. For each output pixel the source coordinate is computed and sampled with bilinear interpolation, which fills every output pixel. Forward mapping would leave holes.

### Rotation by 45°

![Rotation](outputs/rotation_test.png)

```python
R = [[cos(θ), -sin(θ)],
     [sin(θ),  cos(θ)]]
```

### Affine warp

![Affine warp](outputs/affine_test.png)

```
[x']   [a  b][x]   [tx]
[y'] = [c  d][y] + [ty]
```

Parallel lines stay parallel, so the rectangle becomes a parallelogram.

### Perspective warp

![Perspective warp](outputs/perspective_test.png)

```
[x']   [h00 h01 h02][x]
[y'] = [h10 h11 h12][y]  (then divide by w)
[w ]   [h20 h21  1 ][1]
```

Parallel lines are not preserved. The top edge comes out shorter than the bottom edge, as if the rectangle were viewed from below.

## Implementation notes

### Cross-correlation and convolution

`filters.py` keeps the two operations separate. `correlated2d` is the sliding-window engine, and `convolve2d` flips the kernel and calls it. Both take a `padding` argument of `"zero"`, `"replicate"` or `"mirror"`.

```python
def correlated2d(image: np.ndarray, kernel: np.ndarray, padding: str = "zero") -> np.ndarray:
    """2D cross-correlation (no kernel flip). Core sliding-window engine."""
    h, w = image.shape
    kh, kw = kernel.shape
    padded = _pad_image(image, pad_h=kh // 2, pad_w=kw // 2, mode=padding)
    output = np.zeros_like(image, dtype="float32")
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return output

def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: str = "zero") -> np.ndarray:
    """True convolution = cross-correlation with 180°-flipped kernel."""
    return correlated2d(image, np.flip(kernel), padding)
```

The nested loop costs O(H × W × K²). It is written for clarity, and separable kernels or FFT-based convolution would be the choice for speed.

### Non-maximum suppression

`edge_detection.py` assigns each gradient angle in [0°, 180°) to one of four directions (0°, 45°, 90°, 135°), compares each pixel with its two neighbours along that direction and suppresses it unless it is the local maximum.

### Warps

`transformers.py` holds `warp_affine` (2×3 matrix) and `warp_perspective` (3×3 homography). Both map a meshgrid of output coordinates back to source coordinates in one matrix multiply and then call `_bilinear_interpolate`. The perspective version divides by the homogeneous coordinate first.

```python
def warp_perspective(image: np.ndarray, matrix: np.ndarray,
                     output_shape: Tuple[int, int]) -> np.ndarray:
    """Inverse perspective mapping with homogeneous divide and bilinear interpolation."""
    height, width = output_shape
    H_inv = np.linalg.inv(matrix)

    y_coords, x_coords = np.mgrid[0:height, 0:width]
    coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones(height * width)])
    hom = H_inv @ coords
    src_x = (hom[0] / hom[2]).reshape(height, width)
    src_y = (hom[1] / hom[2]).reshape(height, width)

    return _bilinear_interpolate(image, src_x, src_y)
```

## Usage

```bash
pip install numpy opencv-python matplotlib

python filters.py             # Gaussian and Sobel figures
python test_filters.py        # blur, Sobel and Canny comparisons with OpenCV
python test_transformers.py   # rotation, affine and perspective comparisons
python -m pytest test_unit.py -v
```

The figures are written to `outputs/`.

## Files

| File | Contents |
|---|---|
| `filters.py` | Padding, cross-correlation, convolution, Gaussian kernel and blur, Sobel |
| `edge_detection.py` | Non-maximum suppression, double threshold, hysteresis, Canny |
| `transformers.py` | Affine and perspective matrices, warps, bilinear interpolation, rotate, resize |
| `test_filters.py`, `test_transformers.py` | Comparison figures against OpenCV |
| `test_unit.py` | Unit tests |

## References

- Canny, J. (1986). *A Computational Approach to Edge Detection.* IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6), 679-698.
- Sobel, I., & Feldman, G. (1968). *A 3×3 Isotropic Gradient Operator for Image Processing.* Talk at the Stanford Artificial Intelligence Laboratory.
- Gonzalez, R. C., & Woods, R. E. (2017). *Digital Image Processing* (4th ed.). Pearson.
- Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.

Part of the [applied-ml portfolio](../../../README.md).
