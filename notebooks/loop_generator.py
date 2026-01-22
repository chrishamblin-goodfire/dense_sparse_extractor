#%%
import numpy as np
import matplotlib.pyplot as plt

"""
Parameterized 28x28 loop image generator.

- Background: black (0)
- Loop: bright ring with configurable geometry
"""


def _stable_superellipse_radius(xu: np.ndarray, yv: np.ndarray, p_exp: float, eps: float = 1e-12) -> np.ndarray:
    """
    Stable computation of r = (xu^p + yv^p)^(1/p) for p>0, using log-sum-exp.
    xu, yv must be nonnegative arrays.
    """
    p_exp = float(p_exp)
    if not (p_exp > 0):
        raise ValueError(f"p_exp must be > 0, got {p_exp}")

    # log(x^p) = p*log(x). Use eps to avoid log(0).
    a = p_exp * np.log(xu + eps)
    b = p_exp * np.log(yv + eps)
    m = np.maximum(a, b)
    # log(xu^p + yv^p) = m + log(exp(a-m) + exp(b-m))
    log_sum = m + np.log(np.exp(a - m) + np.exp(b - m))
    return np.exp(log_sum / p_exp)


def generate_loop_image(
    *,
    size: float,
    tx: float,
    ty: float,
    aspect: float,
    rotation: float,
    thickness: float,
    brightness: float,
    p_shape: float,
    image_size: int = 28,
    antialias: bool = True,
) -> np.ndarray:
    """
    Generate a single (image_size x image_size) grayscale image with a loop (ring).

    Parameters
    - size: minor-axis radius in pixels (can be any positive float)
    - tx, ty: translation in pixels (can move off-screen)
    - aspect: major/minor axis ratio (>= 1)
    - rotation: radians
    - thickness: ring thickness in pixels
    - brightness: loop intensity in [0.5, 1.0] (background is 0)
    - p_shape: log-scale "p-norm-ish" control:
        exponent = 2 * exp(p_shape)
        p_shape=0 -> exponent=2 (ellipse)
        p_shape large -> exponent -> inf (rectangle-ish)
        p_shape negative -> exponent < 2, can go < 1 (concave superellipse)
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if size <= 0:
        raise ValueError("size must be > 0")
    if thickness <= 0:
        raise ValueError("thickness must be > 0")
    if aspect < 1:
        raise ValueError("aspect must be >= 1 (major/minor ratio)")

    brightness = float(np.clip(brightness, 0.5, 1.0))

    # Semi-axes in pixels
    b = float(size)  # minor
    a = float(size * aspect)  # major

    # Coordinate grid in pixel units, centered at image center.
    c = (image_size - 1) / 2.0
    xs = np.arange(image_size, dtype=np.float32)
    ys = np.arange(image_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    dx = (xx - c) - float(tx)
    dy = (yy - c) - float(ty)

    # Rotate coordinates by -rotation to rotate the shape by +rotation.
    ct = float(np.cos(rotation))
    st = float(np.sin(rotation))
    u = ct * dx + st * dy
    v = -st * dx + ct * dy

    xu = np.abs(u) / a
    yv = np.abs(v) / b

    p_exp = 2.0 * float(np.exp(p_shape))
    p_exp = float(np.clip(p_exp, 1e-3, 1e6))
    r = _stable_superellipse_radius(xu, yv, p_exp=p_exp)  # boundary at r=1

    # Convert thickness (pixels) into a band thickness in r-units (approx).
    mean_radius = 0.5 * (a + b)
    half_band = 0.5 * float(thickness) / mean_radius
    # Anti-alias band edges across ~1 pixel in r-units.
    edge = (0.5 / mean_radius) if antialias else 1e-9

    dist = np.abs(r - 1.0)
    alpha = np.clip((half_band - dist) / edge + 0.5, 0.0, 1.0)
    img = (brightness * alpha).astype(np.float32)
    return img


#%%
# Random sampling demo: show a small grid of loop images + their parameters.

rng = np.random.default_rng(1)

n_show = 12
ncols = 6
nrows = int(np.ceil(n_show / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 1.9 * nrows))
axes = np.asarray(axes).reshape(nrows, ncols)

samples = []
for i in range(n_show):
    # Size in pixels (minor axis radius). Small sizes will look like tiny rings; large can go off-screen.
    size = float(rng.uniform(3.0, 6.0))
    # Translation can move off-screen.
    tx = float(rng.uniform(-10.0, 10.0))
    ty = float(rng.uniform(-10.0, 10.0))
    # Major/minor ratio (>=1).
    aspect = float(rng.uniform(1.0, 3.0))
    rotation = float(rng.uniform(0.0, 2 * np.pi))
    thickness = float(rng.uniform(1.0, 4.0))
    brightness = float(rng.uniform(0.5, 1.0))
    p_shape = float(rng.uniform(0.0, 2.0))

    img = generate_loop_image(
        size=size,
        tx=tx,
        ty=ty,
        aspect=aspect,
        rotation=rotation,
        thickness=thickness,
        brightness=brightness,
        p_shape=p_shape,
        image_size=28,
        antialias=True,
    )

    samples.append(
        {
            "size": size,
            "tx": tx,
            "ty": ty,
            "aspect": aspect,
            "rotation": rotation,
            "thickness": thickness,
            "brightness": brightness,
            "p_shape": p_shape,
        }
    )

    r, c = divmod(i, ncols)
    ax = axes[r, c]
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "size={:.1f} asp={:.2f}\n(tx,ty)=({:.1f},{:.1f})\nrot={:.2f} th={:.1f} b={:.2f}\np={:.2f}".format(
            size, aspect, tx, ty, rotation, thickness, brightness, p_shape
        ),
        fontsize=8,
    )

# Hide unused axes
for j in range(n_show, nrows * ncols):
    r, c = divmod(j, ncols)
    axes[r, c].axis("off")

plt.tight_layout()
plt.show()

for i, s in enumerate(samples):
    print(f"[{i:02d}] {s}")

# %%
