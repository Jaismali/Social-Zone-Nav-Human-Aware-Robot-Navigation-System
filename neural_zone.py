"""
neural_zone.py
==============
Neural network models for data-driven social zone learning.

Implemented entirely in NumPy — no PyTorch, TensorFlow, or CUDA required.
Weights are persisted as .npy files and are compatible with any platform
that can run NumPy >= 1.21.

This module provides three classes:

  ZoneLearnerNet
      A shallow MLP trained on inter-agent trajectory pairs that predicts
      whether a given relative position is socially comfortable.  The learned
      boundary implicitly encodes the direction-dependent structure observed
      in real pedestrian data without requiring the hand-crafted zone
      geometry of social_zone.py.

  SocialCostCNN
      A three-layer fully-convolutional network that maps a 2-channel
      occupancy+orientation grid to a normalised social cost map in a single
      forward pass.  Used by NeuralPlanner to replace the O(N_cells *
      N_humans) analytical cost computation with an O(1) CNN inference.

  NeuralSocialZone
      A drop-in replacement for SocialZone that delegates cost and violation
      queries to a trained ZoneLearnerNet.  Falls back to the analytical
      asymmetric zone when the model is unavailable or untrained.

Activation functions
~~~~~~~~~~~~~~~~~~~~
ReLU is used throughout the hidden layers.  The numerically stable sigmoid
implementation handles large positive and negative pre-activations separately
to avoid overflow in exp(-x) for large positive x and underflow for large
negative x (Press et al., 2007, Section 6.2).

Weight initialisation
~~~~~~~~~~~~~~~~~~~~~
ZoneLearnerNet uses Xavier (Glorot) initialisation (Glorot & Bengio, 2010),
which sets the initial weight variance to 2 / n_in.  This keeps the variance
of activations approximately constant across layers at initialisation,
accelerating early training.  SocialCostCNN uses Kaiming He initialisation
(He et al., 2015), which is preferred for layers followed by ReLU because it
accounts for the halving of the effective variance caused by the ReLU clamp.

Optimiser
~~~~~~~~~
Both networks use Adam (Kingma & Ba, 2015) with standard hyperparameters
(beta1=0.9, beta2=0.999, eps=1e-8).  Bias correction is applied at every
step to remove the initialisation bias of the first- and second-moment
estimates.  Adam was chosen over SGD with momentum because it adapts the
learning rate per parameter, which is important here: the orientation
features (sin, cos) have naturally bounded range [-1, 1] whereas the
displacement features (dx, dy) can range up to ±3 m, giving very different
gradient magnitudes across the four input dimensions.

References
----------
Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training
    deep feedforward neural networks. In Proc. AISTATS, pp. 249-256.
    Xavier initialisation; motivates our weight init for ZoneLearnerNet.

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers:
    Surpassing human-level performance on ImageNet classification. In Proc.
    ICCV, pp. 1026-1034.
    Kaiming initialisation for ReLU networks; used for SocialCostCNN.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
    In Proc. ICLR.  Adaptive optimiser used for all networks.

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based
    learning applied to document recognition. Proc. IEEE, 86(11), 2278-2324.
    Convolutional architecture motivation; same-padding preserves spatial dims.

Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007).
    Numerical Recipes: The Art of Scientific Computing (3rd ed.). Cambridge
    University Press.  Numerically stable sigmoid implementation.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning
    representations by back-propagating errors. Nature, 323, 533-536.
    Backpropagation algorithm implemented in backward() methods.
"""

import os
import numpy as np


# ===========================================================================
# Activation functions
# ===========================================================================

def relu(x):
    """
    Rectified Linear Unit activation: max(0, x).

    Applied element-wise.  ReLU is preferred over tanh or sigmoid in hidden
    layers because it does not saturate for large positive inputs, avoiding
    the vanishing-gradient problem in deep networks (He et al., 2015).

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    np.ndarray, same shape as x
    """
    return np.maximum(0.0, x)


def relu_grad(x):
    """
    Sub-gradient of ReLU with respect to its pre-activation input x.

    Returns 1 where x > 0, 0 elsewhere.  The sub-gradient at x = 0 is
    conventionally set to 0 (consistent with PyTorch and TensorFlow defaults).

    Parameters
    ----------
    x : np.ndarray
        Pre-activation values (before ReLU is applied).

    Returns
    -------
    np.ndarray, dtype float, same shape as x
    """
    return (x > 0).astype(np.float32)


def sigmoid(x):
    """
    Numerically stable logistic sigmoid: 1 / (1 + exp(-x)).

    Uses the two-branch formulation from Press et al. (2007) to avoid
    overflow in exp(-x) for large positive x and underflow for large
    negative x::

        x >= 0:  1 / (1 + exp(-x))      -- avoids exp overflow
        x <  0:  exp(x) / (1 + exp(x))  -- avoids underflow in denominator

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    np.ndarray in (0, 1), same shape as x
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def sigmoid_grad(s):
    """
    Gradient of sigmoid given its *output* s = sigma(x).

    d/dx sigma(x) = sigma(x) * (1 - sigma(x)) = s * (1 - s).

    Accepting the output rather than the input avoids recomputing exp,
    which is consistent with how the gradient is used in backward(): the
    sigmoid output is already cached from the forward pass.

    Parameters
    ----------
    s : np.ndarray
        Sigmoid output values in (0, 1).

    Returns
    -------
    np.ndarray, same shape as s
    """
    return s * (1.0 - s)


# ===========================================================================
# A. ZoneLearnerNet
# ===========================================================================

class ZoneLearnerNet:
    """
    Three-layer MLP that learns the asymmetric social comfort boundary.

    Architecture::

        Input (4) -> Linear(64) -> ReLU -> Linear(64) -> ReLU
                  -> Linear(1)  -> Sigmoid -> output in [0, 1]

    The network is trained on (dx, dy, sin_orient, cos_orient) feature
    vectors derived from real pedestrian trajectory pairs (ETH/UCY datasets).
    The binary label is 1 if the inter-agent distance exceeds the zone
    threshold (comfortable), 0 otherwise.

    Input feature design
    ~~~~~~~~~~~~~~~~~~~~
    Orientation is encoded as (sin theta, cos theta) rather than theta itself
    for two reasons: (1) it is continuous and bounded in [-1, 1], avoiding
    the 2pi discontinuity at ±180 deg that would require special handling;
    (2) it provides a two-dimensional representation of a circular quantity,
    which is more informative for linear layers than a single angle value.
    This encoding is standard in neural networks for periodic inputs
    (Vaswani et al., 2017, positional encodings).

    Hidden layer width
    ~~~~~~~~~~~~~~~~~~
    64 units per hidden layer was chosen to match the representational
    capacity needed for a 4-dimensional piecewise-linear boundary (the four
    angular zones of the asymmetric model).  Wider networks (128, 256)
    showed no improvement in validation accuracy on ETH/UCY data; narrower
    networks (16, 32) underfit the back-zone boundary where the comfort
    gradient changes sharply within a short angular range.

    Parameters
    ----------
    lr : float
        Adam learning rate.  Default 0.01 for ZoneLearnerNet; lower values
        (0.001) work but require more epochs.
    seed : int
        Random seed for weight initialisation.

    References
    ----------
    Vaswani, A. et al. (2017). Attention is all you need. In Proc. NeurIPS.
        Periodic encoding of angular quantities; motivates sin/cos features.
    """

    def __init__(self, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)

        # Xavier (Glorot) initialisation: std = sqrt(2 / n_in).
        # Biases are zero-initialised (standard practice).
        self.W1 = rng.normal(0, np.sqrt(2.0 / 4),  (64,  4)).astype(np.float32)
        self.b1 = np.zeros(64, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / 64), (64, 64)).astype(np.float32)
        self.b2 = np.zeros(64, dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2.0 / 64), (1,  64)).astype(np.float32)
        self.b3 = np.zeros(1,  dtype=np.float32)

        self.lr         = float(lr)
        self.is_trained = False   # set to True after train_zone_learner completes

        self._adam_init()

    # ------------------------------------------------------------------
    # Adam optimiser state
    # ------------------------------------------------------------------

    def _adam_init(self):
        """
        Initialise first- and second-moment estimates for Adam.

        All moment vectors are zero-initialised; the bias-correction terms
        in _adam_update() compensate for this at early steps.  The step
        counter self.t is also reset here so that load() correctly restarts
        Adam from a fresh optimiser state (we do not persist Adam state
        between training runs).
        """
        self.t                 = 0         # global step counter for bias correction
        self.beta1             = 0.9       # first-moment decay (standard)
        self.beta2             = 0.999     # second-moment decay (standard)
        self.eps               = 1e-8      # numerical stability constant
        params = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        self.m = {k: np.zeros_like(getattr(self, k)) for k in params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in params}

    def _adam_update(self, grads):
        """
        Apply one Adam parameter update step (Kingma & Ba, 2015).

        For each parameter theta with gradient g::

            m = beta1 * m + (1 - beta1) * g          # first moment
            v = beta2 * v + (1 - beta2) * g^2         # second moment
            m_hat = m / (1 - beta1^t)                  # bias-corrected
            v_hat = v / (1 - beta2^t)
            theta -= lr * m_hat / (sqrt(v_hat) + eps)

        Parameters
        ----------
        grads : dict
            Keys matching parameter names (W1, b1, ..., W3, b3).
            Values are numpy arrays with the same shape as the parameters.
        """
        self.t += 1
        for name, g in grads.items():
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * g ** 2
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)
            param  = getattr(self, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward(self, X):
        """
        Forward pass with cached intermediate activations for backprop.

        Caches pre-activations (_z1, _z2, _z3), post-activations (_a1, _a2,
        _out), and the input (_X) as instance attributes.  These are consumed
        by backward() and should not be modified between forward() and
        backward() calls.

        Parameters
        ----------
        X : np.ndarray, shape (N, 4)
            Batch of input feature vectors.

        Returns
        -------
        np.ndarray, shape (N,)
            Comfort probability for each sample, in [0, 1].
        """
        self._X   = X
        self._z1  = X @ self.W1.T + self.b1        # (N, 64)
        self._a1  = relu(self._z1)
        self._z2  = self._a1 @ self.W2.T + self.b2 # (N, 64)
        self._a2  = relu(self._z2)
        self._z3  = self._a2 @ self.W3.T + self.b3 # (N, 1)
        self._out = sigmoid(self._z3)               # (N, 1)
        return self._out.squeeze(1)                 # (N,)

    def backward(self, y_pred, y_true):
        """
        Backpropagation through the network using binary cross-entropy loss.

        Loss::

            L = -mean( y * log(y_hat + eps) + (1-y) * log(1 - y_hat + eps) )

        The combined BCE + sigmoid gradient simplifies to (y_pred - y_true) / N
        at the output layer, which is numerically cleaner than computing
        sigmoid_grad separately (Rumelhart et al., 1986).

        Parameters
        ----------
        y_pred : np.ndarray, shape (N,)
            Sigmoid output from forward().
        y_true : np.ndarray, shape (N,)
            Binary comfort labels (0 or 1).

        Returns
        -------
        loss : float
            Mean binary cross-entropy over the batch.
        grads : dict
            Gradient arrays for W1, b1, W2, b2, W3, b3.
        """
        N   = len(y_true)
        eps = 1e-7  # prevents log(0); smaller values risk underflow

        loss = -float(np.mean(
            y_true * np.log(y_pred + eps) +
            (1.0 - y_true) * np.log(1.0 - y_pred + eps)
        ))

        # Combined sigmoid + BCE gradient: dL/dz3 = (y_pred - y_true) / N
        dout = (y_pred - y_true).reshape(-1, 1) / N   # (N, 1)

        # Output layer gradients
        dW3 = dout.T @ self._a2           # (1, 64)
        db3 = dout.sum(axis=0)            # (1,)
        da2 = dout @ self.W3              # (N, 64)

        # Second hidden layer
        dz2 = da2 * relu_grad(self._z2)  # (N, 64)
        dW2 = dz2.T @ self._a1           # (64, 64)
        db2 = dz2.sum(axis=0)            # (64,)
        da1 = dz2 @ self.W2              # (N, 64)

        # First hidden layer
        dz1 = da1 * relu_grad(self._z1)  # (N, 64)
        dW1 = dz1.T @ self._X            # (64, 4)
        db1 = dz1.sum(axis=0)            # (64,)

        grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3,
        }
        return loss, grads

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Run inference without caching intermediate activations.

        Separate from forward() to avoid overwriting the cached activations
        needed for a pending backward() call during training.

        Parameters
        ----------
        X : np.ndarray, shape (N, 4)

        Returns
        -------
        np.ndarray, shape (N,)
            Comfort probabilities in [0, 1].
        """
        z1 = X @ self.W1.T + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2.T + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3.T + self.b3
        return sigmoid(z3).squeeze(1)

    def predict_single(self, dx, dy, orientation):
        """
        Convenience wrapper for single-sample inference.

        Parameters
        ----------
        dx, dy : float
            Displacement from human to robot (metres).
        orientation : float
            Human heading in radians.

        Returns
        -------
        float
            Comfort score in [0, 1].
        """
        X = np.array(
            [[dx, dy, np.sin(orientation), np.cos(orientation)]],
            dtype=np.float32,
        )
        return float(self.predict(X)[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path="zone_learner.npy"):
        """
        Persist weight matrices and training flag to a .npy file.

        Saves all six parameter arrays plus is_trained as a dict using
        numpy's allow_pickle format.  Adam moment vectors are NOT saved;
        fine-tuning from a checkpoint therefore restarts Adam from scratch,
        which is acceptable for the short training runs used here.

        Parameters
        ----------
        path : str
        """
        np.save(path, {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'is_trained': np.array([self.is_trained]),
        })
        print(f"  Saved ZoneLearnerNet weights -> {path}")

    def load(self, path="zone_learner.npy"):
        """
        Load weight matrices from a .npy file created by save().

        Adam state is re-initialised after loading (moment vectors reset to
        zero, step counter to 0) so that subsequent fine-tuning does not
        inherit stale moment estimates from the original training run.

        Parameters
        ----------
        path : str

        Raises
        ------
        FileNotFoundError
            If path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"ZoneLearnerNet weights not found: {path}")
        weights = np.load(path, allow_pickle=True).item()
        self.W1 = weights['W1'].astype(np.float32)
        self.b1 = weights['b1'].astype(np.float32)
        self.W2 = weights['W2'].astype(np.float32)
        self.b2 = weights['b2'].astype(np.float32)
        self.W3 = weights['W3'].astype(np.float32)
        self.b3 = weights['b3'].astype(np.float32)
        self.is_trained = bool(weights.get('is_trained', [False])[0])
        self._adam_init()   # reset optimiser state after loading
        print(f"  Loaded ZoneLearnerNet weights <- {path}")


# ===========================================================================
# B. SocialCostCNN
# ===========================================================================

def _forward_conv_layer(x, W, b, pad=1):
    """
    Single 2D convolution + bias for a (C_in, H, W) feature map.

    Uses reflect padding (same as PyTorch's 'reflect' mode) rather than zero
    padding to avoid introducing artificial low-cost boundary artefacts in
    the social cost map.  Reflect padding mirrors the nearest real cell values
    at the edges, which is more physically meaningful for a spatial cost field.

    Parameters
    ----------
    x : np.ndarray, shape (C_in, H, W)
    W : np.ndarray, shape (C_out, C_in, kH, kW)
    b : np.ndarray, shape (C_out,)
    pad : int

    Returns
    -------
    np.ndarray, shape (C_out, H, W)
        Same spatial dimensions as input (same-padding).
    """
    C_in, H, Ww = x.shape
    C_out, _, kH, kW = W.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
    out = np.zeros((C_out, H, Ww), dtype=np.float32)
    for co in range(C_out):
        for ci in range(C_in):
            for ki in range(kH):
                for kj in range(kW):
                    out[co] += W[co, ci, ki, kj] * xp[ci, ki:ki+H, kj:kj+Ww]
        out[co] += b[co]
    return out


class SocialCostCNN:
    """
    Fully-convolutional network mapping occupancy grids to social cost maps.

    Architecture (same-padding throughout; spatial dims preserved)::

        Input: (N, 2, 50, 50)
          -> Conv(2->16,  3x3, pad=1) -> ReLU
          -> Conv(16->32, 3x3, pad=1) -> ReLU
          -> Conv(32->1,  3x3, pad=1) -> ReLU
        Output: (N, 1, 50, 50)

    The fully-convolutional design (no fully-connected layers) is used so
    that the network learns spatially-equivariant features: a pedestrian at
    position (i, j) should produce the same cost pattern regardless of its
    absolute grid location.  This property does not hold for architectures
    with fully-connected layers, which learn position-specific weights
    (LeCun et al., 1998).

    Input encoding
    ~~~~~~~~~~~~~~
    Channel 0: binary occupancy grid — 1.0 at each cell containing a pedestrian.
    Channel 1: orientation field     — sin(heading) at occupied cells, 0 elsewhere.

    Encoding orientation as sin(heading) rather than the raw angle avoids
    the 2pi discontinuity and keeps the channel in [-1, 1].  Only sin is
    used (not the (sin, cos) pair from ZoneLearnerNet) to keep the input
    to two channels; the asymmetry front/back can be partially inferred from
    the cost pattern even with a single orientation channel.

    TODO: add a third channel for cos(heading) to provide an unambiguous
          orientation encoding; this may improve cost map accuracy for
          pedestrians facing exactly left or right (sin = +-1, cos = 0).

    Training target
    ~~~~~~~~~~~~~~~
    Ground-truth cost maps are computed analytically using SocialZone and
    normalised to [0, 1] per scenario by dividing by the scenario maximum.
    Normalisation stabilises MSE training; the absolute cost scale can be
    recovered by multiplying CNN output by the stored normalisation factor.

    Parameters
    ----------
    lr : float
        Adam learning rate.
    seed : int
    """

    GRID_H = 50
    GRID_W = 50

    def __init__(self, lr=0.001, seed=1):
        rng = np.random.default_rng(seed)

        # Kaiming He initialisation for ReLU layers: std = sqrt(2 / (C_in * kH * kW))
        self.W1 = rng.normal(0, np.sqrt(2.0 / (2  * 3 * 3)), (16,  2, 3, 3)).astype(np.float32)
        self.b1 = np.zeros(16, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / (16 * 3 * 3)), (32, 16, 3, 3)).astype(np.float32)
        self.b2 = np.zeros(32, dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2.0 / (32 * 3 * 3)), (1,  32, 3, 3)).astype(np.float32)
        self.b3 = np.zeros(1,  dtype=np.float32)

        self.lr         = float(lr)
        self.is_trained = False
        self._adam_init()

    def _adam_init(self):
        """Initialise Adam moment vectors for all six parameter tensors."""
        self.t     = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps   = 1e-8
        params = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        self.m = {k: np.zeros_like(getattr(self, k)) for k in params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in params}

    def _adam_update(self, grads):
        """Apply Adam update (identical formulation to ZoneLearnerNet._adam_update)."""
        self.t += 1
        for name, g in grads.items():
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * g ** 2
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)
            param  = getattr(self, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def _forward_single(self, x):
        """
        Forward pass for a single (2, H, W) input, caching all intermediates.

        Caches padded inputs and activations for use in train_step's backward
        pass.  Returns a 6-tuple so that backprop has everything it needs
        without recomputing any convolutions.

        Parameters
        ----------
        x : np.ndarray, shape (2, H, W)

        Returns
        -------
        out : np.ndarray, shape (1, H, W) — final cost map prediction
        a1  : post-ReLU activations from layer 1, shape (16, H, W)
        a2  : post-ReLU activations from layer 2, shape (32, H, W)
        xp  : padded input, shape (2, H+2, W+2)
        a1p : padded a1, shape (16, H+2, W+2)
        a2p : padded a2, shape (32, H+2, W+2)
        """
        H, W = x.shape[1], x.shape[2]
        pad  = 1

        # Layer 1
        xp = np.pad(x, ((0,0),(pad,pad),(pad,pad)), mode='reflect')
        a1 = np.zeros((16, H, W), dtype=np.float32)
        for co in range(16):
            for ci in range(2):
                for ki in range(3):
                    for kj in range(3):
                        a1[co] += self.W1[co,ci,ki,kj] * xp[ci,ki:ki+H,kj:kj+W]
            a1[co] += self.b1[co]
        a1 = relu(a1)

        # Layer 2
        a1p = np.pad(a1, ((0,0),(pad,pad),(pad,pad)), mode='reflect')
        a2  = np.zeros((32, H, W), dtype=np.float32)
        for co in range(32):
            for ci in range(16):
                for ki in range(3):
                    for kj in range(3):
                        a2[co] += self.W2[co,ci,ki,kj] * a1p[ci,ki:ki+H,kj:kj+W]
            a2[co] += self.b2[co]
        a2 = relu(a2)

        # Layer 3
        a2p = np.pad(a2, ((0,0),(pad,pad),(pad,pad)), mode='reflect')
        out = np.zeros((1, H, W), dtype=np.float32)
        for ci in range(32):
            for ki in range(3):
                for kj in range(3):
                    out[0] += self.W3[0,ci,ki,kj] * a2p[ci,ki:ki+H,kj:kj+W]
        out[0] += self.b3[0]
        out = relu(out)   # ReLU output: cost maps are non-negative

        return out, a1, a2, xp, a1p, a2p

    def predict(self, X):
        """
        Run inference on a batch of input grids.

        Parameters
        ----------
        X : np.ndarray, shape (N, 2, H, W) or (2, H, W)
            Occupancy + orientation channels.

        Returns
        -------
        np.ndarray, shape (N, H, W) or (H, W)
            Predicted normalised social cost maps.
        """
        squeeze = X.ndim == 3
        if squeeze:
            X = X[np.newaxis]

        N   = X.shape[0]
        out = np.zeros((N, self.GRID_H, self.GRID_W), dtype=np.float32)
        for i in range(N):
            result, *_ = self._forward_single(X[i])
            out[i] = result.squeeze(0)

        return out[0] if squeeze else out

    def train_step(self, X_batch, Y_batch):
        """
        One mini-batch forward + backward pass with Adam update.

        Loss is mean squared error between predicted and ground-truth
        normalised cost maps.  MSE is used rather than cross-entropy because
        the output is a continuous regression target (cost in [0, 1]), not a
        probability.

        The backward pass computes approximate gradients for the convolutional
        layers using the cached padded activations from _forward_single().
        The backpropagation through transposed convolution uses the same
        sliding-window loop structure as the forward pass, which is correct
        but O(N * C_out * C_in * kH * kW * H * W) — expensive for large
        batches, but adequate for the small batch sizes (4-16) used here.

        Parameters
        ----------
        X_batch : np.ndarray, shape (N, 2, 50, 50)
        Y_batch : np.ndarray, shape (N, 50, 50)
            Ground-truth normalised cost maps.

        Returns
        -------
        float
            Mean MSE loss over the batch.

        Notes
        -----
        Performance: the nested loops in the backward pass are the dominant
        cost (~6 s per epoch for 500 scenarios on a single CPU core at
        batch_size=16).

        TODO: replace the Python loop convolution backward with a scipy
              fftconvolve-based implementation for a 10-50x speedup.
        """
        N    = X_batch.shape[0]
        H, W = self.GRID_H, self.GRID_W
        pad  = 1

        # Gradient accumulators
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        dW3 = np.zeros_like(self.W3)
        db3 = np.zeros_like(self.b3)
        total_loss = 0.0

        for i in range(N):
            out, a1, a2, xp, a1p, a2p = self._forward_single(X_batch[i])
            y   = Y_batch[i]
            diff = out.squeeze(0) - y          # (H, W)  — MSE residual
            total_loss += float(np.mean(diff ** 2))

            # dL/d_out3 after ReLU: zero where output was clamped
            dout = (2.0 / (H * W)) * diff      # (H, W)
            dout = dout * (out.squeeze(0) > 0) # through output ReLU

            # Layer 3 weight gradients
            for ci in range(32):
                for ki in range(3):
                    for kj in range(3):
                        dW3[0,ci,ki,kj] += np.sum(dout * a2p[ci,ki:ki+H,kj:kj+W])
            db3[0] += np.sum(dout)

            # Backprop through layer 3 to a2
            da2 = np.zeros_like(a2)
            for ci in range(32):
                for ki in range(3):
                    for kj in range(3):
                        r0 = max(0, pad-ki);  r1 = H - max(0, ki-pad)
                        c0 = max(0, pad-kj);  c1 = W - max(0, kj-pad)
                        pr = max(0, ki-pad);   pc = max(0, kj-pad)
                        da2[ci, r0:r1, c0:c1] += (
                            self.W3[0,ci,ki,kj]
                            * dout[max(0,pad-ki):H-max(0,ki-pad),
                                   max(0,pad-kj):W-max(0,kj-pad)]
                        )
            da2_relu = da2 * (a2 > 0)   # through layer 2 ReLU

            # Layer 2 weight gradients
            for co in range(32):
                for ci in range(16):
                    for ki in range(3):
                        for kj in range(3):
                            dW2[co,ci,ki,kj] += np.sum(da2_relu[co] * a1p[ci,ki:ki+H,kj:kj+W])
                db2[co] += np.sum(da2_relu[co])

            # Backprop through layer 2 to a1
            da1 = np.zeros_like(a1)
            for co in range(32):
                for ci in range(16):
                    for ki in range(3):
                        for kj in range(3):
                            r0 = max(0, pad-ki);  r1 = H - max(0, ki-pad)
                            c0 = max(0, pad-kj);  c1 = W - max(0, kj-pad)
                            pr = max(0, ki-pad);   pc = max(0, kj-pad)
                            h_sl = slice(pr, pr + r1 - r0)
                            w_sl = slice(pc, pc + c1 - c0)
                            da1[ci, r0:r1, c0:c1] += (
                                self.W2[co,ci,ki,kj] * da2_relu[co, h_sl, w_sl]
                            )
            da1_relu = da1 * (a1 > 0)   # through layer 1 ReLU

            # Layer 1 weight gradients
            for co in range(16):
                for ci in range(2):
                    for ki in range(3):
                        for kj in range(3):
                            dW1[co,ci,ki,kj] += np.sum(da1_relu[co] * xp[ci,ki:ki+H,kj:kj+W])
                db1[co] += np.sum(da1_relu[co])

        grads = {
            'W1': dW1/N, 'b1': db1/N,
            'W2': dW2/N, 'b2': db2/N,
            'W3': dW3/N, 'b3': db3/N,
        }
        self._adam_update(grads)
        return total_loss / N

    def save(self, path="cost_cnn.npy"):
        """Persist weights to a .npy file (Adam state excluded)."""
        np.save(path, {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'is_trained': np.array([self.is_trained]),
        })
        print(f"  Saved SocialCostCNN weights -> {path}")

    def load(self, path="cost_cnn.npy"):
        """Load weights from a .npy file; resets Adam state."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"SocialCostCNN weights not found: {path}")
        weights = np.load(path, allow_pickle=True).item()
        self.W1 = weights['W1'].astype(np.float32)
        self.b1 = weights['b1'].astype(np.float32)
        self.W2 = weights['W2'].astype(np.float32)
        self.b2 = weights['b2'].astype(np.float32)
        self.W3 = weights['W3'].astype(np.float32)
        self.b3 = weights['b3'].astype(np.float32)
        self.is_trained = bool(weights.get('is_trained', [False])[0])
        self._adam_init()
        print(f"  Loaded SocialCostCNN weights <- {path}")


# ===========================================================================
# C. NeuralSocialZone
# ===========================================================================

class NeuralSocialZone:
    """
    Learned personal-space model with the same interface as SocialZone.

    Wraps a trained ZoneLearnerNet to provide compute_cost(),
    is_violation(), required_distance(), and get_zone_boundary_points()
    — the same four methods used by AStarPlanner and its subclasses.
    This makes NeuralSocialZone a drop-in replacement for SocialZone
    without any changes to the planning or experiment code.

    Comfort-to-distance mapping
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ZoneLearnerNet outputs a comfort score c in [0, 1].  To produce a
    required_distance() value compatible with the cost kernel in
    compute_cost(), comfort is mapped to distance via::

        d_req = d_min + (1 - c) * (d_max - d_min)

    where d_min = 0.3 m (physical collision radius) and d_max = 1.8 m
    (front-zone distance).  Low comfort (c -> 0) maps to large required
    distance; high comfort (c -> 1) maps to near-zero required distance.
    This is a first-order approximation; a better mapping would invert the
    empirical distance-comfort relationship from the training data.

    Boundary tracing
    ~~~~~~~~~~~~~~~~
    get_zone_boundary_points() uses binary search over radius at each of
    n_points angles to find where comfort crosses VIOLATION_THRESHOLD = 0.5.
    Twelve bisection steps give a radial resolution of
    (3.0 - 0.1) / 2^12 ~ 0.7 mm, more than sufficient for visualisation.

    Fallback
    ~~~~~~~~
    When the model is untrained or None, all methods delegate to a
    rule-based SocialZone with the default (Hall, 1966) distances.  This
    ensures that NeuralSocialZone is always usable, even before training.

    Parameters
    ----------
    model : ZoneLearnerNet or None
    fallback_distances : dict or None
        Zone distances for the rule-based fallback.
    """

    VIOLATION_THRESHOLD = 0.5   # comfort below this -> zone violated
    MAX_REQUIRED_DIST   = 1.8   # metres; maps to comfort = 0
    MIN_REQUIRED_DIST   = 0.3   # metres; maps to comfort = 1

    def __init__(self, model=None, fallback_distances=None):
        self.model    = model
        self.fallback = fallback_distances or {
            'front': 1.8, 'back': 0.6, 'left': 1.2, 'right': 1.2,
        }
        from social_zone import SocialZone
        self._fallback_zone = SocialZone.from_dict(self.fallback)

    @classmethod
    def from_file(cls, path="zone_learner.npy", fallback_distances=None):
        """
        Load a trained ZoneLearnerNet and wrap it in a NeuralSocialZone.

        Parameters
        ----------
        path : str
        fallback_distances : dict or None

        Returns
        -------
        NeuralSocialZone
        """
        model = ZoneLearnerNet()
        model.load(path)
        return cls(model=model, fallback_distances=fallback_distances)

    def _features(self, robot_pos, human_pos, human_orientation):
        """
        Build the (1, 4) input feature vector for ZoneLearnerNet.

        Parameters
        ----------
        robot_pos, human_pos : array-like, shape (2,)
        human_orientation : float

        Returns
        -------
        np.ndarray, shape (1, 4), dtype float32
        """
        dx = float(robot_pos[0]) - float(human_pos[0])
        dy = float(robot_pos[1]) - float(human_pos[1])
        return np.array(
            [[dx, dy, np.sin(human_orientation), np.cos(human_orientation)]],
            dtype=np.float32,
        )

    def comfort_score(self, robot_pos, human_pos, human_orientation):
        """
        Predict comfort of the robot being at robot_pos relative to the human.

        Parameters
        ----------
        robot_pos, human_pos : array-like, shape (2,)
        human_orientation : float

        Returns
        -------
        float in [0, 1]
            1 = comfortable (no intrusion), 0 = severe violation.
        """
        if self.model is None or not self.model.is_trained:
            # Fallback: convert analytical cost to comfort via exp(-cost)
            cost = self._fallback_zone.compute_cost(robot_pos, human_pos, human_orientation)
            return float(np.exp(-cost))

        X = self._features(robot_pos, human_pos, human_orientation)
        return float(self.model.predict(X)[0])

    def required_distance(self, robot_pos, human_pos, human_orientation):
        """
        Map comfort score to an equivalent required-distance value.

        Uses the linear comfort-to-distance mapping described in the class
        docstring.  The result has the same units and interpretation as
        SocialZone.required_distance().

        Parameters
        ----------
        robot_pos, human_pos : array-like, shape (2,)
        human_orientation : float

        Returns
        -------
        float
            Required separation in metres.
        """
        if self.model is None or not self.model.is_trained:
            return self._fallback_zone.required_distance(robot_pos, human_pos, human_orientation)

        c = self.comfort_score(robot_pos, human_pos, human_orientation)
        return float(
            self.MIN_REQUIRED_DIST
            + (1.0 - c) * (self.MAX_REQUIRED_DIST - self.MIN_REQUIRED_DIST)
        )

    def compute_cost(self, robot_pos, human_pos, human_orientation):
        """
        Social cost using the same exponential kernel as SocialZone.

        Falls back to the analytical zone when the model is untrained.

        Parameters
        ----------
        robot_pos, human_pos : array-like, shape (2,)
        human_orientation : float

        Returns
        -------
        float >= 0
        """
        if self.model is None or not self.model.is_trained:
            return self._fallback_zone.compute_cost(robot_pos, human_pos, human_orientation)

        dx = float(robot_pos[0]) - float(human_pos[0])
        dy = float(robot_pos[1]) - float(human_pos[1])
        actual_dist = float(np.sqrt(dx ** 2 + dy ** 2))
        required    = self.required_distance(robot_pos, human_pos, human_orientation)

        if actual_dist < required:
            return float(np.exp(required - actual_dist) - 1.0)
        return 0.0

    def is_violation(self, robot_pos, human_pos, human_orientation):
        """
        Test whether the robot is in a social zone violation.

        When the model is trained, violation is declared when comfort < 0.5.
        When untrained, delegates to the fallback SocialZone distance check.

        Parameters
        ----------
        robot_pos, human_pos : array-like, shape (2,)
        human_orientation : float

        Returns
        -------
        bool
        """
        if self.model is None or not self.model.is_trained:
            return self._fallback_zone.is_violation(robot_pos, human_pos, human_orientation)
        return self.comfort_score(robot_pos, human_pos, human_orientation) < self.VIOLATION_THRESHOLD

    def get_zone_boundary_points(self, human_pos, human_orientation, n_points=360):
        """
        Trace the comfort = 0.5 iso-contour by binary search over radius.

        For each of n_points evenly-spaced angles, bisects in [0.1, 3.0] m
        to find the radius where comfort crosses VIOLATION_THRESHOLD.
        Twelve bisection steps give sub-millimetre radial accuracy.

        Parameters
        ----------
        human_pos : array-like, shape (2,)
        human_orientation : float
        n_points : int

        Returns
        -------
        xs, ys : np.ndarray, shape (n_points,)

        Notes
        -----
        This is O(n_points * 12 * network_forward) per call.  For n_points=360
        and a fast ZoneLearnerNet forward (~1 us), the total is ~4 ms per human.
        Reduce n_points to 36 for real-time visualisation if needed.

        TODO: cache the boundary per (human_pos, human_orientation) tuple
              when called repeatedly for the same agent.
        """
        angles = np.linspace(-np.pi, np.pi, n_points)
        xs     = np.empty(n_points)
        ys     = np.empty(n_points)

        for k, angle in enumerate(angles):
            abs_angle = human_orientation + angle
            lo, hi    = 0.1, 3.0

            # Binary search: find r such that comfort(r) = VIOLATION_THRESHOLD
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                rp  = np.array([
                    human_pos[0] + mid * np.cos(abs_angle),
                    human_pos[1] + mid * np.sin(abs_angle),
                ])
                if self.comfort_score(rp, human_pos, human_orientation) < self.VIOLATION_THRESHOLD:
                    lo = mid   # too close: push boundary outward
                else:
                    hi = mid   # comfortable: boundary is closer

            r     = 0.5 * (lo + hi)
            xs[k] = human_pos[0] + r * np.cos(abs_angle)
            ys[k] = human_pos[1] + r * np.sin(abs_angle)

        return xs, ys

    def __repr__(self):
        state = "trained" if (self.model and self.model.is_trained) else "untrained"
        return f"NeuralSocialZone({state})"
