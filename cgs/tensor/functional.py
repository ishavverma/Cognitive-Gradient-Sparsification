"""
High-Level Differentiable Functions.

Built on top of primitive ops, these provide commonly needed functions
like softmax, cross-entropy, and contrastive losses.
"""

import numpy as np
from .tensor import CGSTensor, _ensure_tensor, is_grad_enabled
from .ops import _should_track, _unbroadcast


def softmax(x: CGSTensor, axis: int = -1) -> CGSTensor:
    """
    Numerically stable softmax.

    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    # Numerical stability: subtract max
    x_max = np.max(x.data, axis=axis, keepdims=True)
    e_x = np.exp(x.data - x_max)
    s = e_x / np.sum(e_x, axis=axis, keepdims=True)
    track = _should_track(x)

    out = CGSTensor(s, requires_grad=track, _prev={x})

    if track:
        def _backward(grad):
            if x.requires_grad:
                # Jacobian-vector product for softmax
                # d_softmax/d_x = diag(s) - s @ s^T
                # Efficient: grad * s - s * sum(grad * s)
                gs = grad * s
                x_grad = gs - s * np.sum(gs, axis=axis, keepdims=True)
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


def log_softmax(x: CGSTensor, axis: int = -1) -> CGSTensor:
    """
    Numerically stable log-softmax.

    log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
    """
    x_max = np.max(x.data, axis=axis, keepdims=True)
    shifted = x.data - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    out_data = shifted - log_sum_exp
    track = _should_track(x)

    out = CGSTensor(out_data, requires_grad=track, _prev={x})

    if track:
        s = np.exp(out_data)  # softmax values

        def _backward(grad):
            if x.requires_grad:
                x_grad = grad - s * np.sum(grad, axis=axis, keepdims=True)
                x.grad = x_grad if x.grad is None else x.grad + x_grad
        out._backward_fn = _backward

    return out


def cross_entropy(logits: CGSTensor, targets: CGSTensor) -> CGSTensor:
    """
    Cross-entropy loss with built-in log-softmax for numerical stability.

    Args:
        logits: Raw model outputs, shape (batch, num_classes).
        targets: Integer class labels, shape (batch,).

    Returns:
        Scalar loss (mean over batch).
    """
    logits = _ensure_tensor(logits)
    targets = _ensure_tensor(targets)
    batch_size = logits.shape[0]
    target_idx = targets.data.astype(int)

    # Numerically stable log-softmax
    x_max = np.max(logits.data, axis=1, keepdims=True)
    shifted = logits.data - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probs = shifted - log_sum_exp

    # Negative log likelihood
    nll = -log_probs[np.arange(batch_size), target_idx]
    loss = np.mean(nll)
    track = _should_track(logits)

    out = CGSTensor(loss, requires_grad=track, _prev={logits})

    if track:
        s = np.exp(log_probs)  # softmax probabilities

        def _backward(grad):
            if logits.requires_grad:
                g = s.copy()
                g[np.arange(batch_size), target_idx] -= 1.0
                g = g / batch_size * grad
                logits.grad = g if logits.grad is None else logits.grad + g
        out._backward_fn = _backward

    return out


def mse_loss(pred: CGSTensor, target: CGSTensor) -> CGSTensor:
    """
    Mean Squared Error loss.

    MSE = mean((pred - target)^2)
    """
    pred = _ensure_tensor(pred)
    target = _ensure_tensor(target)

    diff = pred.data - target.data
    loss = np.mean(diff ** 2)
    track = _should_track(pred)

    out = CGSTensor(loss, requires_grad=track, _prev={pred})

    if track:
        def _backward(grad):
            if pred.requires_grad:
                g = 2 * diff / diff.size * grad
                g = _unbroadcast(g, pred.shape)
                pred.grad = g if pred.grad is None else pred.grad + g
        out._backward_fn = _backward

    return out


def cosine_similarity(a: CGSTensor, b: CGSTensor, axis: int = -1, eps: float = 1e-8) -> CGSTensor:
    """
    Cosine similarity between two tensors.

    cos_sim(a, b) = (a · b) / (||a|| · ||b||)
    """
    a, b = _ensure_tensor(a), _ensure_tensor(b)

    a_norm = np.sqrt(np.sum(a.data ** 2, axis=axis, keepdims=True) + eps)
    b_norm = np.sqrt(np.sum(b.data ** 2, axis=axis, keepdims=True) + eps)
    a_normalized = a.data / a_norm
    b_normalized = b.data / b_norm
    cos_sim = np.sum(a_normalized * b_normalized, axis=axis, keepdims=True)

    track = _should_track(a, b)
    out = CGSTensor(cos_sim, requires_grad=track, _prev={a, b})

    if track:
        def _backward(grad):
            if a.requires_grad:
                # d/da cos(a,b) = (b/||b|| - cos * a/||a||) / ||a||
                a_grad = (b_normalized - cos_sim * a_normalized) / a_norm
                a_grad = _unbroadcast(a_grad * grad, a.shape)
                a.grad = a_grad if a.grad is None else a.grad + a_grad
            if b.requires_grad:
                b_grad = (a_normalized - cos_sim * b_normalized) / b_norm
                b_grad = _unbroadcast(b_grad * grad, b.shape)
                b.grad = b_grad if b.grad is None else b.grad + b_grad
        out._backward_fn = _backward

    return out


def contrastive_loss(z1: CGSTensor, z2: CGSTensor, temperature: float = 0.5) -> CGSTensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss.

    Used for self-supervised multi-view learning. Pushes representations
    of the same input (from different views) closer together while pushing
    representations of different inputs apart.

    Args:
        z1: Representations from view 1, shape (batch, dim).
        z2: Representations from view 2, shape (batch, dim).
        temperature: Temperature parameter for scaling.

    Returns:
        Scalar contrastive loss.
    """
    z1, z2 = _ensure_tensor(z1), _ensure_tensor(z2)
    batch_size = z1.shape[0]

    # Normalize
    z1_norm = z1.data / (np.sqrt(np.sum(z1.data ** 2, axis=1, keepdims=True)) + 1e-8)
    z2_norm = z2.data / (np.sqrt(np.sum(z2.data ** 2, axis=1, keepdims=True)) + 1e-8)

    # Similarity matrix
    sim = z1_norm @ z2_norm.T / temperature  # (batch, batch)

    # Labels: positive pairs are on the diagonal
    # NTXent loss for each z1[i], positive pair is z2[i]
    sim_max = np.max(sim, axis=1, keepdims=True)
    exp_sim = np.exp(sim - sim_max)
    log_sum_exp = np.log(np.sum(exp_sim, axis=1)) + sim_max.squeeze()

    # Positive similarity (diagonal)
    pos_sim = np.sum(z1_norm * z2_norm, axis=1) / temperature

    loss = np.mean(-pos_sim + log_sum_exp)
    track = _should_track(z1, z2)

    out = CGSTensor(loss, requires_grad=track, _prev={z1, z2})

    if track:
        def _backward(grad):
            # Softmax of similarities
            probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

            if z1.requires_grad:
                # Gradient w.r.t. z1 via similarity matrix
                d_sim = (probs.copy()) / batch_size
                d_sim[np.arange(batch_size), np.arange(batch_size)] -= 1.0 / batch_size
                z1_grad = d_sim @ z2_norm / temperature

                # Account for normalization (simplified)
                z1_len = np.sqrt(np.sum(z1.data ** 2, axis=1, keepdims=True)) + 1e-8
                z1_grad = (z1_grad - z1_norm * np.sum(z1_grad * z1_norm, axis=1, keepdims=True)) / z1_len
                z1_grad *= grad
                z1.grad = z1_grad if z1.grad is None else z1.grad + z1_grad

            if z2.requires_grad:
                d_sim = probs.copy() / batch_size
                d_sim[np.arange(batch_size), np.arange(batch_size)] -= 1.0 / batch_size
                z2_grad = d_sim.T @ z1_norm / temperature

                z2_len = np.sqrt(np.sum(z2.data ** 2, axis=1, keepdims=True)) + 1e-8
                z2_grad = (z2_grad - z2_norm * np.sum(z2_grad * z2_norm, axis=1, keepdims=True)) / z2_len
                z2_grad *= grad
                z2.grad = z2_grad if z2.grad is None else z2.grad + z2_grad
        out._backward_fn = _backward

    return out
