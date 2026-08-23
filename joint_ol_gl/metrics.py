import pygsp
import torch
import numpy as np
from torch.nn import Module
from pygsp.graphs import Graph
from sklearn.metrics import f1_score, recall_score, precision_score

def compute_mse(
        sparsenet: Module,
        train_dataloader: torch.utils.data.DataLoader,
        n_samples: int,
        device: str = 'cpu'
) -> float:
    """
    Computed mean squared error (MSE).
    :param sparsenet: SparseNet instance to evaluate.
    :param test_dataloader: data loader for test dataset.
    :return: mean square error evaluated on test dataset.
    """
    mse = 0
    n_iters = int(np.ceil(n_samples / train_dataloader.batch_size))
    for i in range(n_iters):
        X_train_i = next(iter(train_dataloader)).to(device)
        X_train_i_pred = sparsenet.predict(X_train_i)
        mse += ((X_train_i - X_train_i_pred) ** 2).sum()
    mse /= n_samples
    return mse.item()

def compute_cos_sim(
        sparsenet: Module,
        D0: np.ndarray,
        n_atoms: int
) -> float:
    """
    Cosine similarity.
    :param sparsenet: SparseNet instance to evaluate.
    :param D0: ground truth dictionary.
    :param n_atoms: number of atoms.
    :param device: device in whic to compute MSE (CPU or GPU).
    :return: average cosine similarity between computed dictionary and ground-truth dictionary.
    """
    D_norm = sparsenet.D / torch.linalg.norm(sparsenet.D, axis=0)
    D0_norm = D0.T / torch.linalg.norm(D0.T, axis=0)
    G_cos = torch.abs(D_norm.T @ D0_norm)
    return G_cos.max(0)[0].sum() / sparsenet.n_atoms
    # pairs = []
    # if hasattr(sparsenet, "n_filters"):
    #     n_atoms = sparsenet.n_filters * sparsenet.n_nodes
    # elif hasattr(sparsenet, "n_atoms"):
    #     n_atoms = sparsenet.n_atoms
    # for f in range(n_atoms):
    #     pair = (torch.where(G_cos == G_cos.max())[0][0].item(), torch.where(G_cos == G_cos.max())[1][0].item())
    #     G_cos[pair[0], :] = - 1.
    #     G_cos[:, pair[1]] = - 1.
    #     pairs.append(pair)
    # G_cos = torch.abs(D_norm.T @ D0_norm)
    # sims = G_cos[tuple(zip(*pairs))]
    # return sims.mean()

def compute_nae(
        W: torch.FloatTensor,
        graph: pygsp.graphs.Graph
) -> float:
    """
    Normalized adjacency error between adjacency matrix of computed graph and ground-truth graph.
    :param sparsenet: SparseNet instance to evaluate.
    :param graph: ground-truth graph PyGSP object.
    :return: Weighted F-score of
    """
    W0 = graph.W.todense()
    # W *= np.linalg.norm(W0) / np.linalg.norm(W)
    return (np.linalg.norm(W - W0) / np.linalg.norm(W0))

def compute_filterr_ggsr(
        S: torch.FloatTensor,
        T: torch.FloatTensor,
        h: torch.FloatTensor,
) -> float:
    Ls, Us = torch.linalg.eigh(S)
    Lt, Ut = torch.linalg.eigh(T)
    Gts = torch.abs(Ut.T @ Us)
    map_ts = Gts.argmax(1)
    Ls = Ls[map_ts]

    resp_h = h[0] + h[1] * Ls + h[2] * Ls ** 2
    resp_h = resp_h ** 2
    valsTi = 1 / torch.linalg.eigvalsh(T)
    resp_h = resp_h * valsTi.dot(resp_h) / resp_h.dot(resp_h)
    return torch.norm(resp_h - valsTi) / torch.norm(resp_h)

def laplacian_to_adj(L):
    L = L.cpu().numpy()
    A = -L.copy()
    np.fill_diagonal(A, 0.0)
    # symmetrize
    A = (A + A.T) / 2.0
    # clip small negatives to zero (numerical)
    A[A<0] = 0 # keep positives only; negative shouldn't occur here
    return A

def top_k_from_adj(A_hat, k):
    iu = np.triu_indices_from(A_hat, k=1)
    scores = A_hat[iu]
    # pick indices of top-k scores
    idx = np.argsort(scores)[-k:]
    pred = np.zeros_like(scores, dtype=int)
    pred[idx] = 1
    pred_adj = np.zeros_like(A_hat)
    pred_adj[iu] = pred
    pred_adj = pred_adj + pred_adj.T
    return pred_adj

def evaluate_laplacian(
        graph: Graph,
        L: torch.FloatTensor,
):
    W = laplacian_to_adj(L)
    return evaluate_adjacency(graph, W)

def evaluate_adjacency(
        graph: Graph,
        W: torch.FloatTensor,
):
    performance_metrics = {}
    if graph is not None:
        num_nz = int(np.triu(graph.W.todense()).sum())
        trueW = graph.W.todense().__array__().flatten().astype(int)
        predW = top_k_from_adj(W, k=num_nz).flatten().astype(int)
        nae = compute_nae(W, graph)
        f_score = f1_score(trueW, predW)
        precision = precision_score(trueW, predW)
        recall = recall_score(trueW, predW)
        lap_sparsity = (W[np.abs(W) > 1e-2] != 0).sum() / 2

        performance_metrics = {
            'nae': nae, 'f1_score': f_score, 'precision': precision, 'recall': recall,
            'sp': lap_sparsity
        }
    return performance_metrics

def compute_scores(W: torch.FloatTensor, graph: Graph, thresh: float = 0.5):
    """
    Compute true/false positives/negatives for a learned adjacency matrix.
    Counts are computed over the upper triangle (excluding the diagonal).
    """
    W = W.detach().cpu().numpy()
    W_true = np.asarray(graph.W.todense())
    n = W.shape[0]
    iu = np.triu_indices(n, k=1)

    pred = (np.abs(W[iu]) > thresh).astype(int)
    true = (W_true[iu] != 0).astype(int)

    tp = int((pred * true).sum())
    fp = int((pred * (1 - true)).sum())
    fn = int(((1 - pred) * true).sum())
    tn = int(((1 - pred) * (1 - true)).sum())
    return tp, fp, tn, fn


def compute_filterr_poly(
        L: torch.FloatTensor,
        h: torch.FloatTensor,
        a: torch.FloatTensor,
):
    e = torch.linalg.eigvalsh(L)
    deg = len(h) - 1
    poly = h[0] * (e ** 0)
    for p in range(1, 1 + deg):
        poly += h[p] * (e ** p)
    resp_h = 1 / poly
    resp_h = resp_h ** 2
    resp_a = a[0] + a[1] * e
    for p in range(2, len(a)):
        resp_a += a[p] * e ** p
    resp_a = 1 / resp_a
    resp_a *= resp_h.dot(resp_a) / resp_a.dot(resp_a)
    return torch.linalg.norm(resp_h - resp_a)
