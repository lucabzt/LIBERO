"""
Memory Anchor selection math (Memory Anchors paper, Alg. 1), in torch so it runs on the GPU
without scikit-learn. Mirrors the authors' SelectiveER: per-modality normalized observation
features, k-means split of the new data by its distances to the old data (step 1), and
old data scored by the median distance to its k nearest disagreeing new points (step 3).
"""

import torch


def normalize_representations(reprs, keys, stats=None):
    """
    Concatenates per-key features {key: (N, D_key)} in the order of keys, every dimension
    z-scored with stats {key: {"mean", "std"}} (computed from reprs if None), as the authors'
    individually_normalized representations. Constant dimensions (std 0, e.g. the task
    embedding while only one past task exists) are only centered, like sklearn's
    StandardScaler. Returns the (N, sum D_key) tensor and the stats.
    """
    if stats is None:
        stats = {}
        for key in keys:
            std = reprs[key].std(dim=0)
            stats[key] = {
                "mean": reprs[key].mean(dim=0),
                "std": torch.where(std > 0, std, torch.ones_like(std)),
            }
    normed = [(reprs[key] - stats[key]["mean"]) / stats[key]["std"] for key in keys]
    return torch.cat(normed, dim=1), stats


def lower_tail_distances(old, new, fraction=0.1, new_chunk_size=1024, old_chunk_size=8192):
    """
    For every new point, its sorted distances to the k = fraction * len(old) nearest old
    points, (N_new, k). Chunked over both sets to bound memory (the authors keep a running
    top-k over old chunks).
    """
    k = int(fraction * old.shape[0])
    assert k > 0, "[error] AnchorER step 1 needs at least 1 / fraction old samples"
    result = []
    for i in range(0, new.shape[0], new_chunk_size):
        new_blk = new[i : i + new_chunk_size]
        topk = torch.full((new_blk.shape[0], k), float("inf"), device=new.device)
        for j in range(0, old.shape[0], old_chunk_size):
            dist = torch.cdist(new_blk, old[j : j + old_chunk_size])
            topk = torch.topk(torch.cat([topk, dist], dim=1), k, dim=1, largest=False).values
        result.append(torch.sort(topk, dim=1).values)
    return torch.cat(result, dim=0)


def distance_features(topk):
    """
    Authors' features of the lower-tail distances: nearest neighbor, 1%, 5%, 10% quantile
    of all old data, spread (10% - NN) and local std, z-scored over the new points.
    """
    k = topk.shape[1]
    nn = topk[:, 0]
    q1 = topk[:, int(0.1 * k)]
    q5 = topk[:, int(0.5 * k)]
    q10 = topk[:, -1]
    X = torch.stack([nn, q1, q5, q10, q10 - nn, topk.std(dim=1)], dim=1)
    std = X.std(dim=0)
    return (X - X.mean(dim=0)) / torch.where(std > 0, std, torch.ones_like(std))


def kmeans(X, n_clusters=2, n_init=20, max_iter=300, tol=1e-4, generator=None):
    """
    Lloyd's k-means with k-means++ initialization, best of n_init runs by inertia
    (the defaults of sklearn's KMeans, which the authors use). Returns labels (N,).
    """
    N = X.shape[0]
    assert N >= n_clusters, "[error] fewer points than clusters"
    # sklearn's tolerance is relative to the mean variance of the data
    abs_tol = tol * X.var(dim=0).mean()
    best_labels, best_inertia = None, float("inf")
    for _ in range(n_init):
        first = torch.randint(N, (1,), generator=generator).item()
        centers = [X[first]]
        for _ in range(1, n_clusters):
            d2 = torch.cdist(X, torch.stack(centers)).min(dim=1).values ** 2
            probs = d2 / d2.sum() if d2.sum() > 0 else torch.full_like(d2, 1.0 / N)
            centers.append(X[torch.multinomial(probs.cpu(), 1, generator=generator).item()])
        centers = torch.stack(centers)
        for _ in range(max_iter):
            labels = torch.cdist(X, centers).argmin(dim=1)
            new_centers = torch.stack(
                [
                    X[labels == c].mean(dim=0) if (labels == c).any() else centers[c]
                    for c in range(n_clusters)
                ]
            )
            shift = ((new_centers - centers) ** 2).sum()
            centers = new_centers
            if shift <= abs_tol:
                break
        labels = torch.cdist(X, centers).argmin(dim=1)
        inertia = ((X - centers[labels]) ** 2).sum().item()
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels


def overlap_partition(old, new, n_init=20, generator=None):
    """
    Step 1: splits the new points into two k-means clusters of their distance features;
    the cluster with the smaller mean nearest-neighbor distance overlaps the old data.
    Returns the overlap mask over new (N_new,) and the (unnormalized) NN distances.
    """
    topk = lower_tail_distances(old, new)
    X = distance_features(topk)
    labels = kmeans(X, n_clusters=2, n_init=n_init, generator=generator)
    in_cluster = 0 if X[labels == 0, 0].mean() < X[labels == 1, 0].mean() else 1
    return labels == in_cluster, topk[:, 0]


def anchor_scores(old, target, n_neighbors, chunk_size=8192):
    """
    Step 3: score of every old point, the median distance to its n_neighbors nearest
    target points (fewer if the target set is smaller), (N_old,). Lower = stronger anchor.
    """
    k = min(n_neighbors, target.shape[0])
    scores = []
    for i in range(0, old.shape[0], chunk_size):
        dist = torch.cdist(old[i : i + chunk_size], target)
        knn = torch.topk(dist, k, dim=1, largest=False).values  # sorted ascending
        # numpy's median (mean of the two middle values for even k), as the authors
        mid = (k - 1) // 2
        scores.append(0.5 * (knn[:, mid] + knn[:, k // 2]))
    return torch.cat(scores, dim=0)
