import math
import random
import numpy as np
import torch

K_LIST = [1, 2, 3, 5, 10, 20]


def auc_acc(labels, pred):
    labels, pred = np.asarray(labels).reshape(-1), np.asarray(pred).reshape(-1)
    if len(labels) != len(pred) or not len(labels):
        raise ValueError("Invalid input")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(pred).all():
        raise ValueError("Invalid input")
    order = np.argsort(pred, kind="stable")
    sorted_scores = pred[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(pred)]

    ranks = np.repeat((boundaries[:-1] + 1 + boundaries[1:]) / 2, np.diff(boundaries))
    positives = labels[order] == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    auc = ((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
           if n_pos and n_neg else float("nan"))
    binary = pred >= 0.5
    tp = int(((labels == 1) & binary).sum())
    denominator = int((labels == 1).sum() + binary.sum())
    return float(auc), float((binary == labels).mean()), 2 * tp / denominator if denominator else 0.0


def logLoss(labels, pred):
    scores = np.clip(np.asarray(pred), 1e-15, 1 - 1e-15)
    labels = np.asarray(labels)
    return float(-(labels * np.log(scores) + (1 - labels) * np.log1p(-scores)).mean())


def get_user_record(data, is_train):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    result = {}
    for user, item, label in data:
        if is_train or label == 1:
            result.setdefault(int(user), set()).add(int(item))
    return result


def get_top_K_data(train_data, test_data, n_user, n_item):
    train = get_user_record(train_data, True)
    test = get_user_record(test_data, False)
    return sorted(train.keys() & test.keys()), train, test, set(range(n_item)), K_LIST


def ranking_metrics(ranking, positives, k_list=K_LIST):

    if not positives:
        raise ValueError("Invalid input")
    values = []
    for k in k_list:
        relevant = np.array([item in positives for item in ranking[:k]], dtype=float)
        hits = relevant.sum()
        precision, recall = hits / k, hits / len(positives)
        dcg = (relevant / np.log2(np.arange(len(relevant)) + 2)).sum()
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(positives))))
        ap = ((relevant.cumsum() / (np.arange(len(relevant)) + 1)) * relevant).sum()
        values.append((precision, recall, dcg / ideal, float(hits > 0), ap / min(k, len(positives))))
    return np.asarray(values)


@torch.no_grad()
def topk_eval(model, train_data, test_data, n_user, n_item, user_embedding, item_embedding,
              negative_count=1000, max_users=0):
    users, train, test, items, ks = get_top_K_data(train_data, test_data, n_user, n_item)
    if max_users:
        users = users[:max_users]
    if not users:
        return tuple([float("nan")] * len(ks) for _ in range(6))
    total = np.zeros((len(ks), 5))
    for user in users:
        pool = sorted(items - train[user] - test[user])
        sampled = random.Random(user).sample(pool, min(negative_count, len(pool)))
        candidates = np.array(sorted(set(sampled) | test[user]), dtype=np.int64)

        scores = model.get_logits(user_embedding, item_embedding,
                                  np.full(len(candidates), user), candidates).cpu().numpy()
        ranking = candidates[np.argsort(-scores, kind="stable")].tolist()
        total += ranking_metrics(ranking, test[user], ks)
    precision, recall, ndcg, hit, maps = (total / len(users)).T
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=precision + recall > 0)
    return tuple(x.tolist() for x in (precision, recall, f1, ndcg, hit, maps))


@torch.no_grad()
def classification_eval(model, rows, user_embedding, item_embedding, batch_size, auc_mode="batch"):
    batches, all_labels, all_scores = [], [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        scores = model.get_rating(user_embedding, item_embedding, batch[:, 0], batch[:, 1]).cpu().numpy()
        labels = batch[:, 2].cpu().numpy()
        batches.append(auc_acc(labels, scores))
        if auc_mode == "global":
            all_labels.append(labels)
            all_scores.append(scores)
    if auc_mode == "global":
        metrics = auc_acc(np.concatenate(all_labels), np.concatenate(all_scores))
    else:
        values = np.asarray(batches)
        valid = values[:, 0][np.isfinite(values[:, 0])]
        metrics = (float(valid.mean()) if len(valid) else float("nan"),
                   float(values[:, 1].mean()), float(values[:, 2].mean()))
    return {"AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2],
            "undefined_auc_batches": sum(not np.isfinite(x[0]) for x in batches)}
