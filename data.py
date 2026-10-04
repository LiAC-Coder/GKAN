from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

DATASETS = {
    "cf": ("DianPing", "Tmall"),
    "social": ("Ciao", "Epinions"),
    "heterogeneous": ("Douban", "KKBox"),
}


def read_rows(path):
    values = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if values.shape[1] != 3 or not len(values):
        raise ValueError("Invalid input")
    return values


def canonical_edges(pairs, num_nodes, undirected=True):

    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if np.any(pairs < 0) or np.any(pairs >= num_nodes):
        raise ValueError("Invalid input")
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if undirected:
        pairs = np.concatenate((pairs, pairs[:, ::-1]))
    return np.unique(pairs, axis=0)


def mean_adjacency(pairs, num_nodes):

    pairs = canonical_edges(pairs, num_nodes, undirected=False)
    degree = np.bincount(pairs[:, 0], minlength=num_nodes)
    values = 1.0 / degree[pairs[:, 0]]
    indices = torch.from_numpy(pairs.T.copy()).long()
    return torch.sparse_coo_tensor(
        indices, torch.from_numpy(values.astype(np.float32)),
        (num_nodes, num_nodes), check_invariants=True,
    ).coalesce()


@dataclass
class DatasetBundle:
    name: str
    task: str
    num_user: int
    num_item: int
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    short_edges: np.ndarray
    long_edges: np.ndarray
    graph_info: dict

    @property
    def num_nodes(self):
        return self.num_user + self.num_item

    @property
    def train_positives(self):
        rows = self.train.numpy()
        return rows[rows[:, 2] == 1, :2]

    @property
    def graph_fingerprint(self):
        digest = hashlib.sha256()
        digest.update(f"paper-G-v1:{self.num_user}:{self.num_item}".encode())
        digest.update(self.short_edges.tobytes())
        return digest.hexdigest()


def load_data(args, task):
    if args.dataset not in DATASETS[task]:
        raise ValueError("Invalid input")
    folder = Path(args.data_dir).resolve() / args.dataset
    arrays = {name: read_rows(folder / (name + ".txt"))
              for name in ("train", "eval", "test")}
    all_rows = np.concatenate(list(arrays.values()))
    if not np.all(np.isin(all_rows[:, 2], (0, 1))):
        raise ValueError("Invalid input")
    user_ids, item_ids = np.unique(all_rows[:, 0]), np.unique(all_rows[:, 1])
    if not np.array_equal(user_ids, np.arange(len(user_ids))):
        raise ValueError("Invalid input")
    if not np.array_equal(item_ids, np.arange(len(item_ids))):
        raise ValueError("Invalid input")
    n_user, n_item = len(user_ids), len(item_ids)
    n_nodes = n_user + n_item
    train_positive = arrays["train"][arrays["train"][:, 2] == 1, :2]
    if not len(train_positive):
        raise ValueError("Invalid input")
    interaction_edges = train_positive.copy()
    interaction_edges[:, 1] += n_user
    long_parts = []
    info = {"training_positive_records": len(train_positive), "relation_files": {}}

    def relation(filename, size, offset=0):
        rows = read_rows(folder / filename)
        if np.any(rows[:, :2] < 0) or np.any(rows[:, :2] >= size):
            raise ValueError("Invalid input")
        if not np.all(rows[:, 2] == 1):
            raise ValueError("Invalid input")
        info["relation_files"][filename] = len(rows)
        return rows[:, :2] + offset

    long_parts.append(relation("user_long.txt", n_user))
    long_parts.append(relation("item_long.txt", n_item, n_user))


    edges = canonical_edges(interaction_edges, n_nodes)
    long_edges = canonical_edges(np.concatenate(long_parts), n_nodes)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        info["merge_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    info.update(directed_edges=len(edges), undirected_edges=len(edges) // 2,
                long_directed_edges=len(long_edges), neighborhood_policy="interactions",
                node_coverage=len(np.unique(edges)))
    return DatasetBundle(
        args.dataset, task, n_user, n_item,
        torch.from_numpy(arrays["train"]), torch.from_numpy(arrays["eval"]),
        torch.from_numpy(arrays["test"]), edges, long_edges, info,
    )


class NegativeSampler:


    def __init__(self, positive_pairs, num_user, num_item, seed):
        self.num_item = num_item
        self.known = np.unique(positive_pairs[:, 0] * num_item + positive_pairs[:, 1])
        counts = np.bincount(self.known // num_item, minlength=num_user)
        if np.any(counts == num_item):
            raise ValueError("Invalid input")
        self.rng = np.random.default_rng(seed)

    def sample(self, users):
        users = np.asarray(users, dtype=np.int64)
        result = self.rng.integers(self.num_item, size=len(users))
        pending = np.arange(len(users))
        while len(pending):
            keys = users[pending] * self.num_item + result[pending]
            indices = np.searchsorted(self.known, keys)
            invalid = (indices < len(self.known)) & (
                self.known[np.minimum(indices, len(self.known) - 1)] == keys)
            pending = pending[invalid]
            result[pending] = self.rng.integers(self.num_item, size=len(pending))
        return result
