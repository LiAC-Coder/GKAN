import argparse
import datetime
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import load_data, mean_adjacency, NegativeSampler
from .evaluation import classification_eval, topk_eval, K_LIST
from .model import GamerKAN

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"DianPing": "cf", "Tmall": "cf", "Ciao": "social", "Epinions": "social",
         "Douban": "heterogeneous", "KKBox": "heterogeneous"}


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Use true/false")


def build_parser(default_dataset):
    parser = argparse.ArgumentParser(description="GamerKAN hierarchical MoE with existing long-range relation files")
    parser.add_argument("--dataset", choices=TASKS, default=default_dataset)
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--log_dir", type=Path, default=ROOT / "log")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_ada1", "--lambda_low", dest="lambda_low", type=float, default=1e-4)
    parser.add_argument("--weight_ada2", "--lambda_high", dest="lambda_high", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=5)
    parser.add_argument("--spline_order", type=int, default=3)
    parser.add_argument("--activation_checkpoint", type=boolean, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--show_topk", type=boolean, default=True)
    parser.add_argument("--topk_negatives", type=int, default=1000)
    parser.add_argument("--auc_mode", choices=("batch", "global"), default="batch")
    parser.add_argument("--patience", type=int, default=10, help="0 disables validation early stopping")
    parser.add_argument("--max_train_batches", type=int, default=0, help="Smoke testing only; 0 trains the full epoch")
    parser.add_argument("--max_topk_users", type=int, default=0, help="Smoke testing only; 0 evaluates all eligible users")
    parser.add_argument("--inspect_only", action="store_true", help="Validate graph/data configuration without training")

    parser.add_argument("--KAN_model", default="four_basis")
    parser.add_argument("--pre_model", type=int, choices=(0,), default=0)
    parser.add_argument("--n_sample", type=int, default=None, help="Deprecated: existing neighbor unions are used in full")
    return parser


def pair_objective(model, output, users, positive_items, negative_items, lambda_low, lambda_high):
    user_embedding, item_embedding, low, high = output
    positive = model.get_logits(user_embedding, item_embedding, users, positive_items)
    negative = model.get_logits(user_embedding, item_embedding, users, negative_items)

    recommendation = (F.softplus(-positive) + F.softplus(negative)).mean()
    item_nodes = positive_items + model.num_user
    low_loss = (low[users] + low[item_nodes]).mean()
    high_loss = (high[users] + high[item_nodes]).mean()
    return recommendation + lambda_low * low_loss + lambda_high * high_loss


@torch.no_grad()
def evaluate_model(model, bundle, user_embedding, item_embedding, args):

    record = {}
    for name, rows in (("train", bundle.train), ("test", bundle.test), ("validation", bundle.validation)):
        record[name] = classification_eval(model, rows, user_embedding, item_embedding,
                                           args.batch_size, args.auc_mode)
    if args.show_topk:
        results = topk_eval(model, bundle.train, bundle.test, bundle.num_user, bundle.num_item,
                            user_embedding, item_embedding, args.topk_negatives, args.max_topk_users)
        record["topk"] = {"K": K_LIST, **dict(zip(("precision", "recall", "F1", "NDCG", "HR", "MAP"), results))}
    return record


def format_metrics(record):

    if "notice" in record:
        return record["notice"]
    if "epoch" in record:
        lines = [f"Epoch:{record['epoch']}", f"Testing: {record['epoch']}",
                 f"Train loss: {record['loss']:.6f}"]
    else:
        lines = [f"Best model results: Epoch:{record['best_epoch']}"]
    for name, label, prefix in (("train", "Train", "train"), ("test", "Test", "test"),
                                 ("validation", "Eval", "eval")):
        metrics = record[name]
        lines.append(f"{label} set results:  {prefix}_AUC: {metrics['AUC']:.4f}"
                     f"  {prefix}_ACC: {metrics['ACC']:.4f}  {prefix}_F1: {metrics['F1']:.4f}")
        if metrics["undefined_auc_batches"]:
            lines.append(f"{label} undefined AUC batches skipped: {metrics['undefined_auc_batches']}")
    if "topk" in record:
        topk = record["topk"]
        headings = [str(k) + ("," if index < len(topk["K"]) - 1 else "")
                    for index, k in enumerate(topk["K"])]
        lines.extend(["All data", "topK:  " + "".join(f"{heading:<10}" for heading in headings).rstrip()])
        for label, name in (("pre ", "precision"), ("rec ", "recall"), ("F1  ", "F1"),
                            ("ndcg", "NDCG"), ("hit ", "HR"), ("map ", "MAP")):
            lines.append(label + ":  " + "".join(f"{value:<10.6f}" for value in topk[name]).rstrip())
    return "\n".join(lines)


def main(default_dataset="DianPing", argv=None):
    args = build_parser(default_dataset).parse_args(argv)
    for name in ("epochs", "batch_size", "dim", "layers", "num_experts", "resolution", "num_threads"):
        if getattr(args, name) < 1:
            raise ValueError("Invalid input")
    if args.spline_order < 1 or min(args.patience, args.max_train_batches, args.max_topk_users, args.topk_negatives) < 0:
        raise ValueError("Invalid input")
    if min(args.lambda_low, args.lambda_high, args.weight_decay) < 0 or args.lr <= 0:
        raise ValueError("Invalid input")
    torch.set_num_threads(args.num_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(args.gpu_id)
        torch.cuda.manual_seed_all(args.seed)
    bundle = load_data(args, TASKS[args.dataset])
    dataset_info = {"dataset": bundle.name, "users": bundle.num_user, "items": bundle.num_item,
                      "positive_interactions_all_splits": sum(int((x[:, 2] == 1).sum())
                          for x in (bundle.train, bundle.validation, bundle.test)),
                      **{key: value for key, value in bundle.graph_info.items() if key != "merge_manifest"}}
    if args.inspect_only:
        print(json.dumps(dataset_info, ensure_ascii=False), flush=True)
        return bundle
    short_adj = mean_adjacency(bundle.short_edges, bundle.num_nodes)
    long_adj = mean_adjacency(bundle.long_edges, bundle.num_nodes)
    model = GamerKAN(args, bundle.num_user, bundle.num_item, short_adj, long_adj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = NegativeSampler(bundle.train_positives, bundle.num_user, bundle.num_item, args.seed)
    loader = DataLoader(torch.from_numpy(bundle.train_positives), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = args.output_dir.resolve() / f"{args.dataset}-{timestamp}"
    output_dir.mkdir(parents=True)
    config = vars(args).copy()
    config.update(device_used=str(device), graph_info=bundle.graph_info)
    (output_dir / "config.json").write_text(json.dumps(config, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    json_log_path = output_dir / "metrics.jsonl"
    args.log_dir.resolve().mkdir(parents=True, exist_ok=True)
    text_log_path = args.log_dir.resolve() / f"{args.dataset}.txt"

    def write_text(message):
        print(message, flush=True)
        with text_log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n\n")

    write_text(f"Run: {timestamp}  Dataset: {args.dataset}\n"
               f"Parameters: {json.dumps(vars(args), default=str, ensure_ascii=False)}\n"
               f"Dataset statistics: {json.dumps(dataset_info, ensure_ascii=False)}\n"
               f"Saved results directory: {output_dir}")

    def log(record):
        write_text(format_metrics(record))
        with json_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.KAN_model != "four_basis":
        log({"notice": "KAN_model is retained for CLI compatibility; every KEN uses all FOUR bases."})
    if args.n_sample is not None:
        log({"notice": "n_sample is deprecated; all supplied neighbor relations are used, without a top-20 cap."})
    if args.max_train_batches or args.max_topk_users:
        log({"notice": "SMOKE RUN: limited batches/users; these metrics are not a full experiment."})
    best_auc, best_epoch, stale = -float("inf"), 0, 0
    best_metrics = None
    checkpoint = output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss, pairs_seen = 0.0, 0
        for index, pairs in enumerate(loader):
            if args.max_train_batches and index >= args.max_train_batches:
                break
            negative = sampler.sample(pairs[:, 0].numpy())
            users, items = pairs[:, 0].to(device), pairs[:, 1].to(device)
            negatives = torch.from_numpy(negative).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = pair_objective(model, model(), users, items, negatives, args.lambda_low, args.lambda_high)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; check data and learning rate")
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * len(pairs)
            pairs_seen += len(pairs)
        model.eval()
        with torch.no_grad():
            user_embedding, item_embedding = model(return_ladapter=False)
            metrics = evaluate_model(model, bundle, user_embedding, item_embedding, args)
        log({"epoch": epoch, "loss": sum_loss / pairs_seen, **metrics})
        score = metrics["validation"]["AUC"]
        if not np.isfinite(score):
            raise ValueError("Invalid input")
        if score > best_auc:
            best_auc, best_epoch, stale = score, epoch, 0
            best_metrics = metrics
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if args.patience and stale >= args.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        user_embedding, item_embedding = model(return_ladapter=False)

        final = {"best_epoch": best_epoch, "auc_mode": args.auc_mode, **best_metrics}
        np.save(output_dir / "user_embeddings.npy", user_embedding.cpu().numpy())
        np.save(output_dir / "item_embeddings.npy", item_embedding.cpu().numpy())
    log(final)
    write_text(f"Saved results: {output_dir}\nSaved text log: {text_log_path}")
    return output_dir
