import argparse
import os
from os.path import join, dirname
import json
import pickle
from itertools import chain
from datetime import datetime

import torch
import pandas as pd
import numpy as np

from pyhealth.data import Patient
from pyhealth.datasets import OMOPDataset, SampleEHRDataset, get_dataloader
from pyhealth.tokenizer import Tokenizer
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn

from tqdm import tqdm
from tqdm.autonotebook import trange
import wandb

from src.ehrt.models.pyhealth_bert import Bert
from src.ehrt.models.pyhealth_mlp import MLP
from src.ehrt.train.trainer import Trainer
from src.ehrt.train.utils import set_seed, split_datasets


def get_metrics_fn(mode: str):
    if mode == "binary":
        return binary_metrics_fn
    elif mode == "multiclass":
        return multiclass_metrics_fn
    else:
        raise ValueError(f"Mode {mode} is not supported")


def main(args):
    set_seed(42)

    with open(args.dataset_path, "rb") as f:
        ds = pickle.load(f)

    ratios = [0.1, 0.7, 0.1, 0.1]
    _, ds_tr, ds_vl, ds_ts = split_datasets(ds, ratios)

    if args.seed is not None:
        set_seed(args.seed)

    with open(join(args.output_path, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project="ehrt-train-from-scratch",
        name=args.exp_name,
        config=args
        )

    with open(args.tokenizer_path, 'rb') as f:
        feat_tokenizers = pickle.load(f)

    dl_tr = get_dataloader(ds_tr, batch_size=args.batch_size, shuffle=True)
    dl_vl = get_dataloader(ds_vl, batch_size=args.batch_size, shuffle=False)
    dl_ts = get_dataloader(ds_ts, batch_size=args.batch_size, shuffle=False)

    model = Bert(
        dataset=ds,
        feature_keys=args.feature_keys,
        feat_tokenizers=feat_tokenizers,
        embedding_dim=args.embedding_dim
    )
    model.switch_to_finetune(ds, args.label_key, args.mode, args.dropout)

    for param in model.parameters():
        param.requires_grad = True

    model.to(device)

    # set optimizer    
    param = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_class = torch.optim.AdamW
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in param if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer_params = {
        "lr": args.lr,
        "weight_decay": args.weight_decay
        }
    optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

    # initialize
    data_iterator = iter(dl_tr)
    best_score = -1 * float("inf") if args.monitor_criterion == "max" else float("inf")
    # best_val_loss = float("inf")
    steps_per_epoch = len(dl_tr)
    global_step = 0

    # epoch training loop
    for epoch in range(args.num_epoch):
        train_loss_all = []
        model.zero_grad()
        model.train()
        # batch training loop
        for _ in trange(
            steps_per_epoch,
            desc=f"Epoch {epoch} / {args.num_epoch}",
            smoothing=0.05,
        ):
            try:
                data = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dl_tr)
                data = next(data_iterator)
            # forward
            output = model(**data)
            loss = output["loss"]

            if not torch.isfinite(loss):
                optimizer.zero_grad()  # Clear any gradients from this batch
                continue

            # backward
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
            # update
            optimizer.step()
            optimizer.zero_grad()
            train_loss_all.append(loss.item())
            global_step += 1

        # log and save
        train_loss = sum(train_loss_all) / len(train_loss_all)

        wandb.log({
            "train_loss": train_loss,
            "epoch": epoch
        })

        # ckpt_path = join(args.output_path, "last.ckpt")
        # torch.save(model.state_dict(), ckpt_path)

        # validation
        val_loss_all = []
        y_prob_all = []
        y_true_all = []

        for data in tqdm(dl_vl, desc="Evaluation"):
            model.eval()
            with torch.no_grad():
                output = model(**data)

                loss = output["loss"]
                y_prob = output["y_prob"].cpu().numpy()
                y_true = output["y_true"].cpu().numpy()

                val_loss_all.append(loss.item())
                y_prob_all.append(y_prob)
                y_true_all.append(y_true)

        val_loss = sum(val_loss_all) / len(val_loss_all)
        y_prob_all = np.concatenate(y_prob_all, axis=0)
        y_true_all = np.concatenate(y_true_all, axis=0)

        metrics_fn = get_metrics_fn(args.mode)
        scores = metrics_fn(y_true_all, y_prob_all, metrics=args.metrics)

        wandb.log({
            "val_loss": val_loss,
            **scores,
            "epoch": epoch
        })

        # save best model
        if scores[args.monitor] > best_score:
            best_score = scores[args.monitor]
            
            wandb.log({
                f"best_{args.monitor}": best_score,
                "best_epoch": epoch,
                "epoch": epoch
            })

            ckpt_path = join(args.output_path, "best.ckpt")
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(
        torch.load(join(args.output_path, "best.ckpt"), weights_only=True)
        )
    model.to(device)

    # test
    test_loss_all = []
    y_prob_all = []
    y_true_all = []

    for data in tqdm(dl_ts, desc="Evaluation"):
        model.eval()
        with torch.no_grad():
            output = model(**data)

            loss = output["loss"]
            y_prob = output["y_prob"].cpu().numpy()
            y_true = output["y_true"].cpu().numpy()

            test_loss_all.append(loss.item())
            y_prob_all.append(y_prob)
            y_true_all.append(y_true)

    test_loss = sum(test_loss_all) / len(test_loss_all)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)

    metrics_fn = get_metrics_fn(args.mode)
    scores = metrics_fn(y_true_all, y_prob_all, metrics=args.metrics)
    scores = {f"test_{key}": value for key, value in scores.items()}

    wandb.log({
        "test_loss": test_loss,
        **scores,
        "epoch": epoch
    })


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--p", type=float, default=0.20)
    parser.add_argument("--embedding_dim", type=int, default=128)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-01)   
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    
    parser.add_argument(
        "--task_fn",
        type=str,
        choices=[
            "readmission_prediction_omop_fn",
            "mortality_prediction_omop_fn",
            "length_of_stay_prediction_omop_fn"
            ])
    # parser.add_argument("--label_key", type=str, default="label")
    # parser.add_argument("--mode", type=str, default="binary")

    parser.add_argument("--dataset_path", type=str, required=True)
    # parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str)

    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    args.output_path = join(args.output_path, args.exp_name)
    os.makedirs(args.output_path, exist_ok=True)

    args.tables = ["condition_occurrence", "procedure_occurrence", "drug_exposure"]
    args.feature_keys = ["conditions", "procedures", "drugs"]

    args.label_key = "label"
    args.mode = "multiclass" if args.task_fn == "length_of_stay_prediction_omop_fn" else "binary"
    args.metrics = ["pr_auc", "roc_auc", "f1", "accuracy"] if args.mode == "binary" else ["roc_auc_macro_ovo", "roc_auc_macro_ovr", "f1_macro", "f1_micro", "accuracy"]
    args.monitor = args.metrics[0]
    args.monitor_criterion = "max"

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)