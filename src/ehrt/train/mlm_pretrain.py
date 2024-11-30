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

from tqdm import tqdm
from tqdm.autonotebook import trange
import wandb

from src.ehrt.models.pyhealth_bert import Bert
from src.ehrt.train.trainer import Trainer
from src.ehrt.train.utils import set_seed, split_datasets


def mlm_pretrain_omop_fn(patient: Patient):
    samples = []

    for visit in patient:

        conditions = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="drug_exposure")

        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
            }
        )

    return samples


def main(args):
    set_seed(42)

    with open(join(args.output_path, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project="ehrt-pretrain",
        name=args.exp_name,
        config=args
        )

    with open(args.tokenizer_path, 'rb') as f:
        feat_tokenizers = pickle.load(f)

    with open(args.fake_dataset_path, "rb") as f:
        ds_fake = pickle.load(f)

    with open(args.real_dataset_path, "rb") as f:
        ds_real = pickle.load(f)

    ratios = [0.1, 0.9]
    ds_vl_1, _ = split_datasets(ds_real, ratios)

    dl_tr_1 = get_dataloader(ds_fake, batch_size=args.batch_size, shuffle=True)
    dl_vl_1 = get_dataloader(ds_vl_1, batch_size=args.batch_size, shuffle=False)

    model = Bert(
        dataset=ds_fake,
        feature_keys=args.feature_keys,
        feat_tokenizers=feat_tokenizers,
        embedding_dim=args.embedding_dim,
        mlm_probability=args.p
    ).to(device)

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
    data_iterator = iter(dl_tr_1)
    # best_score = -1 * float("inf") if args.monitor_criterion == "max" else float("inf")
    best_val_loss = float("inf")
    steps_per_epoch = len(dl_tr_1)
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
                data_iterator = iter(dl_tr_1)
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

        ckpt_path = join(args.output_path, "last.ckpt")
        torch.save(model.state_dict(), ckpt_path)

        # validation
        val_loss_all = []
        for data in tqdm(dl_vl_1, desc="Evaluation"):
            model.eval()
            with torch.no_grad():
                output = model(**data)
                loss = output["loss"]
                val_loss_all.append(loss.item())
        
        val_loss = sum(val_loss_all) / len(val_loss_all)
      
        wandb.log({
            "val_loss": val_loss,
            "epoch": epoch
        })

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            wandb.log({
                "best_val_loss": best_val_loss,
                "best_epoch": epoch,
                "epoch": epoch
            })

            ckpt_path = join(args.output_path, "best.ckpt")
            torch.save(model.state_dict(), ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--p", type=float, default=0.20)
    parser.add_argument("--embedding_dim", type=int, default=128)

    # parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-01)   
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    
    # parser.add_argument(
    #     "--task_fn",
    #     type=str,
    #     choices=[
    #         "readmission_prediction_omop_fn",
    #         "mortality_prediction_omop_fn",
    #         "length_of_stay_prediction_omop_fn"
    #         ])
    # parser.add_argument("--label_key", type=str, default="label")
    # parser.add_argument("--mode", type=str, default="binary")

    parser.add_argument("--fake_dataset_path", type=str, required=True)
    parser.add_argument("--real_dataset_path", type=str, required=True)
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
    # args.mode = "multiclass" if args.task_fn == "length_of_stay_prediction_omop_fn" else "binary"
    # args.metrics = ["roc_auc", "pr_auc", "f1", "accuracy"] if args.mode == "binary" else ["roc_auc_macro_ovo", "roc_auc_macro_ovr", "f1_macro", "f1_micro", "accuracy"]
    # args.monitor = args.metrics[0]
    # args.monitor_criterion = "max"

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)