import argparse
from os.path import join
import time
import pickle
import random
import math
import itertools

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataListLoader as GraphLoader
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import pandas as pd

from src.ehrt.train.utils import set_seed, count_parameters
from src.ehrt.dataset.gdset import GDSet

#TODO: Scheduler


def build_dataloaders(args):
    with open(args.data_path, 'rb') as handle:
        dataset = pickle.load(handle)
    
    rs1 = ShuffleSplit(n_splits=1, test_size=.20, random_state=random_seed)
    indices_dev, _ = next(rs1.split(dataset))

    rs2 = ShuffleSplit(n_splits=1, test_size=.10/.80, random_state=random_seed)
    indices_tr, indices_vl = next(rs2.split(indices_dev))

    dataset_tr = [dataset[x] for x in indices_tr]
    dataset_vl = [dataset[x] for x in indices_vl]

    dataloader_tr = GraphLoader(GDSet(dataset_tr), batch_size=args.batch_size, shuffle=True)
    dataloader_vl = GraphLoader(GDSet(dataset_vl), batch_size=args.batch_size, shuffle=False)

    return dataloader_tr, dataloader_vl


def run_epoch(model, dataloader_tr, optimizer, device, args):
    loss_tr = 0
    start = time.time()
    model.train()

    for step, data in enumerate(dataloader_tr):
        optimizer.zero_grad()

        batched_data = Batch()
        graph_batch = batched_data.from_data_list(list(itertools.chain.from_iterable(data)))
        graph_batch = graph_batch.to(device)
        nodes = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_index_readout = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch = graph_batch.batch
        age_ids = torch.reshape(graph_batch.age, [graph_batch.age.shape[0] // args.max_num_token, args.max_num_token])
        time_ids = torch.reshape(graph_batch.time, [graph_batch.time.shape[0] // args.max_num_token, args.max_num_token])
        delta_ids = torch.reshape(graph_batch.delta, [graph_batch.delta.shape[0] // args.max_num_token, args.max_num_token])
        type_ids = torch.reshape(graph_batch.adm_type, [graph_batch.adm_type.shape[0] // args.max_num_token, args.max_num_token])
        posi_ids = torch.reshape(graph_batch.posi_ids, [graph_batch.posi_ids.shape[0] // args.max_num_token, args.max_num_token])
        attMask = torch.reshape(graph_batch.mask_v, [graph_batch.mask_v.shape[0] // args.max_num_token, args.max_num_token])
        attMask = torch.cat((torch.ones((attMask.shape[0], 1)).to(device), attMask), dim=1)
        los = torch.reshape(graph_batch.los, [graph_batch.los.shape[0] // args.max_num_token, args.max_num_token])

        labels = torch.reshape(graph_batch.label, [graph_batch.label.shape[0] // args.max_num_token, args.max_num_token])[:, 0].float()
        masks = torch.reshape(graph_batch.mask, [graph_batch.mask.shape[0] // args.max_num_token, args.max_num_token])[:, 0]
        loss, logits = model(nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids,delta_ids,type_ids,posi_ids,attMask, labels, masks, los)

        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps

        loss.backward()
        loss_tr += loss.item()

        if step % 500 == 0:
            print(loss.item())

        optimizer.step()
        #sched.step()
        del loss
        #result = result + torch.sum(torch.sum(torch.mul(torch.abs(torch.subtract(pred, label)), target_mask), dim = 0)).cpu()
        #sum_labels = sum_labels + torch.sum(target_mask, dim=0).cpu()
    #print(result / sum_labels)

    loss_tr *= args.batch_size / len(dataloader_tr)
    time_cost_tr = time.time() - start
    return loss_tr, time_cost_tr


def eval(model, dataloader_vl, device, args):
    loss_vl = 0
    tr_g_loss = 0
    tr_d_un = 0
    tr_d_sup = 0
    temp_loss = 0
    start = time.time()
    model.eval()

    # with open(join(args.output_path, "preds", "v_behrt_preds.csv"), 'w') as f:
    #     f.write('')
    # with open(join(args.output_path, "preds", "v_behrt_labels.csv"), 'w') as f:
    #     f.write('')
    # with open(join(args.output_path, "preds", "v_behrt_masks.csv"), 'w') as f:
    #     f.write('')

    for step, data in enumerate(dataloader_vl):
        batched_data = Batch()
        graph_batch = batched_data.from_data_list(list(itertools.chain.from_iterable(data)))
        graph_batch = graph_batch.to(device)
        nodes = graph_batch.x
        edge_index = graph_batch.edge_index
        edge_index_readout = graph_batch.edge_index
        edge_attr = graph_batch.edge_attr
        batch = graph_batch.batch
        age_ids = torch.reshape(graph_batch.age, [graph_batch.age.shape[0] // args.max_num_token, args.max_num_token])
        time_ids = torch.reshape(graph_batch.time, [graph_batch.time.shape[0] // args.max_num_token, args.max_num_token])
        delta_ids = torch.reshape(graph_batch.delta, [graph_batch.delta.shape[0] // args.max_num_token, args.max_num_token])
        type_ids = torch.reshape(graph_batch.adm_type, [graph_batch.adm_type.shape[0] // args.max_num_token, args.max_num_token])
        posi_ids = torch.reshape(graph_batch.posi_ids, [graph_batch.posi_ids.shape[0] // args.max_num_token, args.max_num_token])
        attMask = torch.reshape(graph_batch.mask_v, [graph_batch.mask_v.shape[0] // args.max_num_token, args.max_num_token])
        attMask = torch.cat((torch.ones((attMask.shape[0], 1)).to(device), attMask), dim=1)
        los = torch.reshape(graph_batch.los, [graph_batch.los.shape[0] // args.max_num_token, args.max_num_token])

        labels = torch.reshape(graph_batch.label, [graph_batch.label.shape[0] // args.max_num_token, args.max_num_token])[:, 0].float()
        masks = torch.reshape(graph_batch.mask, [graph_batch.mask.shape[0] // args.max_num_token, args.max_num_token])[:, 0]
        loss, logits = model(nodes, edge_index, edge_index_readout, edge_attr, batch, age_ids, time_ids,delta_ids,type_ids,posi_ids,attMask, labels, masks, los)

        # with open(join(args.output_path, "preds", "v_behrt_preds.csv"), 'a') as f:
        #     pd.DataFrame(logits.detach().cpu().numpy()).to_csv(f, header=False)
        # with open(join(args.output_path, "preds", "v_behrt_labels.csv"), 'a') as f:
        #     pd.DataFrame(labels.detach().cpu().numpy()).to_csv(f, header=False)
        # with open(join(args.output_path, "preds", "v_behrt_masks.csv"), 'a') as f:
        #     pd.DataFrame(masks.detach().cpu().numpy()).to_csv(f, header=False)

        loss_vl += loss.item()
        del loss

    loss_vl *= args.batch_size / len(dataloader_vl)
    time_cost_vl = time.time() - start
    return loss_vl, time_cost_vl, logits, labels, masks


def main():
    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.use_wandb:
        import wandb
        wandb.init(project="ehrt-train", config=args)

    dataloader_tr, dataloader_vl = build_dataloaders(args)

    model_config = BertConfig(args)
    model = BertForMTR(model_config)
    model = model.to(device)

    # criterion = WeightedMSELoss(alpha=args.alpha)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, eta_min=1e-6)

    # if args.resume_from_checkpoint:
    #     checkpoint_last_path = join(args.output_dir, "checkpoint_last.pth")
    #     if exists(checkpoint_last_path):
    #         checkpoint_last = torch.load(checkpoint_last_path)
        
    #         global_step = checkpoint_last["global_step"]
    #         best_score = checkpoint_last["best_score"]
    #         model.load_state_dict(checkpoint_last["model"])
    #         optimizer.load_state_dict(checkpoint_last["optimizer"])
    #         scheduler.load_state_dict(checkpoint_last["scheduler"])
    #         print(f"Resuming from checkpoint at step {global_step}")
    #         print(f"Best score: {best_score:.4f}")
    #     else:
    #         args.resume_from_checkpoint = None
    #         global_step = 0
    #         best_score = 0
    # else:
    #     global_step = 0
    #     best_score = 0

    # train_loader_iter = iter(dataloader_tr)

    # for global_step in progress_bar:
    #     try:
    #         images, labels = next(train_loader_iter)
    #     except StopIteration:
    #         train_loader_iter = iter(dataloader_tr)
    #         images, labels = next(train_loader_iter)

    #     loss_tr = train_step(model, images, labels, criterion, optimizer, scheduler, device)

    #     # Log training loss to wandb
    #     if args.use_wandb:
    #         wandb.log({"loss_tr": loss_tr, "global_step": global_step})

    #     # Every args.checkpointing_steps, validate and save the model
    #     if global_step % args.checkpointing_steps == 0:
    #         loss_vl, mean_corr_cells, max_corr_genes, heg_corr, hvg_corr = validate(model, dataloader_vl, criterion, device)
    #         score = (mean_corr_cells + max_corr_genes + heg_corr + hvg_corr) / 4

    #         print(f'Step [{global_step}/{args.max_train_steps}], Train Loss: {loss_tr:.4f}, Val Loss: {loss_vl:.4f}')
    #         print(f'Mean Corr (Cells): {mean_corr_cells:.4f}, Max Corr (Genes): {max_corr_genes:.4f}')
    #         print(f'HEG Corr: {heg_corr:.4f}, HVG Corr: {hvg_corr:.4f}')
    #         print(f'Score: {score:.4f}')

    #         # Log validation loss to wandb
    #         if args.use_wandb:
    #             wandb.log({
    #                 "loss_vl": loss_vl,
    #                 "mean_corr_cells": mean_corr_cells,
    #                 "max_corr_genes": max_corr_genes,
    #                 "heg_corr": heg_corr,
    #                 "hvg_corr": hvg_corr,
    #                 "score": score,
    #                 "global_step": global_step
    #             })

    #         checkpoint_path = join(args.output_dir, 'checkpoint_last.pth')
    #         os.makedirs(dirname(checkpoint_path), exist_ok=True)
    #         torch.save({
    #             'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'scheduler': scheduler.state_dict(),
    #             'global_step': global_step,
    #             'best_score': best_score
    #             }, checkpoint_path
    #             )
    #         print(f'Model saved at {checkpoint_path}')

    #         if score > best_score:
    #             best_score = score
    #             print(f'*************** NEW BEST SCORE: {best_score:.4f} ***************')

    #             checkpoint_path = join(args.output_dir, 'checkpoint_best.pth')
    #             os.makedirs(dirname(checkpoint_path), exist_ok=True)
    #             torch.save({
    #                 'model': model.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'scheduler': scheduler.state_dict(),
    #                 'global_step': global_step,
    #                 'best_score': best_score
    #                 }, checkpoint_path
    #                 )
    #             print(f'Model saved at {checkpoint_path}')


    num_params = count_parameters(model)
    if args.use_wandb:
        wandb.log({
            "num_params": num_params,
        })

    loss_vl_best = math.inf

    progress_bar = tqdm(
        range(0, args.num_epoch),
        initial=0,
        desc="Epoch",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in progress_bar:
        loss_tr, time_cost_tr = run_epoch(model, dataloader_tr, optimizer, device, args)
        loss_vl, time_cost_vl, pred, label, mask = eval(model, dataloader_vl, device, args)

        if args.use_wandb:
            wandb.log({
                "loss_tr": loss_tr,
                "loss_vl": loss_vl,
                "time_cost_tr": time_cost_tr,
                "time_cost_vl": time_cost_vl,
            })

        if loss_vl < loss_vl_best:
            model_to_save = model.module if hasattr(behrt, 'module') else model            
            torch.save(model_to_save.state_dict(), join(args.output_path, "models", "v_behrt"))
            loss_vl_best = loss_vl

            if args.use_wandb:
                wandb.log({
                    "loss_vl_best": loss_vl_best,
                    "epoch_best": epoch,
                })


def parse_args():
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument('--vocab_size', type=int, default=7204, help='Number of disease + symbols for word embedding')
    parser.add_argument('--hidden_size', type=int, default=108*5, help='Word embedding and seg embedding hidden size')
    parser.add_argument('--seg_vocab_size', type=int, default=2, help='Number of vocab for seg embedding')
    parser.add_argument('--age_vocab_size', type=int, default=103, help='Number of vocab for age embedding')
    parser.add_argument('--delta_size', type=int, default=144, help='Number of vocab for delta embedding')
    parser.add_argument('--gender_vocab_size', type=int, default=2, help='Number of vocab for gender embedding')
    parser.add_argument('--ethnicity_vocab_size', type=int, default=2, help='Number of vocab for ethnicity embedding')
    parser.add_argument('--race_vocab_size', type=int, default=6, help='Number of vocab for race embedding')
    parser.add_argument('--num_labels', type=int, default=1, help='Number of labels for the output layer')
    parser.add_argument('--feature_dict', type=int, default=7204, help='Feature dictionary size')
    parser.add_argument('--max_num_token', type=int, default=50, help='Maximum number of tokens')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2, help='Dropout rate for hidden layers')
    parser.add_argument('--graph_dropout_prob', type=float, default=0.2, help='Dropout rate for graph layers')
    parser.add_argument('--num_hidden_layers', type=int, default=6, help='Number of multi-head attention layers')
    parser.add_argument('--num_attention_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.2, help='Dropout rate for attention probabilities')
    parser.add_argument('--intermediate_size', type=int, default=512, help='Size of the intermediate layer in the transformer encoder')
    parser.add_argument('--hidden_act', type=str, default='gelu', help='Non-linear activation function (gelu, relu, swish)')
    parser.add_argument('--initializer_range', type=float, default=0.02, help='Parameter weight initializer range')
    parser.add_argument('--number_output', type=int, default=1, help='Number of output classes')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers (3-1)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value')

    # Training configuration
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-02, help='Weight decay')   
    parser.add_argument("--batch_size", type=int, default=64)    
    
    parser.add_argument("--data_path", type=str, required=True, default="/content/drive/My Drive/GANBEHRT/final_data/new/data")
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Whether or not to resume from checkpoint.")
    parser.add_argument("--use_wandb", action="store_true", help="Whether or not to use wandb.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()



