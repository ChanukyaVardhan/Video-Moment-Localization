import argparse
import json
import os
from collections import defaultdict

import torch
import yaml
from dataset import ActivityNet, CharadesSTA, TACoS
from models import SMIN
from torch.utils.data import DataLoader
from utils import compute_ious


def get_parameters():
        parser  = argparse.ArgumentParser()
        parser.add_argument("--config_path", default = "config/charadessta.yml", help = "Path to config file.")
        parser.add_argument("--num_epochs",  default = 0, type = int, help = "Number of epochs to override value in the config.")
        parser.add_argument("--test",            default = False, action = "store_true", help = "Test the saved model for this config.")
        parser.add_argument("--fian", default = False, action = "store_true", help ="Integrate Fian model")
        parser.add_argument("--nms", default = False, action = "store_true", help ="If Evaluation should use nms")
        parser.add_argument("--num_heads",default = 1, type = int, help = "Number of attention heads") 
        parser.add_argument("--test_model_path", default = "./checkpoints/charadessta_model_7.pt", help = "Path to saved model.")
        parser.add_argument("--experiment", default = "", help = "checkpoint names.")
        args    = parser.parse_args()

        with open(args.config_path, "r") as f:
                params = yaml.load(f, Loader=yaml.SafeLoader)
        params["experiment"] = os.path.splitext(os.path.basename(args.config_path))[0]
        params["test"]           = args.test
        params["fian"]           = args.fian
        params["nms"]            = args.nms
        params["num_heads"]      = args.num_heads
        params["test_model_path"] = args.test_model_path

        if args.num_epochs != 0:
                params["num_epochs"] = args.num_epochs
        if args.experiment != "":
                params["experiment"] = args.experiment
        return params

def get_dataset(params):
        dataset = None
        if params["dataset"] == "charadessta":
                dataset = CharadesSTA
        elif params["dataset"] == "activitynet":
                dataset = ActivityNet
        elif params["dataset"] == "tacos":
                dataset = TACoS
        else:
                raise Exception(f'Dataset {params["dataset"]} is not a valid dataset!')

        return dataset

def get_training_datasets(params):
        dataset                 = get_dataset(params)
        # CHCEK - FIX EVAL DATASET FOR CHARADESSTA BY SPLITTING TRAIN SET?
        train_dataset   = dataset(params["data_dir"], params["T"], params["L"], params["max_query_length"], split = "train")
        eval_dataset    = dataset(params["data_dir"], params["T"], params["L"], params["max_query_length"], split = "test" if params["dataset"] == "charadessta" else "val")

        return train_dataset, eval_dataset

def get_test_dataset(params):
        dataset                 = get_dataset(params)
        test_dataset    = dataset(params["data_dir"], params["T"], params["L"], params["max_query_length"], split = "test")

        return test_dataset

def get_dataloader(params, dataset, shuffle = False, half = False):
        dataloader = DataLoader(
                dataset,
                batch_size      = params["batch_size"] if not half else params["batch_size"] // 2,
                shuffle         = shuffle,
                collate_fn      = dataset.collate_fn,
                num_workers = params["num_workers"],
        )

        return dataloader

def get_model(params):
        model = None
        if params["model"] == "SMIN":
                model = SMIN(params["T"], params["L"], params["C"], params["d"], params["dl"], params["num_heads"], params["fian"], params["num_smi_layers"], params["input_video_dim"], params["max_query_length"], params["lstm_hidden_size"], params["device"])
        else:
                raise Exception(f'Model {params["model"]} is not a valid model!')

        return model

def get_optimizer(model, params):
        optimizer = None
        if params["optimizer"] == "Adam":
                optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr = params["lr"]
                        #lr = 0.0001
                )
        else:
                raise Exception(f'Optimizer {params["optimizer"]} is not supported!')

        return optimizer

def bce_loss(p, y, s, mask): 
    # p: (B, L, L), y: (B, L, L), mask: s: (B, L, L) -> loss: (B, L, L)
    if s is not None:
        #loss_layer_1 = torch.nn.BCELoss(weight=s*y.long(), reduction = 'none')
        #loss_layer_2 = torch.nn.BCELoss(weight=(1-s)*(1-y.long()), reduction ='none')
        loss_layer_1 = torch.nn.BCELoss(weight=s, reduction = 'none')
        loss_layer_2 = torch.nn.BCELoss(weight=(1-s), reduction ='none')        
        loss = loss_layer_1(p, y.float())+ loss_layer_2(1-p, 1-y.float())
        loss = loss * mask
    else:
        loss_layer = torch.nn.BCELoss(reduction = 'none')
        loss = loss_layer(p, y.float()) * mask

    if mask.dim() == 3: # L_m case
        loss = torch.sum(loss, dim = (1, 2)) / torch.sum(mask, dim = (1, 2))
    else: # L_s, L_e, L_a cases
        loss = torch.sum(loss, dim = 1) / torch.sum(mask, dim = 1)

    # B x 1 -> 1
    loss = torch.mean(loss)

    return loss

def loss_fn(pm, ym, sm, moment_mask, ps, ys, ss, pe, ye, se, pa, ya, length_mask):
        L_m = bce_loss(pm, ym, sm, moment_mask)
        L_s = bce_loss(ps, ys, ss, length_mask)
        L_e = bce_loss(pe, ye, se, length_mask)
        L_a = bce_loss(pa, ya, None, length_mask)

        return L_m + L_s + L_e + 0.5 * L_a

def get_batch_entries(batch, device):
        video_features  = batch["video_features"].to(device)
        video_mask              = batch["video_mask"].to(device)
        query_features  = batch["query_features"].to(device)
        query_mask              = batch["query_mask"].to(device)
        length_mask     = batch["length_mask"].to(device)
        moment_mask     = batch["moment_mask"].to(device)
        sm                              = batch["sm"].to(device)
        ym                              = batch["ym"].to(device)
        ss                              = batch["ss"].to(device)
        ys                              = batch["ys"].to(device)
        se                              = batch["se"].to(device)
        ye                              = batch["ye"].to(device)
        ya                              = batch["ya"].to(device)

        return video_features, video_mask, query_features, query_mask, length_mask, moment_mask, sm, ym, ss, ys, se, ye, ya

def train_epoch(model, optimizer, train_dataloader, device, params, n = [1, 5], m = [0.1, 0.3, 0.5, 0.7]):
        model.train()
        train_loss              = 0.0
        iou_metrics     = defaultdict(lambda: 0.0)
        num_samples     = 0

        for i, batch in enumerate(train_dataloader):
                video_features, video_mask, query_features, query_mask, length_mask, moment_mask, sm, ym, ss, ys, se, ye, ya = get_batch_entries(batch, device)
                batch_size              = video_features.shape[0]

                optimizer.zero_grad()

                pm, ps, pe, pa  = model(video_features, video_mask, query_features, query_mask, length_mask, moment_mask)

                loss                    = loss_fn(pm, ym, sm, moment_mask, ps, ys, ss, pe, ye, se, pa, ya, length_mask)

                train_loss         += loss.item()*batch_size
                # CHECK - COMPUTE IOU AS ACCURACY HERE?

                iou_batch               = compute_ious(pm, ps, pe, moment_mask, sm, n, m)
                iou_metrics             = {k: iou_metrics[k] + iou_batch[k] for k in iou_batch.keys()}

                loss.backward()
                optimizer.step()

                num_samples    += batch_size

        train_loss /= num_samples
        iou_metrics             = {k: iou_metrics[k] / num_samples for k in iou_metrics.keys()}

        torch.cuda.empty_cache()

        return train_loss, iou_metrics

def eval_epoch(model, eval_dataloader, device, params, n = [1, 5], m = [0.1, 0.3, 0.5, 0.7]):
        model.eval()
        eval_loss               = 0.0
        iou_metrics     = defaultdict(lambda: 0.0)
        num_samples     = 0

        for i, batch in enumerate(eval_dataloader):
                video_features, video_mask, query_features, query_mask, length_mask, moment_mask, sm, ym, ss, ys, se, ye, ya = get_batch_entries(batch, device)
                batch_size              = video_features.shape[0]

                pm, ps, pe, pa  = model(video_features, video_mask, query_features, query_mask, length_mask, moment_mask)

                loss                    = loss_fn(pm, ym, sm, moment_mask, ps, ys, ss, pe, ye, se, pa, ya, length_mask)

                eval_loss          += loss.item()*batch_size

                iou_batch               = compute_ious(pm, ps, pe, moment_mask, sm, n, m)
                iou_metrics             = {k: iou_metrics[k] + iou_batch[k] for k in iou_batch.keys()}

                num_samples    += batch_size

        eval_loss /= num_samples
        iou_metrics             = {k: iou_metrics[k] / num_samples for k in iou_metrics.keys()}

        torch.cuda.empty_cache()

        return eval_loss, iou_metrics

def test_model(model, test_dataloader, device, params, n = [1, 5], m = [0.1, 0.3, 0.5, 0.7]):
        model.eval()
        iou_metrics = defaultdict(lambda: 0.0)
        num_samples = 0

        for i, batch in enumerate(test_dataloader):
                video_features, video_mask, query_features, query_mask, length_mask, moment_mask, sm, _, _, _, _, _, _ = get_batch_entries(batch, device)
                batch_size              = video_features.shape[0]

                pm, ps, pe, _   = model(video_features, video_mask, query_features, query_mask, length_mask, moment_mask)
                #extract indexs of pm, ps, pe
                
                iou_batch               = compute_ious(pm, ps, pe, moment_mask, sm, n, m, device, params["nms"])
                #boxes dimension: [N,4], scores: [N]
                #torchvision.ops.nms(boxes: Tensor, scores: Tensor, iou_threshold: float) 
                iou_metrics             = {k: iou_metrics[k] + iou_batch[k] for k in iou_batch.keys()}

                num_samples    += batch_size
                #print(iou_metrics)

        iou_metrics             = {k: iou_metrics[k] / num_samples for k in iou_metrics.keys()}

        return iou_metrics

def get_prefix(params):
        prefix                  = f'{params["checkpoint_path"]}/{params["experiment"]}_'

        return prefix

def get_save_paths(params):
        prefix                  = get_prefix(params)
        model_path              = f'{prefix}model.pt'
        train_stat_path = f'{prefix}stats.json'

        return model_path, train_stat_path

def get_existing_stats(train_stat_path, start_epoch, params):
        train_stats = defaultdict(lambda: [])

        if params["resume_training"] and os.path.exists(train_stat_path):
                print('entering this getexistingstats?')
                existing_stats = json.load(open(train_stat_path, "r"))

                for key, val in existing_stats.items():
                        train_stats[key] = val[:start_epoch - 1]

        return train_stats

def train_model(model, train_dataloader, eval_dataloader, device, params):
        model.train()

        start_epoch = 1
        optimizer       = get_optimizer(model, params)
        model_path, train_stat_path = get_save_paths(params)
        if params["resume_training"] and os.path.exists(model_path):
                print('entering this initial training point?')
                model_details   = torch.load(model_path)
                start_epoch             = model_details["epoch"] + 1 # Start from the epoch after the checkpoint
                model.load_state_dict(model_details["model"])
                optimizer.load_state_dict(model_details["optimizer"])
        
        train_stats = get_existing_stats(train_stat_path, start_epoch, params)

        for epoch in range(start_epoch, params["num_epochs"] + 1):
                print(f"Training Epoch - {epoch}")
                train_loss, train_iou_metrics   = train_epoch(model, optimizer, train_dataloader, device, params)
                # HAVE EVAL AFTER A FEW EPOCHS RATHER THAN EVERY EPOCH??
                eval_loss, eval_iou_metrics     = eval_epoch(model, eval_dataloader, device, params)

                # Print Stats
                print(f"Training Loss - {train_loss:.4f}, Eval Loss - {eval_loss:.4f}")
                for k, v in train_iou_metrics.items():
                        print(f"train_{k} - {v}")
                for k, v in eval_iou_metrics.items():
                        print(f"eval_{k} - {v}")
                
                train_stats["epoch"].append(epoch)
                train_stats["train_loss"].append(train_loss)
                train_stats["eval_loss"].append(eval_loss)
                for k, v in train_iou_metrics.items():
                        train_stats[f"train_{k}"].append(v)
                for k, v in eval_iou_metrics.items():
                        train_stats[f"eval_{k}"].append(v)

                with open(train_stat_path, "w") as f:
                        json.dump(train_stats, f)

                # FIX - SAVE MODEL AND OPTIMIZER ON SOME CONDITION, ALSO SAVE THE CONDITION IN THE PATH AS WELL TO RELOAD FROM THAT CONDITION
                # SAVING EVERY EPOCH
                current_epoch_path = get_prefix(params) + f"model_{epoch}.pt"
                torch.save({
                        "epoch":                epoch,
                        "model":                model.state_dict(),
                        "optimizer":    optimizer.state_dict()
                # }, model_path) # -> best path
                }, current_epoch_path) # -> save every epoch

        return model

if __name__ == "__main__":
        params = get_parameters()
        # Set seed
        torch.manual_seed(params["seed"])
        torch.cuda.manual_seed_all(params["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params["device"] = device

        model   = get_model(params)
        model   = model.to(device)

        if not params["test"]: # Training
                train_dataset, eval_dataset             = get_training_datasets(params)
                train_dataloader                                        = get_dataloader(params, train_dataset, shuffle = True)
                eval_dataloader                                         = get_dataloader(params, eval_dataset, shuffle = False, half = True)

                train_model(model, train_dataloader, eval_dataloader, device, params)
        else: # Test the model
                test_dataset            = get_test_dataset(params)
                test_dataloader         = get_dataloader(params, test_dataset, shuffle = False)

                # Load state dict of the saved model
                # model_path = f'{params["checkpoint_path"]}/{params["experiment"]}_model.pt'
                model_path = params["test_model_path"]
                if os.path.exists(model_path):
                        model.load_state_dict(torch.load(model_path)["model"])
                else:
                        raise Exception(f'No saved model at {model_path}!')                
                iou_metrics = test_model(model, test_dataloader, device, params)

                for k, v in iou_metrics.items():
                        print(f"{k} - {v}")
