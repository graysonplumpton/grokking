import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import AbstractDataset
from utils import combine_logs
from tqdm.auto import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objs import load_item


class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super().__init__()
        assert split in {'train', 'val'}
        self.dataset = dataset
        self.fetch_f = dataset.fetch_train_example if split == 'train' else dataset.fetch_val_example

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)


def evaluate_accuracy(model, dataloader, device, max_batches=None):
    """Evaluate accuracy on a dataloader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits[:, -1, :].argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total if total > 0 else 0.0


def run_robustness_tests(model, val_dataloader, device, config, step):
    """Run robustness tests on output layer weights."""
    model.eval()
    output_layer = model.transformer.output
    eval_batches = config['train']['eval_batches']
    
    baseline_acc = evaluate_accuracy(model, val_dataloader, device, max_batches=eval_batches)
    
    with torch.no_grad():
        W_original = output_layer.weight.data.clone()
        U, S, Vh = torch.linalg.svd(W_original, full_matrices=False)
        total_rank = len(S)
        
        # Test 1: SVD Truncation (keep top k)
        svd_trunc_results = {}
        for k in [5, 10, 20, 50]:
            if k > total_rank:
                continue
            S_trunc = S.clone()
            S_trunc[k:] = 0
            output_layer.weight.data = U @ torch.diag(S_trunc) @ Vh
            svd_trunc_results[k] = evaluate_accuracy(model, val_dataloader, device, max_batches=eval_batches)
            output_layer.weight.data = W_original
        
        # Test 2: SVD Thresholding (zero values below threshold)
        svd_thresh_results = {}
        for thresh in [0.1, 0.5, 1.0, 2.0]:
            S_thresh = S.clone()
            S_thresh[S_thresh < thresh] = 0
            kept = (S >= thresh).sum().item()
            output_layer.weight.data = U @ torch.diag(S_thresh) @ Vh
            svd_thresh_results[thresh] = (kept, evaluate_accuracy(model, val_dataloader, device, max_batches=eval_batches))
            output_layer.weight.data = W_original
    
    # Print summary
    print(f"\n┌{'─'*60}┐")
    print(f"│ Robustness Tests @ Step {step:<33}│")
    print(f"├{'─'*60}┤")
    print(f"│ Baseline Accuracy: {baseline_acc:<39.4f}│")
    print(f"│ Total Singular Values: {total_rank:<35}│")
    print(f"├{'─'*60}┤")
    print(f"│ SVD Truncation (keep top k):{'':>31}│")
    for k, acc in svd_trunc_results.items():
        delta = acc - baseline_acc
        print(f"│   k={k:<3} → {acc:.4f}  ({delta:+.4f}){'':<27}│")
    print(f"├{'─'*60}┤")
    print(f"│ SVD Thresholding (zero if < threshold):{'':>19}│")
    for thresh, (kept, acc) in svd_thresh_results.items():
        delta = acc - baseline_acc
        print(f"│   τ={thresh:<4} → {acc:.4f}  ({delta:+.4f})  kept {kept}/{total_rank}{'':<13}│")
    print(f"└{'─'*60}┘\n")
    
    model.train()
    return {'baseline': baseline_acc, 'total_rank': total_rank, 
            'svd_truncation': svd_trunc_results, 'svd_threshold': svd_thresh_results}


def truncate_output_layer(model, k):
    """Truncate output layer weights to top-k singular values."""
    output_layer = model.transformer.output
    
    with torch.no_grad():
        W = output_layer.weight.data
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        original_norm = W.norm().item()
        original_rank = (S > 1e-6).sum().item()
        
        S_trunc = S.clone()
        S_trunc[k:] = 0
        W_trunc = U @ torch.diag(S_trunc) @ Vh
        
        truncated_norm = W_trunc.norm().item()
        energy_retained = (S[:k]**2).sum() / (S**2).sum()
        
        output_layer.weight.data = W_trunc
        
        print(f"\n┌{'─'*60}┐")
        print(f"│ SVD Truncation Applied{'':>37}│")
        print(f"├{'─'*60}┤")
        print(f"│ Shape: {str(tuple(W.shape)):<51}│")
        print(f"│ Kept: {k} / {len(S)} singular values{'':<32}│"[:62] + "│")
        print(f"│ Effective rank: {original_rank:<42}│")
        print(f"│ Frobenius norm: {original_norm:.4f} → {truncated_norm:.4f}{'':<24}│"[:62] + "│")
        print(f"│ Energy retained: {energy_retained*100:.1f}%{'':<40}│"[:62] + "│")
        print(f"└{'─'*60}┘\n")
    
    return output_layer


def reset_optimizer_for_layer(optim, layer):
    """Reset optimizer state for a specific layer."""
    reset_count = 0
    for param in layer.parameters():
        if param in optim.state:
            state = optim.state[param]
            if 'exp_avg' in state:
                state['exp_avg'].zero_()
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'].zero_()
            if 'step' in state:
                state['step'] = torch.tensor(0.0)
            reset_count += 1
    print(f"Reset optimizer state for {reset_count} parameter tensors")


def train(config):
    print('Using config:', config)
    train_cfg = config['train']
    wandb_cfg = config['wandb']
    
    truncate_at_step = config.get('truncate_at_step', None)
    truncate_k = config.get('truncate_k', 20)
    reset_optimizer_on_truncate = config.get('reset_optimizer_on_truncate', True)
    
    if truncate_at_step is not None:
        print(f"\n*** SVD truncation enabled: k={truncate_k} at step {truncate_at_step} ***\n")
    
    if wandb_cfg['use_wandb']:
        wandb.init(project=wandb_cfg['wandb_project'], config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_item(config['dataset'])
    train_data = GroupDataset(dataset, 'train')
    val_data = GroupDataset(dataset, 'val')
    train_dataloader = DataLoader(train_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    val_dataloader = DataLoader(val_data, num_workers=train_cfg['num_workers'], batch_size=train_cfg['bsize'])
    
    model = load_item(config['model'], dataset.n_vocab, dataset.n_out, device)
    model.train()
    
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], 
                              weight_decay=train_cfg['weight_decay'], betas=train_cfg['betas'])
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: min(s / train_cfg['warmup_steps'], 1)
    )
    
    step = 0
    truncation_applied = False

    for x, y in train_dataloader:
        # Apply truncation if scheduled
        if truncate_at_step and not truncation_applied and step >= truncate_at_step:
            output_layer = truncate_output_layer(model, truncate_k)
            if reset_optimizer_on_truncate:
                reset_optimizer_for_layer(optim, output_layer)
            truncation_applied = True
            
            if wandb_cfg['use_wandb']:
                wandb.log({"truncation/applied_at_step": step, "truncation/k": truncate_k}, step=step)
        
        # Training step
        loss, logs = model.get_loss(x.to(device), y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schedule.step()
        
        # Evaluation
        if (step + 1) % train_cfg['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                all_val_logs = []
                for i, (val_x, val_y) in enumerate(val_dataloader):
                    if i >= train_cfg['eval_batches']:
                        break
                    _, val_logs = model.get_loss(val_x.to(device), val_y.to(device))
                    all_val_logs.append(val_logs)
            
            out_log = {'val': combine_logs(all_val_logs), 'train': combine_logs([logs]), 
                       'step': step + 1, 'lr': float(lr_schedule.get_last_lr()[0])}
            
            if wandb_cfg['use_wandb']:
                wandb.log(out_log)
            model.train()

        # Periodic summary
        if (step + 1) % 2000 == 0:
            model.eval()
            with torch.no_grad():
                val_logs = []
                for i, (val_x, val_y) in enumerate(val_dataloader):
                    if i >= train_cfg['eval_batches']:
                        break
                    val_logs.append(model.get_loss(val_x.to(device), val_y.to(device))[1])
                val_combined = combine_logs(val_logs)
            
            print(f"\n{'='*60}")
            print(f"Step {step + 1:,}")
            if truncation_applied:
                print(f"[Truncated to k={truncate_k} at step {truncate_at_step}]")
            print(f"Val Accuracy: {val_combined.get('accuracy', 0.0):.4f}")
            print(f"{'='*60}")
            
            run_robustness_tests(model, val_dataloader, device, config, step + 1)
            model.train()

        step += 1
        if train_cfg['max_steps'] and step >= train_cfg['max_steps']:
            break

    print(f"\nTraining completed at step {step}")
    if wandb_cfg['use_wandb']:
        wandb.finish()


@hydra.main(config_path="../config", config_name="train_grokk")
def main(cfg: DictConfig):
    train(OmegaConf.to_container(cfg))


if __name__ == "__main__":
    main()
