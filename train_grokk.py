import torch
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import IterableDataset
from datasets import AbstractDataset
from utils import combine_logs
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from load_objs import load_item

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go
import plotly.express as px

class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {'train', 'val'}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == 'train':
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == 'val':
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)

def create_circle_matrix(height):
  
    n_out = 51
    idx = 0
  
    A = torch.zeros(n_out, 3)  # 3D coordinates (x, y, z)

    # === BOTTOM CIRCLE (z = -r/3) ===
    z_bottom = -height / 3
    
    # Bottom center: 2 points at origin
    for i in range(2):
        A[idx, :] = torch.tensor([0, 0, z_bottom])
        idx += 1
    
    # Bottom circle: 16 islands with 2 points each
    angles = torch.linspace(0, 2*torch.pi, 16)

    for angle in angles:
        for j in range(2):
            A[idx, 0] = torch.cos(angle)
            A[idx, 1] = torch.sin(angle)
            A[idx, 2] = z_bottom  # Bottom layer
            idx += 1
    
    # === TOP CIRCLE (z = 2r/3) ===
    z_top = 2 * height / 3
    
    # Top center: 1 point at origin (but elevated)
    A[idx, :] = torch.tensor([0, 0, z_top])
    idx += 1
    
    # Top circle: 16 islands with 1 point each (aligned with bottom islands)
    for angle in angles:  # Same angles as bottom circle
        A[idx, 0] = torch.cos(angle)
        A[idx, 1] = torch.sin(angle)
        A[idx, 2] = z_top  # Top layer
        idx += 1
    
    # The configuration is now already centered (mean z should be 0)
    # But let's subtract mean anyway to be sure for x and y as well
    A_centered = A - torch.mean(A, dim=0, keepdim=True)
    
    return A_centered


def create_column_matrix(height):
    n_out = 51
    A = torch.zeros(n_out, 3)  # 3D coordinates (x, y, z)
    
    idx = 0
    n_columns = 3
    points_per_column = n_out // n_columns  # 17 points per column
    
    # 3 columns at angles 0°, 120°, 240°
    angles = torch.linspace(0, 2*torch.pi, n_columns + 1)[:-1]
    
    for angle in angles:
        # Position of this column in the x-y plane
        x_pos = torch.cos(angle)
        y_pos = torch.sin(angle)
        
        # Create evenly spaced z-coordinates for this column
        z_coords = torch.linspace(-height/2, height/2, points_per_column)
        
        for z in z_coords:
            A[idx, 0] = x_pos
            A[idx, 1] = y_pos
            A[idx, 2] = z
            idx += 1
    
    # Center the entire configuration
    A_centered = A - torch.mean(A, dim=0, keepdim=True)
    
    return A_centered

def procrustes_calculations(model, step, dataset, config):
  
    model.eval()

    min_distance_circles = 100
    min_height_circles = 100
    min_distance_columns = 100
    min_height_columns = 100
    
    with torch.no_grad():
        # Extract output layer weights - need to find the right layer
        output_weights = None
        
        # Try different common patterns for the output layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check if this is likely the output layer
                if module.weight.shape[0] == dataset.n_out:
                    output_weights = module.weight.detach().cpu().numpy()
                    break
        
        # If we found weights, create visualization
        if output_weights is not None:
            n_out = dataset.n_out

        pca_3d = PCA(n_components=3, random_state=42)
        W = pca_3d.fit_transform(output_weights)
        W = torch.from_numpy(pca_3d.fit_transform(output_weights))

        # First, Procrustes with two circles
        for height in torch.linspace(0.1, 2, 40):
            A = create_circle_matrix(height)

            M = W @ A.T
            U, S, Vt = torch.linalg.svd(M)

            #Optimal orthogonal matrix:
            Q = U @ Vt

            pro_distance = torch.linalg.norm(Q @ A - W, ord = 'fro')
            if pro_distance < min_distance_circles:
                min_distance_circles = pro_distance
                min_height_circles = height

        for height in torch.linspace(0.1, 2, 40):
            A = create_column_matrix(height)
  
            M = W @ A.T
            U, S, Vt = torch.linalg.svd(M)
  
            #Optimal orthogonal matrix:
            Q = U @ Vt
  
            pro_distance = torch.linalg.norm(Q @ A - W, ord = 'fro')
            if pro_distance < min_distance_columns:
                min_distance_columns = pro_distance
                min_height_columns = height

        print("="*70)
        print(f"Minimum procrustes distance for circles: {min_distance_circles}, achieved at height {min_height_circles}")
        print(f"Minimum procrustes distance for columns: {min_distance_columns}, achieved at height {min_height_columns}")



def create_pca_visualization(model, step, dataset, config):
    """
    Create PCA visualization of output layer weights 
    """
    model.eval()
    
    with torch.no_grad():
        # Extract output layer weights - need to find the right layer
        output_weights = None
        
        # Try different common patterns for the output layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check if this is likely the output layer
                if module.weight.shape[0] == dataset.n_out:
                    output_weights = module.weight.detach().cpu().numpy()
                    break
        
        # If we found weights, create visualization
        if output_weights is not None:
            n_out = dataset.n_out
            
            # Create modulo class labels for coloring
            modulo_labels = np.arange(n_out)
            
            try:      
                # ============ 3D PCA with PLOTLY (PC1-3) ============
                pca_3d = PCA(n_components=min(6, output_weights.shape[1]), random_state=42)
                projection_full = pca_3d.fit_transform(output_weights)
                projection_3d_primary = projection_full[:, :3]  # PC1-3
                
                # Create interactive 3D plot with Plotly (PRIMARY)
                fig = go.Figure(data=[go.Scatter3d(
                    x=projection_3d_primary[:, 0],
                    y=projection_3d_primary[:, 1],
                    z=projection_3d_primary[:, 2],
                    mode='markers+text',
                    marker=dict(
                        size=8,
                        color=modulo_labels,
                        colorscale='HSV',
                        showscale=True,
                        colorbar=dict(
                            title=dict(
                                text=f"Output Class<br>(0 to {n_out-1})",
                                side='right'
                            ),
                            tickmode='linear',
                            tick0=0,
                            dtick=max(1, n_out // 10)
                        ),
                        line=dict(color='black', width=0.5),
                        opacity=0.9
                    ),
                    text=[str(i) for i in range(n_out)],
                    textposition='top center',
                    textfont=dict(size=10, color='black', family='Arial Black'),
                    hovertemplate='<b>Class:</b> %{text}<br>' +
                                  '<b>PC1:</b> %{x:.3f}<br>' +
                                  '<b>PC2:</b> %{y:.3f}<br>' +
                                  '<b>PC3:</b> %{z:.3f}<br>' +
                                  '<extra></extra>'
                )])
                
                # Update layout for better visualization
                fig.update_layout(
                    title=dict(
                        text=f'Interactive PCA 3D (PC1-3) of Output Layer Weights (Step {step})<br>' +
                             f'<sub>PC1: {pca_3d.explained_variance_ratio_[0]:.3f}, ' +
                             f'PC2: {pca_3d.explained_variance_ratio_[1]:.3f}, ' +
                             f'PC3: {pca_3d.explained_variance_ratio_[2]:.3f} ' +
                             f'(Total: {pca_3d.explained_variance_ratio_[:3].sum():.3f} variance explained)</sub>',
                        x=0.5,
                        xanchor='center',
                        font=dict(size=16, color='black')
                    ),
                    scene=dict(
                        xaxis=dict(
                            title=f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%} variance)',
                            backgroundcolor="rgb(230, 230, 230)",
                            gridcolor="white",
                            showbackground=True
                        ),
                        yaxis=dict(
                            title=f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%} variance)',
                            backgroundcolor="rgb(230, 230, 230)",
                            gridcolor="white",
                            showbackground=True
                        ),
                        zaxis=dict(
                            title=f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%} variance)',
                            backgroundcolor="rgb(230, 230, 230)",
                            gridcolor="white",
                            showbackground=True
                        ),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.3)
                        )
                    ),
                    width=1000,
                    height=800,
                    showlegend=False
                )
                
                # Save interactive HTML (PRIMARY)
                html_filename = f"pca_3d_interactive_pc1-3_step_{step}.html"
                fig.write_html(html_filename)
                
                # Log to wandb if enabled
                if config['wandb']['use_wandb']:
                    wandb.log({
                        "pca_3d_interactive_pc1-3": wandb.Html(html_filename),
                        "pca_3d_step": step,
                        "pca_3d_pc1_variance_ratio": pca_3d.explained_variance_ratio_[0],
                        "pca_3d_pc2_variance_ratio": pca_3d.explained_variance_ratio_[1],
                        "pca_3d_pc3_variance_ratio": pca_3d.explained_variance_ratio_[2],
                        "pca_3d_total_variance_explained": pca_3d.explained_variance_ratio_[:3].sum()
                    }, step=step)
                
                
            except Exception as e:
                print(f"Error creating PCA at step {step}: {e}")
                import traceback
                traceback.print_exc()
                
    model.train()


def train(config):
    print('Using config:', config)
    train_cfg = config['train']
    wandb_cfg = config['wandb']
    if wandb_cfg['use_wandb']:
        wandb.init(project=wandb_cfg['wandb_project'], config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_item(config['dataset'])
    train_data = GroupDataset(dataset, 'train')
    val_data = GroupDataset(dataset, 'val')
    model = load_item(config['model'], dataset.n_vocab, dataset.n_out, device)
    model.train()
    train_dataloader = DataLoader(train_data, num_workers=train_cfg['num_workers'], 
                                  batch_size=train_cfg['bsize'])
    val_dataloader = DataLoader(val_data, num_workers=train_cfg['num_workers'], 
                                batch_size=train_cfg['bsize'])
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], 
                              weight_decay=train_cfg['weight_decay'], 
                              betas=train_cfg['betas'])
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optim, 
                                                     lr_lambda=lambda s: min(s / train_cfg['warmup_steps'], 1))
    step = 0

    n_out = dataset.n_out

    # Training loop
    for x, y in train_dataloader:
        loss, logs = model.get_loss(x.to(device), y.to(device))
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schedule.step()
        
        # Evaluation every eval_every steps (but not printing)
        if (step+1) % train_cfg['eval_every'] == 0:
            model.eval()
            with torch.no_grad():
                all_val_logs = []
                for i, (val_x, val_y) in enumerate(val_dataloader):
                    if i >= train_cfg['eval_batches']:
                        break
                    _, val_logs = model.get_loss(val_x.to(device), val_y.to(device))
                    all_val_logs.append(val_logs)
            
            out_log = {'val': combine_logs(all_val_logs), 'train': combine_logs([logs]), 
                      'step': (step+1), 'lr': float(lr_schedule.get_last_lr()[0])}
            
            if wandb_cfg['use_wandb']:
                wandb.log(out_log)
            
            model.train()

        # Print summary and evaluate every 10000 steps
        if (step+1) % 500 == 0:
            # Compute validation and training accuracy
            model.eval()
            with torch.no_grad():
                # Validation metrics
                all_val_logs = []
                for i, (val_x, val_y) in enumerate(val_dataloader):
                    if i >= train_cfg['eval_batches']:
                        break
                    _, val_logs = model.get_loss(val_x.to(device), val_y.to(device))
                    all_val_logs.append(val_logs)
                val_combined = combine_logs(all_val_logs)
                
                # Training metrics (use multiple batches for better estimate)
                all_train_logs = []
                train_dataloader_iter = iter(train_dataloader)
                for i in range(train_cfg['eval_batches']):
                    try:
                        train_x, train_y = next(train_dataloader_iter)
                        _, train_logs = model.get_loss(train_x.to(device), train_y.to(device))
                        all_train_logs.append(train_logs)
                    except StopIteration:
                        break
                train_combined = combine_logs(all_train_logs)
            
            print(f"\n{'='*70}")
            print(f"Step: {step+1:,}")
            print(f"{'='*70}")
            print(f"Training Accuracy:   {train_combined.get('acc', train_combined.get('accuracy', 0.0)):.4f}")
            print(f"Validation Accuracy: {val_combined.get('acc', val_combined.get('accuracy', 0.0)):.4f}")
            
            # Create visualizations
            create_pca_visualization(model, step+1, dataset, config)

            # Procrustes calculations
            procrustes_calculations(model, step+1, dataset, config)
            
            # Log variance thresholds to wandb
            
            model.train()

        step += 1
        if train_cfg['max_steps'] is not None and step >= train_cfg['max_steps']:
            break

    print(f"\nTraining completed at step {step}")
    if wandb_cfg['use_wandb']:
        wandb.finish()


@hydra.main(config_path="../config", config_name="train_grokk")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)

if __name__ == "__main__":
    main()
