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

from collections import Counter
from sklearn.metrics import silhouette_score

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D

import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

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

from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def comprehensive_2d_subspace_clustering_search(data, n_samples=1000, seed=42):
    """
    Search over random 2D subspaces and test for different cluster structures.
    """
    np.random.seed(seed)
    n_features = data.shape[1]
    n_points = data.shape[0]
    
    # Test different numbers of clusters
    k_tests = list(range(2, n_points//3+1))
    
    best_results = {k: {'ch_score': -np.inf, 'projection': None, 'labels': None} 
                    for k in k_tests}
    
    k_preference_counts_ch = {k: 0 for k in k_tests}
    
    print("Searching over 2D subspaces...")
    
    for iteration in range(n_samples):
        if (iteration + 1) % 500 == 0:
            print(f"  Iteration {iteration + 1}/{n_samples}")
        
        # Random 2D projection (orthonormal)
        projection_matrix = np.random.randn(n_features, 2)
        projection_matrix, _ = np.linalg.qr(projection_matrix)
        
        # Project data to 2D
        data_2d = data @ projection_matrix
        
        # Try different cluster numbers
        ch_k_scores = {}
        
        for k in k_tests:
            try:
                # Create predetermined mod-k labels
                labels = np.array([i % k for i in range(n_points)])
                
                # Check if we have valid clustering (multiple unique labels)
                if k > 1 and k < n_points and len(np.unique(labels)) == k:
                    # Compute scores
                    ch_score = calinski_harabasz_score(data_2d, labels)
                    
                    ch_k_scores[k] = ch_score
                    
                    
                    if ch_score > best_results[k]['ch_score']:
                        best_results[k]['ch_score'] = ch_score
                        best_results[k]['projection'] = projection_matrix.copy()
                        best_results[k]['labels'] = labels.copy()
                else:
                    ch_k_scores[k] = -np.inf
                
            except Exception as e:
                print(f"Error at iteration {iteration}, k={k}: {e}")
                ch_k_scores[k] = -np.inf
        
        
        # Which k won for CH this projection?
        if ch_k_scores:  # Make sure dictionary is not empty
            best_k_ch = max(ch_k_scores, key=ch_k_scores.get)
            k_preference_counts_ch[best_k_ch] += 1
    
    # Summary statistics
    print("\n" + "="*70)
    print("2D SUBSPACE SEARCH RESULTS")
    print("="*70)
    
    
    print(f"\nOut of {n_samples} random 2D projections (Calinski-Harabasz):")
    for k in k_tests:
        count = k_preference_counts_ch[k]
        percentage = 100 * count / n_samples
        print(f"  k={k:2d}: {count:5d} times ({percentage:5.2f}%)")
    
    print(f"\nBest results for each k:")
    for k in k_tests:
        best = best_results[k]
        print(f"  k={k}: CH={best['ch_score']:.4f}")
    
    # Create bar graph for CH percentage distribution
    fig_ch = plt.figure(figsize=(10, 6))
    k_values = list(k_preference_counts_ch.keys())
    percentages = [100 * k_preference_counts_ch[k] / n_samples for k in k_values]
    
    plt.bar(k_values, percentages, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Percentage of Projections (%)', fontsize=12)
    plt.title(f'Calinski-Harabasz: Distribution of Preferred k Values\nAcross {n_samples} Random 2D Projections', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(k_values)
    
    # Add percentage labels on top of bars
    for i, (k, pct) in enumerate(zip(k_values, percentages)):
        if pct > 0:
            plt.text(k, pct + 0.5, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    ch_bar_filename = 'clustering_ch_percentages.png'
    plt.savefig(ch_bar_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    return best_results, k_preference_counts_ch, ch_bar_filename


def visualize_best_2d_projections(data, best_results, step, k_tests=[3, 5, 7]):
    """
    Visualize the data in the best 2D projections for different k values
    """
    # Filter out k values that have no valid results
    valid_k_tests = [k for k in k_tests if best_results[k]['projection'] is not None]
    
    if not valid_k_tests:
        print("Warning: No valid projections found for any k value. Skipping visualization.")
        return None
    
    fig, axes = plt.subplots(1, len(valid_k_tests), figsize=(7 * len(valid_k_tests), 6))
    
    if len(valid_k_tests) == 1:
        axes = [axes]
    
    for idx, k in enumerate(valid_k_tests):
        best = best_results[k]
        projection = best['projection']
        labels = best['labels']
        
        # Project data
        data_2d = data @ projection
        
        # Plot
        scatter = axes[idx].scatter(data_2d[:, 0], data_2d[:, 1],
                                   c=labels, cmap='tab20', s=100, alpha=0.7, edgecolors='black')
        
        # Add point labels
        for i in range(len(data_2d)):
            axes[idx].text(data_2d[i, 0], data_2d[i, 1], str(i), 
                         fontsize=8, ha='center', va='center')
        
        # Show both scores in title
        axes[idx].set_title(f'Best 2D projection for k={k}\n' +
                           f'CH: {best["ch_score"]:.3f}')
        axes[idx].set_xlabel('Projection 1')
        axes[idx].set_ylabel('Projection 2')
        axes[idx].grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=axes[idx], label='Cluster')
    
    plt.tight_layout()
    filename = f'2d_subspace_clustering_step_{step}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n2D Visualization saved as '{filename}'")
    plt.close()
    
    return filename


def subspace_clustering_analysis(model, step, dataset, config):
    """
    2D subspace clustering analysis
    """
    model.eval()
    
    with torch.no_grad():
        # Extract output weights
        output_weights = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if module.weight.shape[0] == dataset.n_out:
                    output_weights = module.weight.detach().cpu().numpy()
                    break
        
        if output_weights is not None:
            print(f"\n{'='*70}")
            print(f"2D SUBSPACE CLUSTERING ANALYSIS - Step {step}")
            print(f"{'='*70}")

            n_pca_components = 20
            pca_reduce = PCA(n_components=n_pca_components)
            search_data = pca_reduce.fit_transform(output_weights)
            variance_explained = pca_reduce.explained_variance_ratio_.sum()
            print(f"Reduced to {n_pca_components}D ({variance_explained:.1%} variance)")

            # 2D analysis - now returns 3 values including the bar graph filename
            best_2d, k_counts_ch, ch_bar_filename = comprehensive_2d_subspace_clustering_search(
                search_data, n_samples=2000, seed=42
            )
            
            # Visualize
            viz_filename = visualize_best_2d_projections(search_data, best_2d, step, k_tests=[3, 5, 7])
            
            # Log to wandb if enabled
            if config['wandb']['use_wandb']:
                log_dict = {}
                # Only add visualization if it was created
                if viz_filename is not None:
                    log_dict['2d_visualization'] = wandb.Image(viz_filename)
                
                # Add the bar graph visualization
                if ch_bar_filename is not None:
                    log_dict['clustering_ch_percentages'] = wandb.Image(ch_bar_filename)
                
                wandb.log(log_dict, step=step)
            
            return best_2d, k_counts_ch
    
    model.train()
    return None, None



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
        if (step+1) % 2000 == 0:
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
            
            subspace_clustering_analysis(model, step+1, dataset, config)
            
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
