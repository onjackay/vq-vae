import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from datetime import datetime
from .utils.metrics import calculate_fid

def train_model(model, train_loader, optimizer, device, args):
    """Train the VQ-VAE model."""
    model.train()
    # Create checkpoint directory
    checkpoint_dir = os.path.join('checkpoints', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_fid = float('inf')
    
    for epoch in range(args.epochs):
        total_recon_loss = 0
        total_commit_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}') as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, commit_loss, _ = model(data)
                
                # Calculate losses
                recon_loss = F.mse_loss(recon_batch, data)
                loss = recon_loss + commit_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                total_recon_loss += recon_loss.item()
                total_commit_loss += commit_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'recon_loss': total_recon_loss / (batch_idx + 1),
                    'commit_loss': total_commit_loss / (batch_idx + 1)
                })
        
        # Calculate FID score and save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            fid_score = calculate_fid(model, train_loader, device)
            print(f'FID Score: {fid_score:.2f}')
            
            # Save checkpoint if best FID
            if fid_score < best_fid:
                best_fid = fid_score
                checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fid_score': fid_score,
                }, checkpoint_path)