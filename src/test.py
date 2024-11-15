import torch
import torch.nn.functional as F
from .utils.metrics import calculate_fid
from torchvision.utils import save_image
import os

def test_model(model, test_loader, device):
    """Evaluate the VQ-VAE model."""
    model.eval()
    test_loss = 0
    test_n_samples = 0
    
    # Create directory for test reconstructions
    os.makedirs('test_results', exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            
            # Forward pass
            recon_batch, commit_loss, _ = model(data)
            
            # Calculate loss
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')
            test_loss += recon_loss.item()
            test_n_samples += data.size(0)
            
            # Save first batch reconstructions
            if batch_idx == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                     recon_batch[:n]])
                save_image(comparison.cpu(),
                         'test_results/reconstruction.png', nrow=n)
    
    test_loss /= test_n_samples
    fid_score = calculate_fid(model, test_loader, device)
    
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Test set FID score: {:.2f}'.format(fid_score))
    
    return test_loss, fid_score