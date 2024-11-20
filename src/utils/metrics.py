import torch
import torch.nn as nn
import torchvision.models as models
from scipy import linalg
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
import torch.nn.functional as F
# class InceptionV3(nn.Module):
#     def __init__(self):
#         super().__init__()
#         weights = models.Inception_V3_Weights.DEFAULT
#         self.inception = models.inception_v3(weights=weights)
#         # Remove last layer (删除最后一层)
#         self.inception.fc = nn.Identity()
#         # Remove aux layer (删除辅助层)
#         self.inception.aux_logits = False
        
#         # Freeze parameters (冻结参数)
#         for param in self.inception.parameters():
#             param.requires_grad = False
            
#     def forward(self, x):
#         # Check if input has correct dimensions
#         if x.dim() == 3:  # If not batched
#             x = x.unsqueeze(0)
        
#         # Resize if needed (如果需要调整大小)
#         if x.shape[-1] != 299:
#             x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
#         # Ensure we have correct number of channels (确保通道数正确)
#         if x.shape[1] != 3:
#             x = x.repeat(1, 3, 1, 1)
            
#         # Get features (获取特征)
#         x = self.inception(x)
#         return x

# def calculate_activation_statistics(images, model, batch_size=128, dims=2048, device='cuda'):
#     model.eval()
#     act = np.empty((len(images), dims))
    
#     with torch.no_grad():
#         for i in range(0, len(images), batch_size):
#             # Get batch of images (获取一批图片)
#             batch = torch.stack(images[i:i + batch_size]).to(device)
#             batch = batch.type(torch.FloatTensor).to(device)
            
#             # Resize for inception (调整大小为inception所需)
#             if batch.size(-1) != 299:
#                 batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
#             # Get features (获取特征)
#             pred = model(batch)  # [batch_size, 2048]
            
#             # No need for pooling now since inception already gives us the features
#             # 不需要池化因为inception已经给我们特征了
#             act[i:i + batch.shape[0]] = pred.cpu().numpy()
    
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma

# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)
    
#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)
    
#     assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
#     assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    
#     diff = mu1 - mu2
    
#     # Product might be almost singular
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
#         print(msg)
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
#     # Numerical error might give slight imaginary component
#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             raise ValueError(f"Imaginary component {m}")
#         covmean = covmean.real
    
#     tr_covmean = np.trace(covmean)
    
#     return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

# def calculate_fid(model, dataloader, device, batch_size=50):
#     inception = InceptionV3().to(device)
#     real_images = []
#     generated_images = []
    
#     with torch.no_grad():
#         for batch in dataloader:
#             images = batch[0].to(device)
#             recon_images, _, _ = model(images)
            
#             real_images.extend(images.cpu())
#             generated_images.extend(recon_images.cpu())
            
#             if len(real_images) >= batch_size:
#                 break
    
#     mu_real, sigma_real = calculate_activation_statistics(real_images, inception, device=device)
#     mu_fake, sigma_fake = calculate_activation_statistics(generated_images, inception, device=device)
    
#     fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
#     return fid_value



#construct the feature extracter
class Inception(nn.Module):
    def __init__(self,device='cuda'):
        super().__init__()
        self.device = device
        #define the inception
        weights=models.Inception_V3_Weights.DEFAULT
        self.inception=models.inception_v3(weights=weights)
        #delete the layers in the inception we do not need
        self.inception.fc=nn.Identity()
        self.inception.aux_logits=False
        #freeze the gradient
        for param in self.inception.parameters():
            param.requires_grad=False
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # 确保在GPU上计算
        return self.inception(x.to(self.device))

#calculate the FID
class FIDcalculator():
    def __init__(self,device='cuda'):
        self.inception = Inception(device=device).to(device)
        self.device=device
    
    def calculate_activation_statistics(self, images, batch_size=128):
        self.inception.eval()
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                # 图片已经在GPU上，直接stack
                batch = torch.stack(images[i:i + batch_size])
                batch_features = self.inception(batch)
                features_list.append(batch_features)
                
            # 在GPU上合并和计算统计量
            features = torch.cat(features_list, dim=0)
            mu = features.mean(0)
            sigma = torch.cov(features.T)
            
            return mu, sigma  # 直接返回tensor而不是numpy数组
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        # 已经是tensor，不需要从numpy转换
        diff = mu1 - mu2
        diff_squared = torch.dot(diff, diff)
        
        # 使用svd计算矩阵平方根
        try:
            # sigma1和sigma2应该是对称矩阵
            prod = torch.mm(sigma1, sigma2)
            u, s, vh = torch.linalg.svd(prod)
            covmean = torch.mm(torch.mm(u, torch.diag(torch.sqrt(s))), vh)
        except:
            # 如果SVD失败，添加一个小的对角矩阵
            print(f"SVD失败，添加 {eps} 到对角线")
            offset = torch.eye(sigma1.shape[0], device=self.device) * eps
            prod = torch.mm(sigma1 + offset, sigma2 + offset)
            u, s, vh = torch.linalg.svd(prod)
            covmean = torch.mm(torch.mm(u, torch.diag(torch.sqrt(s))), vh)
        
        tr_covmean = torch.trace(covmean)
        
        return (diff_squared + torch.trace(sigma1) + 
                torch.trace(sigma2) - 2 * tr_covmean).item()
            
    def calculate_fid(self, real_images, fake_images, n_samples=5):
        """
        计算随机n张图片的FID分数
        
        Args:
            real_images: 真实图片列表
            fake_images: 生成图片列表
            n_samples: 要计算的图片数量（默认5张）
        """
        # 确保有足够的图片
        assert len(real_images) >= n_samples, f"真实图片数量不足{n_samples}张"
        assert len(fake_images) >= n_samples, f"生成图片数量不足{n_samples}张"
        
        # 随机选择n张图片
        indices = torch.randperm(len(real_images))[:n_samples]
        real_samples = [real_images[i] for i in indices]
        fake_samples = [fake_images[i] for i in indices]
        
        # 计算统计量
        mu_real, sigma_real = self.calculate_activation_statistics(real_samples)
        mu_fake, sigma_fake = self.calculate_activation_statistics(fake_samples)
        
        # 计算FID距离
        fid_value = self.calculate_frechet_distance(
            mu_real, sigma_real,
            mu_fake, sigma_fake
        )
        
        return fid_value
        