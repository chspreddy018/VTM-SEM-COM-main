import torch.nn as nn
from .reshape import forward_6d_as_4d, from_6d_to_3d, from_3d_to_6d
from .attention import CrossAttention
import torch.nn.functional as F       
        
class VTMImageBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, t_idx=None, **kwargs):
        return forward_6d_as_4d(self.backbone, x, t_idx=t_idx, get_features=True, **kwargs)

    def bias_parameters(self):
        # bias parameters for similarity adaptation
        for p in self.backbone.bias_parameters():
            yield p

    def bias_parameter_names(self):
        return [f'backbone.{name}' for name in self.backbone.bias_parameter_names()]


class VTMLabelBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        
    def encode(self, x, t_idx=None, **kwargs):
        return forward_6d_as_4d(self.backbone.encode, x, t_idx=t_idx, **kwargs)
    
    def decode(self, x, t_idx=None, **kwargs):
        return forward_6d_as_4d(self.backbone.decode, x, t_idx=t_idx, **kwargs)
    
    def forward(self, x, t_idx=None, encode_only=False, decode_only=False, **kwargs):
        assert not (encode_only and decode_only)
        if not decode_only:
            x = self.encode(x, t_idx=t_idx, **kwargs)
        if not encode_only:
            x = self.decode(x, t_idx=t_idx, **kwargs)
        
        return x
        

class VTMMatchingModule(nn.Module):
    def __init__(self, dim_w, dim_z, config):
        super().__init__()
        self.matching = nn.ModuleList([CrossAttention(dim_w, dim_z, dim_z, num_heads=config.n_attn_heads)
                                       for i in range(config.n_levels)])
        self.n_levels = config.n_levels
            
    def forward(self, W_Qs, W_Ss, Z_Ss, attn_mask=None):
        B, T, N, _, H, W = W_Ss[-1].size()
        
        assert len(W_Qs) == self.n_levels
        
        if attn_mask is not None:
            attn_mask = from_6d_to_3d(attn_mask)
            
        Z_Qs = []
        for level in range(self.n_levels):
            Q = from_6d_to_3d(W_Qs[level])
            K = from_6d_to_3d(W_Ss[level])
            V = from_6d_to_3d(Z_Ss[level])
            
            O = self.matching[level](Q, K, V, N=N, H=H, mask=attn_mask)
            Z_Q = from_3d_to_6d(O, B=B, T=T, H=H, W=W)
            Z_Qs.append(Z_Q)
        
        return Z_Qs


import torch.nn as nn
from .reshape import forward_6d_as_4d, from_6d_to_3d, from_3d_to_6d
from .attention import CrossAttention
import torch.nn.functional as F       

# ... (Previous code for VTMImageBackbone, VTMLabelBackbone, VTMMatchingModule) ...

class VTM(nn.Module):
    def __init__(self, image_backbone, label_backbone, matching_module, config):
        super().__init__()
        self.image_backbone = image_backbone
        self.label_backbone = label_backbone
        self.matching_module = matching_module
        self.config = config  # Pass config to the class

    def compute_loss(self, data):
        # Implement your loss function here based on the provided inputs and model
        X_S, Y_S, M_S, t_idx = data
        Y_Q_pred = self(X_S, Y_S, X_S, t_idx=t_idx, sigmoid=False)
        loss = F.binary_cross_entropy(Y_Q_pred, Y_S, weight=M_S)
        return loss

    def calculate_miou(self, Y_true, Y_pred, M):
        # Implement the calculation of Mean Intersection over Union (mIoU) here
        pass

    def compute_metric(self, Y_true, Y_pred, M, task):
        # Implement your metric computation here based on the provided inputs
        # For example, you can calculate the Mean Intersection over Union (mIoU) for semantic segmentation
        miou = self.calculate_miou(Y_true, Y_pred, M)
        return miou

    def forward(self, X_S, Y_S, X_Q, t_idx=None, sigmoid=True):
        # encode support input, query input, and support output
        W_Ss = self.image_backbone(X_S, t_idx)
        W_Qs = self.image_backbone(X_Q, t_idx)
        Z_Ss = self.label_backbone.encode(Y_S, t_idx)
        
        # mix support output by matching
        Z_Q_preds = self.matching_module(W_Qs, W_Ss, Z_Ss)
        
        # decode support output
        Y_Q_pred = self.label_backbone.decode(Z_Q_preds, t_idx)
        
        if sigmoid:
            Y_Q_pred = Y_Q_pred.sigmoid()
        
        return Y_Q_pred

    def train_dataloader(self):
        # Define your training dataloader here. It should return a DataLoader object.
        # Example: DataLoader(dataset, batch_size=self.config.global_batch_size, shuffle=True, num_workers=self.config.num_workers)
        pass

    def configure_optimizers(self):
        # Define your optimizer and learning rate scheduler here.
        # Example: optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        #          scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        #          return [optimizer], [scheduler]
        pass

    def training_step(self, batch, batch_idx):
        X_S, Y_S, M_S, t_idx = batch
        Y_Q_pred = self(X_S, Y_S, X_S, t_idx=t_idx, sigmoid=False)
        loss = self.compute_loss((X_S, Y_S, M_S, t_idx))  # Call the compute_loss method
        metric = self.compute_metric(Y_S, Y_Q_pred, M_S, self.task)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_metric', metric, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ... Implement other necessary methods ...

    def bias_parameters(self):
        # bias parameters for similarity adaptation
        for p in self.image_backbone.bias_parameters():
            yield p

    def bias_parameter_names(self):
        return [f'image_backbone.{name}' for name in self.image_backbone.bias_parameter_names()]

    def pretrained_parameters(self):
        return self.image_backbone.parameters()

    def scratch_parameters(self):
        modules = [self.label_backbone, self.matching_module]
        for module in modules:
            for p in module.parameters():
                yield p

