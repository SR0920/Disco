import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import morphology, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import convolve

from ..backbones import vmnet
from ..builder import SEGMENTORS
from ..heads import DiscoHead
from ..losses import BatchMultiClassDiceLoss
from .base import BaseSegmentor


class AttrDict(dict):
  
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


@SEGMENTORS.register_module()
class BipartiteDisco(BaseSegmentor):
    
    def __init__(self, num_classes=4, train_cfg=None, test_cfg=None):
        super(BipartiteDisco, self).__init__()
        self.num_classes = num_classes
        self.train_cfg = train_cfg if train_cfg else {}
        
    
        default_test_cfg = {
            'mode': 'split',
            'crop_size': (256, 256),
            'overlap_size': (40, 40),
            'radius': 1,
            'rotate_degrees': [0, 90],
            'flip_directions': ['none', 'horizontal', 'vertical', 'diagonal'],
        }
        
        if test_cfg:
           
            merged_cfg = default_test_cfg.copy()
            merged_cfg.update(test_cfg)
            self.test_cfg = AttrDict(merged_cfg)
        else:
            self.test_cfg = AttrDict(default_test_cfg)
        
        
        self.backbone = vmnet(
            in_channels=3, 
            pretrained=True,
            out_indices=[0, 1, 2, 3, 4, 5]
        )
        self.head = DiscoHead(
            num_classes=self.num_classes,
            bottom_in_dim=512,
            skip_in_dims=(64, 128, 256, 512, 512),
            stage_dims=[16, 32, 64, 128, 256],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN')
        )
        
      
        self.BACKGROUND = 0
        self.BIPARTITE_COLOR_1 = 1
        self.BIPARTITE_COLOR_2 = 2
        self.CONFLICT_COLOR = 3
        
      
        default_weights = {
            'sem_ce_loss': 5.0,
            'sem_dice_loss': 0.5,
            'bipartite_ce_loss': 10.0,
            'bipartite_dice_loss': 1.0,
            'bipartite_consistency_loss': 3.0,
            'conflict_resolution_loss': 8.0,
            'adjacency_constraint_loss': 2.0,
        }
        
        self.loss_weights = self.train_cfg.get('bipartite_loss_weights', default_weights)
    
    def calculate(self, img, encoding=None):
       
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        
       
        sem_pred, bipartite_pred = self.head(bottom_feat, skip_feats, encoding)
        
        return sem_pred, bipartite_pred
    
    def forward(self, data, label=None, metas=None, **kwargs):
    
        if self.training:
            return self._forward_train(data, label)
        else:
            return self._forward_test(data, metas)
    
    def _generate_adjacency_from_inst_gt(self, inst_gt):
    
        batch_size = inst_gt.shape[0]
        adjacency_info = []
        
        for b in range(batch_size):
            instance_mask = inst_gt[b, 0].cpu().numpy()
            
            unique_ids = np.unique(instance_mask)
            adjacency_dict = {int(inst_id): [] for inst_id in unique_ids if inst_id != 0}
            
            
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            
            for inst_id in unique_ids:
                if inst_id == 0:
                    continue
                    
            
                inst_region = (instance_mask == inst_id).astype(np.int32)
                boundary = convolve(inst_region, kernel, mode='constant', cval=0)
                
            
                neighbors = np.unique(instance_mask[boundary > 0])
                neighbors = neighbors[neighbors != 0]  
                neighbors = neighbors[neighbors != inst_id] 
                
                adjacency_dict[int(inst_id)] = neighbors.astype(int).tolist()
            
            adjacency_info.append(adjacency_dict)
        
        return adjacency_info
    
    def _forward_train(self, data, label):
   
        sem_pred, bipartite_pred = self.calculate(data['img'])

        sem_gt_inner = label['sem_gt_inner']
        inst_gt = label['inst_gt']
        weight_map = label.get('loss_weight_map', None)
        
    
        # print(f"Debug - sem_gt_inner shape: {sem_gt_inner.shape}")
        # print(f"Debug - inst_gt shape: {inst_gt.shape}")
        
        if len(sem_gt_inner.shape) == 4:
            sem_gt_inner = sem_gt_inner.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        
        binary_sem_gt = (sem_gt_inner > 0).long()
        
        # print(f"Debug - binary_sem_gt shape: {binary_sem_gt.shape}")
        # print(f"Debug - sem_pred shape: {sem_pred.shape}")
        # print(f"Debug - bipartite_pred shape: {bipartite_pred.shape}")
        
        adjacency_info = self._generate_adjacency_from_inst_gt(inst_gt)
        
        loss_dict = {}
        
        sem_loss = self._semantic_loss(sem_pred, binary_sem_gt, weight_map)
        loss_dict.update(sem_loss)
        
        bipartite_loss = self._bipartite_coloring_loss(
            bipartite_pred, sem_gt_inner, weight_map
        )
        loss_dict.update(bipartite_loss)
        
  
        consistency_loss = self._bipartite_consistency_loss(
            bipartite_pred, sem_gt_inner, inst_gt
        )
        loss_dict.update(consistency_loss)
        
        conflict_loss = self._conflict_resolution_loss(
            bipartite_pred, sem_gt_inner, inst_gt
        )
        loss_dict.update(conflict_loss)
     
        adjacency_loss = self._adjacency_constraint_loss(
            bipartite_pred, inst_gt, adjacency_info
        )
        loss_dict.update(adjacency_loss)
        
        training_metrics = self._training_metrics(sem_pred, binary_sem_gt)
        loss_dict.update(training_metrics)
        
        return loss_dict
        
    def _semantic_loss(self, sem_logit, sem_gt, weight_map=None):
        loss_dict = {}
        
        if len(sem_gt.shape) == 4:
            sem_gt = sem_gt.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        
        ce_loss = F.cross_entropy(sem_logit, sem_gt, reduction='none')
        if weight_map is not None:
            ce_loss = ce_loss * weight_map
        ce_loss = ce_loss.mean()
        
        dice_calculator = BatchMultiClassDiceLoss(num_classes=2)
        dice_loss = dice_calculator(sem_logit, sem_gt)
        
        loss_dict['sem_ce_loss'] = self.loss_weights['sem_ce_loss'] * ce_loss
        loss_dict['sem_dice_loss'] = self.loss_weights['sem_dice_loss'] * dice_loss
        
        return loss_dict
    
    def _bipartite_coloring_loss(self, bipartite_logit, bipartite_gt, weight_map=None):
    
        loss_dict = {}
        
        if len(bipartite_gt.shape) == 4:
            bipartite_gt = bipartite_gt.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        
        ce_loss = F.cross_entropy(bipartite_logit, bipartite_gt, reduction='none')
        if weight_map is not None:
            ce_loss = ce_loss * weight_map
        ce_loss = ce_loss.mean()
        
        dice_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        dice_loss = dice_calculator(bipartite_logit, bipartite_gt)
        
        loss_dict['bipartite_ce_loss'] = self.loss_weights['bipartite_ce_loss'] * ce_loss
        loss_dict['bipartite_dice_loss'] = self.loss_weights['bipartite_dice_loss'] * dice_loss
        
        return loss_dict
    
    def _bipartite_consistency_loss(self, bipartite_logit, bipartite_gt, inst_gt):

        batch_size = bipartite_logit.shape[0]
        consistency_loss = 0.0
        
        bipartite_prob = F.softmax(bipartite_logit, dim=1)
        
        for b in range(batch_size):
           
            bipartite_mask = (bipartite_gt[b] == self.BIPARTITE_COLOR_1) | \
                           (bipartite_gt[b] == self.BIPARTITE_COLOR_2)
            
            if not bipartite_mask.any():
                continue
            
            
            conflict_prob = bipartite_prob[b, self.CONFLICT_COLOR][bipartite_mask]
            consistency_loss += torch.mean(conflict_prob ** 2)
        
        loss_dict = {
            'bipartite_consistency_loss': 
            self.loss_weights['bipartite_consistency_loss'] * (consistency_loss / batch_size)
        }
        
        return loss_dict
    
    def _conflict_resolution_loss(self, bipartite_logit, bipartite_gt, inst_gt):
       
        batch_size = bipartite_logit.shape[0]
        resolution_loss = 0.0
        
        bipartite_prob = F.softmax(bipartite_logit, dim=1)
        
        for b in range(batch_size):
            
            conflict_mask = (bipartite_gt[b] == self.CONFLICT_COLOR)
            
            if not conflict_mask.any():
                continue
            
            conflict_prob = bipartite_prob[b, self.CONFLICT_COLOR][conflict_mask]
            resolution_loss += torch.mean((1 - conflict_prob) ** 2)
        
        loss_dict = {
            'conflict_resolution_loss': 
            self.loss_weights['conflict_resolution_loss'] * (resolution_loss / batch_size)
        }
        
        return loss_dict
    
    def _adjacency_constraint_loss(self, bipartite_logit, inst_gt, adjacency_info):
       
        batch_size = bipartite_logit.shape[0]
        constraint_loss = torch.tensor(0.0, device=bipartite_logit.device, dtype=torch.float32) 
        
        bipartite_pred = torch.argmax(bipartite_logit, dim=1)
        
        for b in range(batch_size):
            if b >= len(adjacency_info):
                continue
                
            adj_dict = adjacency_info[b]
            instance_mask = inst_gt[b, 0]
            
            violation_count = 0
            total_pairs = 0
            
            for inst_id, neighbors in adj_dict.items():
                inst_region = (instance_mask == inst_id)
                if not inst_region.any():
                    continue
                
               
                inst_colors = bipartite_pred[b][inst_region]
                if len(inst_colors) == 0:
                    continue
                
              
                try:
                    inst_color = torch.mode(inst_colors)[0].item()
                except:
                    
                    unique_colors, counts = torch.unique(inst_colors, return_counts=True)
                    inst_color = unique_colors[torch.argmax(counts)].item()
                
                for neighbor_id in neighbors:
                    neighbor_region = (instance_mask == neighbor_id)
                    if not neighbor_region.any():
                        continue
                    
                   
                    neighbor_colors = bipartite_pred[b][neighbor_region]
                    if len(neighbor_colors) == 0:
                        continue
                    
                    try:
                        neighbor_color = torch.mode(neighbor_colors)[0].item()
                    except:
                        unique_colors, counts = torch.unique(neighbor_colors, return_counts=True)
                        neighbor_color = unique_colors[torch.argmax(counts)].item()
                    
                    total_pairs += 1
                    
                   
                    if inst_color == neighbor_color and inst_color != self.CONFLICT_COLOR:
                        violation_count += 1
            
            if total_pairs > 0:
               
                batch_constraint_loss = torch.tensor(
                    violation_count / total_pairs, 
                    device=bipartite_logit.device, 
                    dtype=torch.float32
                )
                constraint_loss += batch_constraint_loss
        
   
        final_loss = self.loss_weights['adjacency_constraint_loss'] * (constraint_loss / batch_size)
        
        loss_dict = {
            'adjacency_constraint_loss': final_loss
        }
        
        return loss_dict
    
    def _training_metrics(self, sem_pred, sem_gt):
      
        pred_labels = torch.argmax(sem_pred, dim=1)
        correct = (pred_labels == sem_gt).float()
        accuracy = correct.mean()
        
        return {'training_acc': accuracy}
    
    def _forward_test(self, data, metas):

        sem_logit, bipartite_logit = self.inference(data['img'], metas[0], True)
        
     
        sem_pred = sem_logit.argmax(dim=1)
        
     
        bipartite_pred = bipartite_logit.argmax(dim=1)
        
     
        bipartite_pred[sem_pred != 1] = self.BACKGROUND
        
   
        final_sem, final_inst = self._bipartite_postprocess(
            bipartite_pred.cpu().numpy()[0]
        )
        
        return [{'sem_pred': final_sem, 'inst_pred': final_inst}]
    
    def _bipartite_postprocess(self, bipartite_pred):
    
        sem_pred = np.zeros_like(bipartite_pred, dtype=np.uint8)
        inst_pred = np.zeros_like(bipartite_pred, dtype=np.int32)
        
        current_inst_id = 0
        
     
        for color_id in [self.BIPARTITE_COLOR_1, self.BIPARTITE_COLOR_2, self.CONFLICT_COLOR]:
            color_mask = (bipartite_pred == color_id)
            
            if not color_mask.any():
                continue
            
      
            labeled_mask = measure.label(color_mask)
            
            for region_id in range(1, labeled_mask.max() + 1):
                region_mask = (labeled_mask == region_id)
                
               
                region_mask = binary_fill_holes(region_mask)
                region_mask = remove_small_objects(region_mask, 5)
                
                if region_mask.any():
               
                    radius = self.test_cfg.get('radius', 1)
                    
                    region_mask = morphology.dilation(
                        region_mask, 
                        morphology.disk(radius)
                    )
                    
                    current_inst_id += 1
                    inst_pred[region_mask] = current_inst_id
                    sem_pred[region_mask] = color_id
        
        return sem_pred, inst_pred
    
    def get_dynamic_statistics(self, data_batch):

        with torch.no_grad():
            sem_pred, bipartite_pred = self.calculate(data_batch['img'])
            
      
            stats = {
                'batch_size': sem_pred.shape[0],
                'bipartite_color_distribution': {},
                'prediction_confidence': {},
            }
            
            bipartite_prob = F.softmax(bipartite_pred, dim=1)
            bipartite_pred_labels = torch.argmax(bipartite_pred, dim=1)
            
      
            for color_id in range(self.num_classes):
                color_pixels = (bipartite_pred_labels == color_id).sum().item()
                stats['bipartite_color_distribution'][f'color_{color_id}'] = color_pixels
            
       
            for color_id in range(self.num_classes):
                color_mask = (bipartite_pred_labels == color_id)
                if color_mask.any():
                    color_confidence = bipartite_prob[:, color_id][color_mask].mean().item()
                    stats['prediction_confidence'][f'color_{color_id}'] = color_confidence
            
            return stats
    
    def visualize_bipartite_coloring(self, image, predictions, save_path=None):

        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
     
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
   
        sem_pred = predictions['sem_pred']
        axes[1].imshow(sem_pred, cmap='gray')
        axes[1].set_title('Semantic Segmentation')
        axes[1].axis('off')
        
     
        bipartite_pred = predictions.get('bipartite_pred', sem_pred)
        color_map = plt.cm.get_cmap('Set1', self.num_classes)
        axes[2].imshow(bipartite_pred, cmap=color_map, vmin=0, vmax=self.num_classes-1)
        axes[2].set_title('Bipartite Coloring')
        axes[2].axis('off')
       
        inst_pred = predictions['inst_pred']
        axes[3].imshow(inst_pred, cmap='nipy_spectral')
        axes[3].set_title('Instance Segmentation')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
