import torch
from torch import nn
from torch.nn import functional as F


class IDRLoss(nn.Module):
    def __init__(self, idr_rgb_weight, sg_rgb_weight, eikonal_weight, mask_weight, alpha,
                 r_patch=-1, normalsmooth_weight=0., loss_type='L1'):
        super().__init__()
        self.idr_rgb_weight = idr_rgb_weight
        self.sg_rgb_weight = sg_rgb_weight
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.alpha = alpha
        if loss_type == 'L1':
            print('Using L1 loss for comparing images!')
            self.img_loss = nn.L1Loss(reduction='mean')
        elif loss_type == 'L2':
            print('Using L2 loss for comparing images!')
            self.img_loss = nn.MSELoss(reduction='mean')
        else:
            raise Exception('Unknown loss_type!')

        self.r_patch = int(r_patch)
        self.normalsmooth_weight = normalsmooth_weight
        print('Patch size in normal smooth loss: ', self.r_patch)

    def get_rgb_loss(self, idr_rgb_values, sg_rgb_values, rgb_gt, network_object_mask, object_mask):
        mask = network_object_mask & object_mask
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()

        idr_rgb_values = idr_rgb_values[mask].reshape((-1, 3))
        sg_rgb_values = sg_rgb_values[mask].reshape((-1, 3))
        rgb_gt = rgb_gt.reshape(-1, 3)[mask].reshape((-1, 3))

        idr_rgb_loss = self.img_loss(idr_rgb_values, rgb_gt)
        sg_rgb_loss = self.img_loss(sg_rgb_values, rgb_gt)

        return idr_rgb_loss, sg_rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(-1), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_normalsmooth_loss(self, normal, network_object_mask, object_mask):
        mask = (network_object_mask & object_mask).reshape(-1, 4*self.r_patch*self.r_patch).all(dim=-1)
        if self.r_patch < 1 or self.normalsmooth_weight == 0. or mask.sum() == 0.:
            return torch.tensor(0.0).cuda().float()

        normal = normal.view((-1, 4*self.r_patch*self.r_patch, 3))
        return torch.mean(torch.var(normal, dim=1)[mask])

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        idr_rgb_loss, sg_rgb_loss = self.get_rgb_loss(model_outputs['idr_rgb_values'], model_outputs['sg_rgb_values'],
                                                      rgb_gt, network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        normalsmooth_loss = self.get_normalsmooth_loss(model_outputs['normal_values'], network_object_mask, object_mask)

        loss = self.idr_rgb_weight * idr_rgb_loss + \
               self.sg_rgb_weight * sg_rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss + \
               self.normalsmooth_weight * normalsmooth_loss

        return {
            'loss': loss,
            'idr_rgb_loss': idr_rgb_loss,
            'sg_rgb_loss': sg_rgb_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
            'normalsmooth_loss': normalsmooth_loss
        }
