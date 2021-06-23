import imageio
imageio.plugins.freeimage.download()
import torch
import torch.nn as nn
import numpy as np
import imageio
import cv2
import os

TINY_NUMBER = 1e-8


def parse_raw_sg(sg):
    SGLobes = sg[..., :3] / (torch.norm(sg[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
    SGLambdas = torch.abs(sg[..., 3:4])
    SGMus = torch.abs(sg[..., -3:])
    return SGLobes, SGLambdas, SGMus



#######################################################################################################
# compute envmap from SG
#######################################################################################################
def SG2Envmap(lgtSGs, H, W, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                           dim=-1)    # [H, W, 3]
    # print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
    # print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
    # print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

    # lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, 3))
    return envmap


# def SG2Envmap(lgtSGs, H, W):
#     numLgtSGs = lgtSGs.shape[0]

#     phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(0.0, 2 * np.pi, W)])
#     viewdirs = torch.stack((torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)),
#                            dim=2).cuda()

#     viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

#     # [n_envsg, 7]
#     sum_sg2 = torch.cat(parse_raw_sg(lgtSGs), dim=-1)

#     # [..., n_envsg, 7]
#     sh = list(viewdirs.shape[:-2])
#     sum_sg2 = sum_sg2.view([1, ] * len(sh) + [numLgtSGs, 7]).expand(sh + [-1, -1])

#     # [..., n_envsg, 3]
#     rgb = sum_sg2[..., -3:] * torch.exp(sum_sg2[..., 3:4] *
#                                         (torch.sum(viewdirs * sum_sg2[..., :3], dim=-1, keepdim=True) - 1.))
#     rgb = torch.sum(rgb, dim=-2)  # [..., 3]

#     env_map = rgb.reshape((H, W, 3))

#     return env_map


# load ground-truth envmap
filename = '/home/kz298/envmap_museum_clamp997.exr'
filename = os.path.abspath(filename)
gt_envmap = imageio.imread(filename)[:,:,:3]
gt_envmap = cv2.resize(gt_envmap, (512, 256), interpolation=cv2.INTER_AREA)
gt_envmap = torch.from_numpy(gt_envmap).cuda()
H, W = gt_envmap.shape[:2]
print(H, W)

out_dir = filename[:-4]
print(out_dir)
os.makedirs(out_dir, exist_ok=True)
assert (os.path.isdir(out_dir))

numLgtSGs = 128
lgtSGs = nn.Parameter(torch.randn(numLgtSGs, 7).cuda())  # lobe + lambda + mu
lgtSGs.data[..., 3:4] *= 100.
lgtSGs.requires_grad = True

optimizer = torch.optim.Adam([lgtSGs,], lr=1e-2)

N_iter = 100000

pretrained_file = os.path.join(out_dir, 'sg_{}.npy'.format(numLgtSGs))
if os.path.isfile(pretrained_file):
    print('Loading: ', pretrained_file)
    lgtSGs.data.copy_(torch.from_numpy(np.load(pretrained_file)).cuda())

for step in range(N_iter):
    optimizer.zero_grad()
    env_map = SG2Envmap(lgtSGs, H, W)
    loss = torch.mean((env_map - gt_envmap) * (env_map - gt_envmap))
    loss.backward()
    optimizer.step()

    if step % 30 == 0:
        print('step: {}, loss: {}'.format(step, loss.item()))

    if step % 100 == 0:
        envmap_check = env_map.clone().detach().cpu().numpy()
        gt_envmap_check = gt_envmap.clone().detach().cpu().numpy()
        im = np.concatenate((gt_envmap_check, envmap_check), axis=0)
        im = np.power(im, 1./2.2)
        im = np.clip(im, 0., 1.)
        # im = (im - im.min()) / (im.max() - im.min() + TINY_NUMBER)
        im = np.uint8(im * 255.)
        imageio.imwrite(os.path.join(out_dir, 'log_im_{}.png'.format(numLgtSGs)), im)

        np.save(os.path.join(out_dir, 'sg_{}.npy'.format(numLgtSGs)), lgtSGs.clone().detach().cpu().numpy())