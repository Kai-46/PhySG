import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import imageio

#######################################################################################################
# compute envmap from SG
#######################################################################################################
def compute_envmap(lgtSGs, H, W, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                           dim=-1)    # [H, W, 3]
    print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
    print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
    print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

    lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, 3))
    return envmap





r = R.from_euler('yxz', [90, 0, 0], degrees=True)
try:
    rotation = r.as_matrix()
except:
    rotation = r.as_dcm()

fpath = '/home/kz298/idr_sg/code/all_envmaps/envmap1_sg_fit/tmp_lgtSGs_100.npy'
lgtSGs = np.load(fpath)
lgtSGLobes = lgtSGs[:, :3] / (np.linalg.norm(lgtSGs[:, :3], axis=-1, keepdims=True) + 1e-8)
lgtSGLambdas = np.abs(lgtSGs[:, 3:4])
lgtSGMus = np.abs(lgtSGs[:, 4:])

lgtSGLobes_rot = np.matmul(lgtSGLobes, rotation.T)
lgtSGs_rot = np.concatenate((lgtSGLobes_rot, lgtSGLambdas, lgtSGMus), axis=-1).astype(np.float32)
np.save(fpath[:-4]+'_rot.npy', lgtSGs_rot)


envmap = compute_envmap(torch.from_numpy(lgtSGs_rot), H=256, W=512).numpy()
im = np.power(envmap, 1./2.2)
im = np.clip(im, 0., 1.)
imageio.imwrite(fpath[:-4]+'_rot_envmap.png', np.uint8(im * 255.))
