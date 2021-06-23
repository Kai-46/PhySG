import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from collections import OrderedDict

from model.embedder import get_embedder


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


### uniformly distribute points on a sphere
def fibonacci_sphere(samples=1):
    '''
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    :param samples:
    :return:
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points

def compute_energy(lgtSGs):
    lgtLambda = torch.abs(lgtSGs[:, 3:4])       # [M, 1]
    lgtMu = torch.abs(lgtSGs[:, 4:])               # [M, 3]
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy

class EnvmapMaterialNetwork(nn.Module):
    def __init__(self, multires=0, dims=[256, 256, 256],
                 white_specular=False,
                 white_light=False,
                 num_lgt_sgs=32,
                 num_base_materials=2,
                 upper_hemi=False,
                 fix_specular_albedo=False,
                 specular_albedo=[-1.,-1.,-1.]):
        super().__init__()

        input_dim = 3
        self.embed_fn = None
        if multires > 0:
            self.embed_fn, input_dim = get_embedder(multires)

        # self.actv_fn = nn.ReLU()
        self.actv_fn = nn.ELU()
        # self.actv_fn = nn.LeakyReLU(0.05)
        ############## spatially-varying diffuse albedo############
        print('Diffuse albedo network size: ', dims)
        diffuse_albedo_layers = []
        dim = input_dim
        for i in range(len(dims)):
            diffuse_albedo_layers.append(nn.Linear(dim, dims[i]))
            diffuse_albedo_layers.append(self.actv_fn)
            dim = dims[i]
        diffuse_albedo_layers.append(nn.Linear(dim, 3))

        self.diffuse_albedo_layers = nn.Sequential(*diffuse_albedo_layers)
        # self.diffuse_albedo_layers.apply(weights_init)

        ##################### specular rgb ########################
        self.numLgtSGs = num_lgt_sgs
        self.numBrdfSGs = num_base_materials
        print('Number of Light SG: ', self.numLgtSGs)
        print('Number of BRDF SG: ', self.numBrdfSGs)
        # by using normal distribution, the lobes are uniformly distributed on a sphere at initialization
        self.white_light = white_light
        if self.white_light:
            print('Using white light!')
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 5), requires_grad=True)   # [M, 5]; lobe + lambda + mu
            # self.lgtSGs.data[:, -1] = torch.clamp(torch.abs(self.lgtSGs.data[:, -1]), max=0.01)
        else:
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True)   # [M, 7]; lobe + lambda + mu
            self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))
            # self.lgtSGs.data[:, -3:] = torch.clamp(torch.abs(self.lgtSGs.data[:, -3:]), max=0.01)

        # make sure lambda is not too close to zero
        self.lgtSGs.data[:, 3:4] = 20. + torch.abs(self.lgtSGs.data[:, 3:4] * 100.)
        # make sure total energy is around 1.
        energy = compute_energy(self.lgtSGs.data)
        # print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())
        self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi
        energy = compute_energy(self.lgtSGs.data)
        print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        # deterministicly initialize lobes
        lobes = fibonacci_sphere(self.numLgtSGs).astype(np.float32)
        self.lgtSGs.data[:, :3] = torch.from_numpy(lobes)
        # check if lobes are in upper hemisphere
        self.upper_hemi = upper_hemi
        if self.upper_hemi:
            print('Restricting lobes to upper hemisphere!')
            self.restrict_lobes_upper = lambda lgtSGs: torch.cat((lgtSGs[..., :1], torch.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)

            # limit lobes to upper hemisphere
            self.lgtSGs.data = self.restrict_lobes_upper(self.lgtSGs.data)

        self.white_specular = white_specular
        self.fix_specular_albedo = fix_specular_albedo
        if self.fix_specular_albedo:
            print('Fixing specular albedo: ', specular_albedo)
            specular_albedo = np.array(specular_albedo).astype(np.float32)
            assert(self.numBrdfSGs == 1)
            assert(np.all(np.logical_and(specular_albedo > 0., specular_albedo < 1.)))
            self.specular_reflectance = nn.Parameter(torch.from_numpy(specular_albedo).reshape((self.numBrdfSGs, 3)),
                                                     requires_grad=False)  # [K, 1]
        else:
            if self.white_specular:
                print('Using white specular reflectance!')
                self.specular_reflectance = nn.Parameter(torch.randn(self.numBrdfSGs, 1),
                                                         requires_grad=True)   # [K, 1]
            else:
                self.specular_reflectance = nn.Parameter(torch.randn(self.numBrdfSGs, 3),
                                                         requires_grad=True)   # [K, 3]
            self.specular_reflectance.data = torch.abs(self.specular_reflectance.data)

        # optimize
        # roughness = [np.random.uniform(-1.5, -1.0) for i in range(self.numBrdfSGs)]       # small roughness
        roughness = [np.random.uniform(1.5, 2.0) for i in range(self.numBrdfSGs)]           # big roughness
        roughness = np.array(roughness).astype(dtype=np.float32).reshape((self.numBrdfSGs, 1))  # [K, 1]
        print('init roughness: ', 1.0 / (1.0 + np.exp(-roughness)))
        self.roughness = nn.Parameter(torch.from_numpy(roughness),
                                      requires_grad=True)

        # blending weights
        self.blending_weights_layers = []
        if self.numBrdfSGs > 1:
            dim = input_dim
            for i in range(3):
                self.blending_weights_layers.append(nn.Sequential(nn.Linear(dim, 256), self.actv_fn))
                dim = 256
            self.blending_weights_layers.append(nn.Linear(dim, self.numBrdfSGs))
            self.blending_weights_layers = nn.Sequential(*self.blending_weights_layers)

    def freeze_all_except_diffuse(self):
        self.lgtSGs.requires_grad = False
        self.specular_reflectance.requires_grad = False
        self.roughness.requires_grad = False
        if self.numBrdfSGs > 1:
            for param in self.blending_weights_layers.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_light(self):
        lgtSGs = self.lgtSGs.clone().detach()
        if self.white_light:
            lgtSGs = torch.cat((lgtSGs, lgtSGs[..., -1:], lgtSGs[..., -1:]), dim=-1)
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        return lgtSGs

    def load_light(self, path):
        assert(path.endswith('.npy'))

        device = self.lgtSGs.data.device
        self.lgtSGs = nn.Parameter(torch.from_numpy(np.load(path)).to(device), requires_grad=True)
        self.numLgtSGs = self.lgtSGs.data.shape[0]
        if self.lgtSGs.data.shape[1] == 7:
            self.white_light = False

    def get_base_materials(self):
        roughness = torch.sigmoid(self.roughness.clone().detach())
        if self.fix_specular_albedo:
            specular_reflectacne = self.specular_reflectance
        else:
            specular_reflectacne = torch.sigmoid(self.specular_reflectance.clone().detach())
            if self.white_specular:
                specular_reflectacne = specular_reflectacne.expand((-1, 3))     # [K, 3]
        return roughness, specular_reflectacne

    def forward(self, points):
        if points is None:
            diffuse_albedo = None
            blending_weights = None
        else:
            if self.embed_fn is not None:
                points = self.embed_fn(points)
            diffuse_albedo = torch.sigmoid(self.diffuse_albedo_layers(points))

            if self.numBrdfSGs > 1:
                blending_weights = F.softmax(self.blending_weights_layers(points), dim=-1)
            else:
                blending_weights = None

        if self.fix_specular_albedo:
            specular_reflectacne = self.specular_reflectance
        else:
            specular_reflectacne = torch.sigmoid(self.specular_reflectance)
            if self.white_specular:
                specular_reflectacne = specular_reflectacne.expand((-1, 3))     # [K, 3]

        roughness = torch.sigmoid(self.roughness)

        lgtSGs = self.lgtSGs
        if self.white_light:
            lgtSGs = torch.cat((lgtSGs, lgtSGs[..., -1:], lgtSGs[..., -1:]), dim=-1)
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        ret = dict([
            ('sg_lgtSGs', lgtSGs),
            ('sg_specular_reflectance', specular_reflectacne),
            ('sg_roughness', roughness),
            ('sg_diffuse_albedo', diffuse_albedo),
            ('sg_blending_weights', blending_weights)
        ])
        return ret
