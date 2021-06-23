import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork
from model.sg_envmap_material import EnvmapMaterialNetwork
from model.sg_render import render_with_sg


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        self.feature_vector_size = feature_vector_size
        print('ImplicitNetowork feature_vector_size: ', self.feature_vector_size)
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            multires_xyz=0
    ):
        super().__init__()

        self.feature_vector_size = feature_vector_size
        print('RenderingNetowork feature_vector_size: ', self.feature_vector_size)

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            print('Applying positional encoding to view directions: ', multires_view)
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.embedxyz_fn = None
        if multires_xyz > 0:
            print('Applying positional encoding to xyz: ', multires_xyz)
            embedxyz_fn, input_ch = get_embedder(multires_xyz)
            self.embedxyz_fn = embedxyz_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors=None):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.embedxyz_fn is not None:
            points = self.embedxyz_fn(points)

        if feature_vectors is not None:
            if self.mode == 'idr':
                rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
            elif self.mode == 'no_view_dir':
                rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
            elif self.mode == 'no_normal':
                rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        else:
            if self.mode == 'idr':
                rendering_input = torch.cat([points, view_dirs, normals], dim=-1)
            elif self.mode == 'no_view_dir':
                rendering_input = torch.cat([points, normals], dim=-1)
            elif self.mode == 'no_normal':
                rendering_input = torch.cat([points, view_dirs], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return (x + 1.) / 2.


class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.envmap_material_network = EnvmapMaterialNetwork(**conf.get_config('envmap_material_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

    def freeze_geometry(self):
        for param in self.implicit_network.parameters():
            param.requires_grad = False

    def unfreeze_geometry(self):
        for param in self.implicit_network.parameters():
            param.requires_grad = True

    def freeze_idr(self):
        self.freeze_geometry()
        for param in self.rendering_network.parameters():
            param.requires_grad = False

    def forward(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        idr_rgb_values = torch.ones_like(points).float().cuda()
        sg_rgb_values = torch.ones_like(points).float().cuda()
        normal_values = torch.ones_like(points).float().cuda()
        sg_diffuse_rgb_values = torch.ones_like(points).float().cuda()
        sg_diffuse_albedo_values = torch.ones_like(points).float().cuda()
        sg_specular_rgb_values = torch.zeros_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            view_dirs = -ray_dirs[surface_mask]  # ----> camera
            ret = self.get_rbg_value(differentiable_surface_points, view_dirs)

            idr_rgb_values[surface_mask] = ret['idr_rgb']
            sg_rgb_values[surface_mask] = ret['sg_rgb']
            normal_values[surface_mask] = ret['normals']

            sg_diffuse_rgb_values[surface_mask] = ret['sg_diffuse_rgb']
            sg_diffuse_albedo_values[surface_mask] = ret['sg_diffuse_albedo']
            sg_specular_rgb_values[surface_mask] = ret['sg_specular_rgb']

        output = {
            'points': points,
            'idr_rgb_values': idr_rgb_values,
            'sg_rgb_values': sg_rgb_values,
            'normal_values': normal_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'sg_diffuse_rgb_values': sg_diffuse_rgb_values,
            'sg_diffuse_albedo_values': sg_diffuse_albedo_values,
            'sg_specular_rgb_values': sg_specular_rgb_values,
        }

        return output

    def get_rbg_value(self, points, view_dirs):
        feature_vectors = None
        if self.feature_vector_size > 0:
            output = self.implicit_network(points)
            feature_vectors = output[:, 1:]

        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)    # ----> camera
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)  # ----> camera

        ret = { 'normals': normals, }

        ### idr renderer
        idr_rgb = self.rendering_network(points, normals, view_dirs, feature_vectors)
        ret['idr_rgb'] = idr_rgb

        ### sg renderer
        sg_envmap_material = self.envmap_material_network(points)
        sg_ret = render_with_sg(lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                roughness=sg_envmap_material['sg_roughness'],
                                diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                                normal=normals, viewdirs=view_dirs,
                                blending_weights=sg_envmap_material['sg_blending_weights'])
        ret.update(sg_ret)
        return ret

    def render_sg_rgb(self, mask, normals, view_dirs, diffuse_albedo):
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)    # ----> camera
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)  # ----> camera

        ### sg renderer
        sg_envmap_material = self.envmap_material_network(points=None)
        ### split
        split_size = 20000
        normals_split = torch.split(normals, split_size, dim=0)
        view_dirs_split = torch.split(view_dirs, split_size, dim=0)
        diffuse_albedo_split = torch.split(diffuse_albedo, split_size, dim=0)
        merged_ret = {}
        for i in range(len(normals_split)):
            sg_ret = render_with_sg(lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                    specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                    roughness=sg_envmap_material['sg_roughness'],
                                    diffuse_albedo=diffuse_albedo_split[i],
                                    normal=normals_split[i], viewdirs=view_dirs_split[i],
                                    blending_weights=sg_envmap_material['sg_blending_weights'])
            if i == 0:
                for x in sorted(sg_ret.keys()):
                    merged_ret[x] = [sg_ret[x].detach(), ]
            else:
                for x in sorted(sg_ret.keys()):
                    merged_ret[x].append(sg_ret[x].detach())
        for x in sorted(merged_ret.keys()):
            merged_ret[x] = torch.cat(merged_ret[x], dim=0)

        sg_ret = merged_ret
        ### maskout
        for x in sorted(sg_ret.keys()):
            sg_ret[x][~mask] = 1.

        output = {
            'sg_rgb_values': sg_ret['sg_rgb'],
            'sg_diffuse_rgb_values': sg_ret['sg_diffuse_rgb'],
            'sg_diffuse_albedo_values': diffuse_albedo,
            'sg_specular_rgb_values': sg_ret['sg_specular_rgb'],
        }

        return output