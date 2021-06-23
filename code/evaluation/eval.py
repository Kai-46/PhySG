import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
import cvxpy as cp
from PIL import Image
import math

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils import vis_util
from model.sg_render import compute_envmap
import imageio
# import pyexr


def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']

    expname = conf.get_string('train.expname') + '-' + kwargs['expname']

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname, os.path.basename(kwargs['data_split_dir']))

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()

    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                           kwargs['data_split_dir'],
                                                                           train_cameras=False)

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  collate_fn=eval_dataset.collate_fn
                                                  )
    total_pixels = eval_dataset.total_pixels
    img_res = eval_dataset.img_res

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
    saved_model_state = torch.load(ckpt_path)
    model.load_state_dict(saved_model_state["model_state_dict"])
    epoch = saved_model_state['epoch']
    print('Loaded checkpoint: ', ckpt_path)

    if kwargs['geometry'].endswith('.pth'):
        print('Reloading geometry from: ', kwargs['geometry'])
        geometry = torch.load(kwargs['geometry'])['model_state_dict']
        geometry = {k: v for k, v in geometry.items() if 'implicit_network' in k}
        print(geometry.keys())
        model_dict = model.state_dict()
        model_dict.update(geometry)
        model.load_state_dict(model_dict)

    #####################################################################################################
    # reset lighting
    #####################################################################################################
    relight = False
    if kwargs['light_sg'].endswith('.npy'):
        print('Loading light from: ', kwargs['light_sg'])
        model.envmap_material_network.load_light(kwargs['light_sg'])
        evaldir = evaldir + '_relight'
        relight = True

    edit_diffuse = False
    if len(kwargs['diffuse_albedo']) > 0:
        print('Setting diffuse albedo to: ', kwargs['diffuse_albedo'])
        evaldir = evaldir + '_editdiffuse'
        edit_diffuse = True

    utils.mkdir_ifnotexists(evaldir)
    print('Output directory is: ', evaldir)

    with open(os.path.join(evaldir, 'ckpt_path.txt'), 'w') as fp:
        fp.write(ckpt_path + '\n')

    ####################################################################################################################
    print("evaluating...")
    model.eval()

    # extract mesh
    if (not edit_diffuse) and (not relight) and eval_dataset.has_groundtruth:
        with torch.no_grad():
            mesh = plt.get_surface_high_res_mesh(
                sdf=lambda x: model.implicit_network(x)[:, 0],
                resolution=kwargs['resolution']
            )

            # Taking the biggest connected component
            components = mesh.split(only_watertight=False)
            areas = np.array([c.area for c in components], dtype=np.float)
            mesh_clean = components[areas.argmax()]
            mesh_clean.export('{0}/mesh.obj'.format(evaldir), 'obj')


    # generate images
    images_dir = evaldir

    all_frames = []
    psnrs = []
    for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
        if eval_dataset.has_groundtruth:
            out_img_name = os.path.basename(eval_dataset.image_paths[indices[0]])[:-4]
        else:
            out_img_name = '{}'.format(indices[0])

        if len(kwargs['view_name']) > 0 and out_img_name != kwargs['view_name']:
            print('Skipping: ', out_img_name)
            continue

        print('Evaluating data_index: ', data_index, len(eval_dataloader))
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["object_mask"] = model_input["object_mask"].cuda()
        model_input['pose'] = model_input['pose'].cuda()

        split = utils.split_input(model_input, total_pixels)
        res = []
        for s in split:
            out = model(s)
            res.append({
                'points': out['points'].detach(),
                'idr_rgb_values': out['idr_rgb_values'].detach(),
                'sg_rgb_values': out['sg_rgb_values'].detach(),
                'normal_values': out['normal_values'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach(),
                'sg_diffuse_albedo_values': out['sg_diffuse_albedo_values'].detach(),
                'sg_diffuse_rgb_values': out['sg_diffuse_rgb_values'].detach(),
                'sg_specular_rgb_values': out['sg_specular_rgb_values'].detach(),
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, total_pixels, batch_size)

        ### re-render with updated diffuse albedo
        if edit_diffuse:
            diffuse_albedo = imageio.imread(kwargs['diffuse_albedo']).astype(np.float32)[:, :, :3]
            if not kwargs['diffuse_albedo'].endswith('.exr'):
                diffuse_albedo /= 255.
            diffuse_albedo = torch.from_numpy(diffuse_albedo).cuda().reshape((-1, 3))

            ray_dirs, _ = rend_util.get_camera_params(model_input["uv"],
                                                            model_input['pose'],
                                                            model_input["intrinsics"])
            sg_ret = model.render_sg_rgb(mask=model_outputs['network_object_mask'],
                                         normals=model_outputs['normal_values'],
                                         view_dirs=-ray_dirs.reshape((-1, 3)),
                                         diffuse_albedo=diffuse_albedo)
            for x in sorted(sg_ret.keys()):
                assert (x in model_outputs)
                model_outputs[x] = sg_ret[x]

        tonemap_img = lambda x: np.power(x, 1./eval_dataset.gamma)
        clip_img = lambda x: np.clip(x, 0., 1.)

        assert (batch_size == 1)

        if kwargs['write_idr']:
            rgb_eval = model_outputs['idr_rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            rgb_eval = clip_img(tonemap_img(rgb_eval))
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/idr_rgb_{1}.png'.format(images_dir, out_img_name))

        rgb_eval = model_outputs['sg_rgb_values']
        rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
        rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        rgb_eval = rgb_eval.transpose(1, 2, 0)
        if kwargs['save_exr']:
            imageio.imwrite('{0}/sg_rgb_{1}.exr'.format(images_dir, out_img_name), rgb_eval)
            # pyexr.write('{0}/sg_rgb_{1}.exr'.format(images_dir, out_img_name), rgb_eval)
            # np.save('{0}/sg_rgb_{1}.npy'.format(images_dir, out_img_name), rgb_eval)

        else:
            rgb_eval = clip_img(tonemap_img(rgb_eval))
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/sg_rgb_{1}.png'.format(images_dir, out_img_name))

            all_frames.append(np.array(img))

        # network_object_mask = model_outputs['network_object_mask']
        # network_object_mask = network_object_mask.reshape(batch_size, total_pixels, 3)
        # network_object_mask = plt.lin2img(network_object_mask, img_res).detach().cpu().numpy()[0]
        # network_object_mask = network_object_mask.transpose(1, 2, 0)
        # img = Image.fromarray((network_object_mask * 255).astype(np.uint8))
        # img.save('{0}/object_mask_{1}.png'.format(images_dir, out_img_name))

        normal = model_outputs['normal_values']
        normal = normal.reshape(batch_size, total_pixels, 3)
        normal = (normal + 1.) / 2.
        normal = plt.lin2img(normal, img_res).detach().cpu().numpy()[0]
        normal = normal.transpose(1, 2, 0)
        if kwargs['save_exr']:
            imageio.imwrite('{0}/normal_{1}.exr'.format(images_dir, out_img_name), normal)
            # pyexr.write('{0}/normal_{1}.exr'.format(images_dir, out_img_name), normal)
            # np.save('{0}/normal_{1}.npy'.format(images_dir, out_img_name), normal)

        else:
            img = Image.fromarray((normal * 255).astype(np.uint8))
            img.save('{0}/normal_{1}.png'.format(images_dir, out_img_name))

        if (not relight) and eval_dataset.has_groundtruth:
            depth = torch.ones(batch_size * total_pixels).cuda().float()
            network_object_mask = model_outputs['network_object_mask'] & model_outputs['object_mask']
            depth_valid = rend_util.get_depth(model_outputs['points'].reshape(batch_size, total_pixels, 3),
                                              model_input['pose']).reshape(-1)[network_object_mask]
            depth[network_object_mask] = depth_valid
            depth[~network_object_mask] = 0.98 * depth_valid.min()
            assert (batch_size == 1)
            network_object_mask = network_object_mask.float().reshape(img_res[0], img_res[1]).cpu()
            depth = depth.reshape(img_res[0], img_res[1]).cpu()

            if kwargs['save_exr']:
                depth = depth * network_object_mask
                depth = depth.numpy()
                imageio.imwrite('{0}/depth_{1}.exr'.format(images_dir, out_img_name), depth)
                # pyexr.write('{0}/depth_{1}.exr'.format(images_dir, out_img_name), depth)
                # np.save('{0}/depth_{1}.npy'.format(images_dir, out_img_name), depth)

            else:
                depth = vis_util.colorize(depth, cmap_name='jet')
                depth = depth * network_object_mask.unsqueeze(-1) + (1. - network_object_mask.unsqueeze(-1))
                depth = depth.numpy()
                img = Image.fromarray((depth * 255).astype(np.uint8))
                img.save('{0}/depth_{1}.png'.format(images_dir, out_img_name))

            # write lighting and materials
            envmap = compute_envmap(lgtSGs=model.envmap_material_network.get_light(), H=256, W=512, upper_hemi=model.envmap_material_network.upper_hemi)
            envmap = envmap.cpu().numpy()
            imageio.imwrite(os.path.join(images_dir, 'envmap.exr'), envmap)

            roughness, specular_reflectance = model.envmap_material_network.get_base_materials()
            with open(os.path.join(images_dir, 'relight_material.txt'), 'w') as fp:
                for i in range(roughness.shape[0]):
                    fp.write('Material {}:\n'.format(i))
                    fp.write('\troughness: {}\n'.format(roughness[i, 0].item()))
                    fp.write('\tspecular_reflectance: ')
                    for j in range(3):
                        fp.write('{}, '.format(specular_reflectance[i, j].item()))
                    fp.write('\n\n')

            rgb_gt = ground_truth['rgb']
            rgb_gt = plt.lin2img(rgb_gt, img_res).numpy()[0].transpose(1, 2, 0)
            if kwargs['save_exr']:
                imageio.imwrite('{0}/gt_{1}.exr'.format(images_dir, out_img_name), rgb_gt)
                # pyexr.write('{0}/gt_{1}.exr'.format(images_dir, out_img_name), rgb_gt)
                # np.save('{0}/gt_{1}.npy'.format(images_dir, out_img_name), rgb_gt)

            else:
                rgb_gt = clip_img(tonemap_img(rgb_gt))
                img = Image.fromarray((rgb_gt * 255).astype(np.uint8))
                img.save('{0}/gt_{1}.png'.format(images_dir, out_img_name))

            mask = model_input['object_mask']
            mask = plt.lin2img(mask.unsqueeze(-1), img_res).cpu().numpy()[0]
            mask = mask.transpose(1, 2, 0)
            rgb_eval_masked = rgb_eval * mask
            rgb_gt_masked = rgb_gt * mask

            psnr = calculate_psnr(rgb_eval_masked, rgb_gt_masked, mask)
            psnrs.append(psnr)

            # verbose mode
            rgb_eval = model_outputs['sg_diffuse_albedo_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            if kwargs['save_exr']:
                imageio.imwrite('{0}/sg_diffuse_albedo_{1}.exr'.format(images_dir, out_img_name), rgb_eval)
                # pyexr.write('{0}/sg_diffuse_albedo_{1}.exr'.format(images_dir, out_img_name), rgb_eval)
                # np.save('{0}/sg_diffuse_albedo_{1}.npy'.format(images_dir, out_img_name), rgb_eval)

            else:
                rgb_eval = clip_img(rgb_eval)
                img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
                img.save('{0}/sg_diffuse_albedo_{1}.png'.format(images_dir, out_img_name))

            rgb_eval = model_outputs['sg_diffuse_rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            if kwargs['save_exr']:
                imageio.imwrite('{0}/sg_diffuse_rgb_{1}.exr'.format(images_dir, out_img_name), rgb_eval)
                # pyexr.write('{0}/sg_diffuse_rgb_{1}.exr'.format(images_dir, out_img_name), rgb_eval)
                # np.save('{0}/sg_diffuse_rgb_{1}.npy'.format(images_dir, out_img_name), rgb_eval)

            else:
                rgb_eval = clip_img(rgb_eval)
                img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
                img.save('{0}/sg_diffuse_rgb_{1}.png'.format(images_dir, out_img_name))

            rgb_eval = model_outputs['sg_specular_rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            if kwargs['save_exr']:
                imageio.imwrite('{0}/sg_specular_rgb_{1}.exr'.format(images_dir, out_img_name), rgb_eval)
                # pyexr.write('{0}/sg_specular_rgb_{1}.exr'.format(images_dir, out_img_name), rgb_eval)
                # np.save('{0}/sg_specular_rgb_{1}.npy'.format(images_dir, out_img_name), rgb_eval)

            else:
                rgb_eval = clip_img(rgb_eval)
                img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
                img.save('{0}/sg_specular_rgb_{1}.png'.format(images_dir, out_img_name))

    if not kwargs['save_exr']:
        imageio.mimwrite(os.path.join(images_dir, 'video_rgb.mp4'), all_frames, fps=15, quality=9)
        print('Done rendering', images_dir)

    if len(psnrs) > 0:
        psnrs = np.array(psnrs).astype(np.float64)
        # print("RENDERING EVALUATION {2}: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std(), scan_id))
        print("RENDERING EVALUATION: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std()))


def get_cameras_accuracy(pred_Rs, gt_Rs, pred_ts, gt_ts,):
    ''' Align predicted pose to gt pose and print cameras accuracy'''

    # find rotation
    d = pred_Rs.shape[-1]
    n = pred_Rs.shape[0]

    Q = torch.addbmm(torch.zeros(d, d, dtype=torch.double), gt_Rs, pred_Rs.transpose(1, 2))
    Uq, _, Vq = torch.svd(Q)
    sv = torch.ones(d, dtype=torch.double)
    sv[-1] = torch.det(Uq @ Vq.transpose(0, 1))
    R_opt = Uq @ torch.diag(sv) @ Vq.transpose(0, 1)
    R_fixed = torch.bmm(R_opt.repeat(n, 1, 1), pred_Rs)

    # find translation
    pred_ts = pred_ts @ R_opt.transpose(0, 1)
    c_opt = cp.Variable()
    t_opt = cp.Variable((1, d))

    constraints = []
    obj = cp.Minimize(cp.sum(
        cp.norm(gt_ts.numpy() - (c_opt * pred_ts.numpy() + np.ones((n, 1), dtype=np.double) @ t_opt), axis=1)))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    t_fixed = c_opt.value * pred_ts.numpy() + np.ones((n, 1), dtype=np.double) * t_opt.value

    # Calculate transaltion error
    t_error = np.linalg.norm(t_fixed - gt_ts.numpy(), axis=-1)
    t_error = t_error
    t_error_mean = np.mean(t_error)
    t_error_medi = np.median(t_error)

    # Calculate rotation error
    R_error = compare_rotations(R_fixed, gt_Rs)

    R_error = R_error.numpy()
    R_error_mean = np.mean(R_error)
    R_error_medi = np.median(R_error)

    print('CAMERAS EVALUATION: R error mean = {0} ; t error mean = {1} ; R error median = {2} ; t error median = {3}'
          .format("%.2f" % R_error_mean, "%.2f" % t_error_mean, "%.2f" % R_error_medi, "%.2f" % t_error_medi))

    # return alignment and aligned pose
    return R_opt.numpy(), t_opt.value, c_opt.value, R_fixed.numpy(), t_fixed

def compare_rotations(R1, R2):
    cos_err = (torch.bmm(R1, R2.transpose(1, 2))[:, torch.arange(3), torch.arange(3)].sum(dim=-1) - 1) / 2
    cos_err[cos_err > 1] = 1
    cos_err[cos_err < -1] = -1
    return cos_err.acos() * 180 / np.pi

def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/default.conf')
    parser.add_argument('--data_split_dir', type=str, default='')
    parser.add_argument('--gamma', type=float, default=1., help='gamma correction coefficient')

    parser.add_argument('--save_exr', default=False, action="store_true", help='')

    parser.add_argument('--light_sg', type=str, default='', help='')
    parser.add_argument('--geometry', type=str, default='', help='')
    parser.add_argument('--diffuse_albedo', type=str, default='', help='')
    parser.add_argument('--view_name', type=str, default='', help='')

    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')

    parser.add_argument('--write_idr', default=False, action="store_true", help='')

    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--is_uniform_grid', default=False, action="store_true", help='If set, evaluate marching cube with uniform grid.')

    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate(conf=opt.conf,
             write_idr=opt.write_idr,
             gamma=opt.gamma,
             data_split_dir=opt.data_split_dir,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name='evals',
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             resolution=opt.resolution,
             save_exr=opt.save_exr,
             light_sg=opt.light_sg,
             geometry=opt.geometry,
             view_name=opt.view_name,
             diffuse_albedo=opt.diffuse_albedo,
             )
