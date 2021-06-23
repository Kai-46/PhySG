import os
import sys
from datetime import datetime

import imageio
import numpy as np
import torch
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from model.sg_render import compute_envmap

imageio.plugins.freeimage.download()


class IDRTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.max_niters = kwargs['max_niters']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.freeze_geometry = kwargs['freeze_geometry']
        self.train_cameras = kwargs['train_cameras']

        self.freeze_idr = kwargs['freeze_idr']
        self.write_idr = kwargs['write_idr']

        self.expname = self.conf.get_string('train.expname') + '-' + kwargs['expname']
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.idr_optimizer_params_subdir = "IDROptimizerParameters"
        self.idr_scheduler_params_subdir = "IDRSchedulerParameters"
        self.sg_optimizer_params_subdir = "SGOptimizerParameters"
        self.sg_scheduler_params_subdir = "SGSchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir))

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        if self.train_cameras:
            self.optimizer_cam_params_subdir = "OptimizerCamParameters"
            self.cam_params_subdir = "CamParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                                          kwargs['data_split_dir'], self.train_cameras)
        # self.train_dataset.return_single_img('rgb_000000.exr')
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                            kwargs['data_split_dir'], self.train_cameras)
        # self.plot_dataset.return_single_img('rgb_000000.exr')
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.idr_optimizer = torch.optim.Adam(list(self.model.implicit_network.parameters()) + list(self.model.rendering_network.parameters()),
                                              lr=self.conf.get_float('train.idr_learning_rate'))
        self.idr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.idr_optimizer,
                                                              self.conf.get_list('train.idr_sched_milestones', default=[]),
                                                              gamma=self.conf.get_float('train.idr_sched_factor', default=0.0))

        self.sg_optimizer = torch.optim.Adam(self.model.envmap_material_network.parameters(),
                                              lr=self.conf.get_float('train.sg_learning_rate'))
        self.sg_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.sg_optimizer,
                                                              self.conf.get_list('train.sg_sched_milestones', default=[]),
                                                              gamma=self.conf.get_float('train.sg_sched_factor', default=0.0))
        # settings for camera optimization
        if self.train_cameras:
            num_images = len(self.train_dataset)
            self.pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).cuda()
            self.pose_vecs.weight.data.copy_(self.train_dataset.get_pose_init())

            self.optimizer_cam = torch.optim.SparseAdam(self.pose_vecs.parameters(), self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.idr_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.idr_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.idr_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.idr_scheduler.load_state_dict(data["scheduler_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.sg_optimizer_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.sg_optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.sg_scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.sg_scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.train_cameras:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        if kwargs['geometry'].endswith('.pth'):
            print('Reloading geometry from: ', kwargs['geometry'])
            geometry = torch.load(kwargs['geometry'])['model_state_dict']
            geometry = {k: v for k, v in geometry.items() if 'implicit_network' in k}
            print(geometry.keys())
            model_dict = self.model.state_dict()
            model_dict.update(geometry)
            self.model.load_state_dict(model_dict)

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_conf = self.conf.get_config('plot')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch * self.n_batches > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.idr_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.idr_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.idr_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.idr_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, "latest.pth"))

        if self.train_cameras:
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))

    def plot_to_disk(self):
        self.model.eval()
        if self.train_cameras:
            self.pose_vecs.eval()
        sampling_idx = self.train_dataset.sampling_idx
        self.train_dataset.change_sampling_idx(-1)
        indices, model_input, ground_truth = next(iter(self.plot_dataloader))

        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["object_mask"] = model_input["object_mask"].cuda()

        if self.train_cameras:
            pose_input = self.pose_vecs(indices.cuda())
            model_input['pose'] = pose_input
        else:
            model_input['pose'] = model_input['pose'].cuda()

        split = utils.split_input(model_input, self.total_pixels)
        res = []
        for s in split:
            out = self.model(s)
            res.append({
                'points': out['points'].detach(),
                'idr_rgb_values': out['idr_rgb_values'].detach(),
                'sg_rgb_values': out['sg_rgb_values'].detach(),
                'sg_diffuse_albedo_values': out['sg_diffuse_albedo_values'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach(),
                'normal_values': out['normal_values'].detach(),
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

        plt.plot(self.write_idr, self.train_dataset.gamma, self.model,
                 indices,
                 model_outputs,
                 model_input['pose'],
                 ground_truth['rgb'],
                 self.plots_dir,
                 self.cur_iter,
                 self.img_res,
                 **self.plot_conf
                 )

        # log environment map
        envmap = compute_envmap(lgtSGs=self.model.envmap_material_network.get_light(), H=256, W=512, upper_hemi=self.model.envmap_material_network.upper_hemi)
        envmap = envmap.cpu().numpy()
        imageio.imwrite(os.path.join(self.plots_dir, 'envmap_{}.exr'.format(self.cur_iter)), envmap)

        self.model.train()
        if self.train_cameras:
            self.pose_vecs.train()
        self.train_dataset.sampling_idx = sampling_idx

    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)
        mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)

        if self.freeze_idr:
            print('Freezing idr (both geometry and rendering network)!!!')
            self.model.freeze_idr()
        elif self.freeze_geometry:
            print('Freezing geometry!!!')
            self.model.freeze_geometry()

        # print('Freezing lighting and specular BRDF!')
        # self.model.envmap_material_network.freeze_all_except_diffuse()

        # print('Freezing appearance!')
        # self.model.envmap_material_network.freeze_all()

        for epoch in range(self.start_epoch, self.nepochs + 1):
            if self.loss.r_patch < 1:
                self.train_dataset.change_sampling_idx(self.num_pixels)
            else:
                self.train_dataset.change_sampling_idx_patch(self.num_pixels // (4*self.loss.r_patch*self.loss.r_patch),
                                                             self.loss.r_patch)

            if self.cur_iter > self.max_niters:
                self.save_checkpoints(epoch)
                self.plot_to_disk()
                print('Training has reached max number of iterations: {}; exiting...'.format(self.cur_iter))
                exit(0)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                if self.cur_iter in self.alpha_milestones:
                    self.loss.alpha = self.loss.alpha * self.alpha_factor

                if self.cur_iter % self.ckpt_freq == 0:
                    self.save_checkpoints(epoch)

                if self.cur_iter % self.plot_freq == 0:
                    self.plot_to_disk()

                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["object_mask"] = model_input["object_mask"].cuda()

                if self.train_cameras:
                    pose_input = self.pose_vecs(indices.cuda())
                    model_input['pose'] = pose_input
                else:
                    model_input['pose'] = model_input['pose'].cuda()

                model_outputs = self.model(model_input)
                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']

                self.idr_optimizer.zero_grad()
                self.sg_optimizer.zero_grad()
                if self.train_cameras:
                    self.optimizer_cam.zero_grad()

                loss.backward()

                self.idr_optimizer.step()
                self.sg_optimizer.step()
                if self.train_cameras:
                    self.optimizer_cam.step()

                if self.cur_iter % 50 == 0:
                    roughness, specular_albedo = self.model.envmap_material_network.get_base_materials()
                    print(
                        '{} [{}/{}] ({}/{}): loss = {}, idr_rgb_loss = {}, sg_rgb_loss = {}, eikonal_loss = {}, '
                        'mask_loss = {}, normalsmooth_loss = {}, alpha = {}, idr_lr = {}, sg_lr = {}, idr_psnr = {}, sg_psnr = {}, '
                        'roughness = {}, specular_albedo = {}, idr_rgb_weight = {}, sg_rgb_weight = {}, mask_weight = {}, eikonal_weight = {}, '
                        'normal_smooth_weight = {} '
                            .format(self.expname, epoch, self.cur_iter, data_index, self.n_batches, loss.item(),
                                    loss_output['idr_rgb_loss'].item(),
                                    loss_output['sg_rgb_loss'].item(),
                                    loss_output['eikonal_loss'].item(),
                                    loss_output['mask_loss'].item(),
                                    loss_output['normalsmooth_loss'].item(),
                                    self.loss.alpha,
                                    self.idr_scheduler.get_lr()[0],
                                    self.sg_scheduler.get_lr()[0],
                                    mse2psnr(loss_output['idr_rgb_loss'].item()),
                                    mse2psnr(loss_output['sg_rgb_loss'].item()),
                                    roughness[0, 0].item(), specular_albedo[0, 0].item(), 
                                    self.loss.idr_rgb_weight, self.loss.sg_rgb_weight, self.loss.mask_weight,
                                    self.loss.eikonal_weight, self.loss.normalsmooth_weight))

                    self.writer.add_scalar('idr_rgb_loss', loss_output['idr_rgb_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('idr_psnr', mse2psnr(loss_output['idr_rgb_loss'].item()), self.cur_iter)
                    self.writer.add_scalar('sg_rgb_loss', loss_output['sg_rgb_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('sg_psnr', mse2psnr(loss_output['sg_rgb_loss'].item()), self.cur_iter)
                    self.writer.add_scalar('eikonal_loss', loss_output['eikonal_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('mask_loss', loss_output['mask_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('alpha', self.loss.alpha, self.cur_iter)
                    self.writer.add_scalar('mask_weight', self.loss.mask_weight, self.cur_iter)
                    self.writer.add_scalar('eikonal_weight', self.loss.eikonal_weight, self.cur_iter)
                    self.writer.add_scalar('idr_rgb_weight', self.loss.idr_rgb_weight, self.cur_iter)
                    self.writer.add_scalar('sg_rgb_weight', self.loss.sg_rgb_weight, self.cur_iter)
                    self.writer.add_scalar('normalsmooth_loss', loss_output['normalsmooth_loss'].item(), self.cur_iter)
                    self.writer.add_scalar('r_patch', self.loss.r_patch, self.cur_iter)
                    self.writer.add_scalar('normalsmooth_weight', self.loss.normalsmooth_weight, self.cur_iter)
                    self.writer.add_scalar('gamma_correction', self.train_dataset.gamma, self.cur_iter)
                    self.writer.add_scalar('roughness', roughness[0, 0].item(), self.cur_iter)
                    self.writer.add_scalar('specular_albedo', specular_albedo[0, 0].item(), self.cur_iter)
                    self.writer.add_scalar('white_specular', float(self.model.envmap_material_network.white_specular), self.cur_iter)
                    self.writer.add_scalar('white_light', float(self.model.envmap_material_network.white_light), self.cur_iter)
                    self.writer.add_scalar('idr_lrate', self.idr_scheduler.get_lr()[0], self.cur_iter)
                    self.writer.add_scalar('sg_lrate', self.sg_scheduler.get_lr()[0], self.cur_iter)

                self.cur_iter += 1

                self.idr_scheduler.step()
                self.sg_scheduler.step()
