#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# test1233366
import copy

import os
import itertools
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import pickle
import utils.model_utils as model_utils
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
from model.Frame import Frame
import utils.camera_frame_utils as camera_frame_utils
import numpy as np


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,data_dict,model = None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, model = model)
    scene = Scene(dataset, gaussians,data_dict = data_dict, model = model)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background


        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, calc_model = op.calc_model)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        



        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)  #+ 2*sum(gaussians.xyz_gradient_accum )/len(gaussians.xyz_gradient_accum)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()
    
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if iteration == 1:
                weights_dict = {}
            if iteration in testing_iterations:
                weights_dict[iteration] = gaussians.weights.detach().cpu().numpy().astype(int)

    
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                



                if (iteration > opt.opcaity_init_iter) and (iteration % opt.opacity_reset_interval == 0) or (dataset.white_background and iteration == opt.densify_from_iter):
                    

                    for param_group in gaussians.optimizer.param_groups:
                        if param_group["name"] == "opacity":
                            param_group['lr'] = 0.1
                        if param_group["name"] == "scaling":
                            param_group['lr'] = opt.scaling_later
                        if param_group["name"] == "rotation_lr":
                            param_group['lr'] = 0.001
                        # if param_group["name"] == "feature_lr":
                        #     param_group['lr'] = 0.0025

                    gaussians.reset_opacity()

                if (iteration == op.xyz_init_iter):
                    op.calc_model = False
                    for param_group in gaussians.optimizer.param_groups:
                        if param_group["name"] == "xyz":
                                    gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=op.init_xyz_late*gaussians.spatial_lr_scale,
                                                                                lr_final=0.0000016*gaussians.spatial_lr_scale,
                                                                                lr_delay_mult=op.position_lr_delay_mult,
                                                                                max_steps=op.position_lr_max_steps)
                                    gaussians.update_learning_rate( iteration)
                                    


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)


            
            # gaussians.right_wing_twist_joint1.clamp_(-10,10)
            # gaussians.left_wing_twist_joint1.clamp_(-10,10)

            # gaussians.right_wing_twist_joint2.clamp_(-10,10)
            # gaussians.left_wing_twist_joint2.clamp_(-10,10)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    return gaussians,weights_dict


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])   
                metrics = {'loss': l1_test, 'psnr': psnr_test}
       
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def save_run_params(param_values,weights_dict,angle_history,model_name,frame_start,frame_end):
    results = {'params': param_values,'weights': weights_dict,'angles': angle_history,'frames':range(frame_start,frame_end)}
    # Save all into one pickle file
    results_path = os.path.join(path_to_save, f'{model_name}_results.pkl')
    with open(results_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_sweep(sweep_combinations,lp, op, pp, args,params_to_update,model,frame_start,frame_end,update_from_prev_frame = True):
    
    # Loop through each combination of parameters
    for combination in sweep_combinations:

        param_values = dict(zip(sweep_params.keys(), combination))
        with open(os.path.join(lp.model_path, "params.txt"), "w") as f:
            json.dump(param_values, f, indent=4)
        model_name = f"fly_cornell_mov9"
        # model_name = f"mosquito"
        lp.model_path = os.path.join(f"{path_to_save}", f"{frame}/{model_name}/")

        # Update optimization parameters dynamically
        for key, value in param_values.items():
            setattr(op, key, value)  # Update the parameter in `op`

        # Set anomaly detection if enabled
        torch.autograd.set_detect_anomaly(args.detect_anomaly)

        model['ew_to_lab'] = list(data_dict[frame][1].values())[0]['ew_to_lab']
        model['wing_body_ini_pose']['body_location'] = model['ew_to_lab'] @ cm_point
        gauss,weights_dict = training(lp, op, pp, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,data_dict[frame],model)
        
        if update_from_prev_frame == True:
            model['wing_body_ini_pose'] = {key: getattr(gauss, key).cpu().detach().tolist()for key in params_to_update}


        for key in angle_history:
            tensor = getattr(gauss, key)
            angle_history[key].append(tensor.cpu().detach().numpy())
        weights_list.append(weights_dict)

        # Save parameters to a text file
        
        torch.cuda.empty_cache()

        save_run_params(param_values,weights_list,angle_history,model_name,frame_start,frame_end)
        return model




if __name__ == "__main__":
    # Set up command line argument parser
    # initilize model

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1,100,800,1000,1200,1400,1600,1800,2000,2200,2500,3000,4000,5000,6000,8000,10000,15000,20000])#[1_0,1_000,5_000,10_000,15_000, 20_000,45_000,60_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1,100,800,1000,1200,1400,1600,1800,2000,2200,2500,3000,4000,5000,6000,8000,10000,15000,20000])#[1_0,1_000,5_000,10_000,15_000, 20_000,45_000,60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    lp = lp.extract(args)
    op = op.extract(args)
    pp = pp.extract(args)


    frame_start = 1765
    frame_end = 2147



    path_to_save = 'D:/Documents/gaussian_model_output/cornell_mov9'
    path_to_mesh = 'D:/Documents/gaussian_reconstruction/mesh'
    image_path = 'G:/My Drive/Research/gaussian_splatting/gaussian_splatting_input/mov9_cornell/'



    path = f'{lp.source_path}/dict/frames_model_cornell.pkl'
    update_from_prev_frame = True
    params_to_update = {'right_wing_angles','left_wing_angles','body_angles',
                            'left_wing_angle_joint1','left_wing_angle_joint2',
                            'right_wing_angle_joint1','right_wing_angle_joint2',
                            'right_wing_twist_joint1','right_wing_twist_joint2',
                            'left_wing_twist_joint1','left_wing_twist_joint2','thorax_ang'}
    


    
    angle_history = {key : [] for key in params_to_update}

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)


    lp.white_background = True
    lp.dist2_th_min = 0.0000001
    pp.antialiasing = False
    pp.antialiasing = False

    with open(path, 'rb') as file:
        data_dict = pickle.load(file)
    
    network_gui.init(args.ip, args.port)
    safe_state(args.quiet)
    # Generate all combinations of hyperparameters using itertools.product
    sweep_params = {
        # 'position_lr_max_steps': [30_000],
        # 'position_lr_init' : [0.0002],
        # 'position_lr_final' : [0],
        'iterations' :[2000],#[5000],#[900],# [1300], #1000

        'densify_grad_threshold' : [0.00035],
        'densify_until_iter' :[1200],#[3500],#[700],# [1100],#[1200], 850
        'densify_from_iter':[700],#[700],#[300],# [700],#450
        'scaling_lr': [0.0000005],
        # 'rotation_lr': [0],
        # 'feature_lr': [0],
        'scale_model' : [0.0005],
        'model_rotation_lr' : [0.1],#[0.1,0.5],
        # 'model_rotation_lr_rwing' : [1],#,[1,0.5],
        # 'model_rotation_lr_lwing' : [1],#,[1,0.5],
        'model_wing_rotation_lr_init' : [0.5],#0.5
        'model_wing_rotation_lr_final' : [0.1],
        'model_rotation_lr_center' : [0.07],#0.07
        'opcaity_init_iter': [700],#[700],
        'opacity_reset_interval' : [15],#[15],
        # 'body_location_init': [0.00003],
        # 'body_location_final' : [0.00001],
        # 'opacity_lr' : [0.1],#,0.12,0.08]
        'xyz_init_iter' : [400],#[400],#[100],#[400],#,0.12,0.08] #150
        'wing_location' : [0],
        'init_xyz_late' : [0.00016],#0.00016
        'model_rotation_lr_twist' : [0.05],#0.05
        'scaling_later' : [0.005],
        'thorax_lr' : [0.0],#[0.01]0.005

    }
    model = {}
    model['wing_body_ini_pose'] = {'right_wing_angles' : [ 0.0 , 20.0  ,   10.0], # -70 -130  [-60,-100,-0.]
                                    'left_wing_angles' :  [-0.0, 20.0,  0.0], # 90 -130 [70,-150,10.0],
                                    'body_angles' : [0.0, 0.0, 240.0],
                                    'right_wing_angle_joint1' : 0.0,
                                    'left_wing_angle_joint1' : 0.0,
                                    'right_wing_angle_joint2' : 0.0,
                                    'left_wing_angle_joint2' : -0.0,

                                    'right_wing_twist_joint1' : -0.0,
                                    'left_wing_twist_joint1' : -0.0,
                                    'right_wing_twist_joint2' : 0.0,
                                    'left_wing_twist_joint2' : -0.0,
                                    'thorax_ang': -0.0} # -100 -25  [-95.0,  -25.0,  0]


    weights_list = []
    # for idx,frame in enumerate(range(1430,1900,1)):#frame = 1448
    root,body,right_wing,left_wing,model['list_joints_pitch_update'] = model_utils.initilize_skeleton_and_skin(path_to_mesh,skeleton_scale=1/1000, skin_scale = 1) # 1/200 fly- 1/1000, change also nerf
    model['joint_list'],model['skin'],model['weights'],model['bones'] = model_utils.build_skeleton(root,body,right_wing,left_wing)
    sweep_combinations = list(itertools.product(*sweep_params.values()))


    for idx,frame in enumerate(range(frame_start,frame_end,1)):#frame = 1448
 
        op.calc_model = True
        frames_per_cam = [Frame(image_path,frame,cam_num,frames_dict = data_dict) for cam_num in range(4)]
        camera_pixel = np.vstack([frame.camera_center_to_pixel_ray(([frame.cm[0],frame.cm[1]])) for frame in  frames_per_cam])
        camera_center = np.vstack([frame.X0.T for frame in  frames_per_cam])
        cm_point = camera_frame_utils.triangulate_least_square(camera_center,camera_pixel)
        model = run_sweep(sweep_combinations,lp, op, pp, args,params_to_update,model,frame_start,frame_end,update_from_prev_frame = True)


    print("\nframe complete.")
