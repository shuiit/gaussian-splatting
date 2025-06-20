
from Joint import Joint
from Skin import Skin
import numpy as np
import torch
def initilize_skeleton_and_skin(path_to_mesh,skeleton_scale = 1,skin_scale = 1):
    
    pitch_body = 0
    root = Joint([1.0,0,0],[0.0,-pitch_body,0],parent = None, end_joint_of_bone = False, scale = skeleton_scale, name = 'root')
    # root = Joint([0.3,0,0.8],[0.0,-pitch_body,0],parent = None, end_joint_of_bone = False, scale = skeleton_scale, name = 'root')

    neck = Joint([0.6,0,0.3],[0.0,pitch_body,0],parent = root, end_joint_of_bone = False, scale = skeleton_scale,name = 'neck')
    neck_thorax =  Joint([0.6,0,0.3],[0.0,-25,0], parent = root, end_joint_of_bone = False, scale = skeleton_scale,name = 'neck_thorax')
    head  =Joint([1,0.0,0],[0,0,0.0], parent = neck, scale = skeleton_scale,name = 'head')
    thorax  =Joint([-1,0,0.0],[0,25,0.0], parent= neck_thorax ,scale = skeleton_scale,name = 'thorax')
    abdomen = Joint([-2,0,0.0],[0.0,0,0], parent = thorax, scale = skeleton_scale,name = 'abdomen')

    right_sp_no_bone = Joint([0,0,0.34],[0.0,pitch_body,0],parent = root, end_joint_of_bone = False , scale = skeleton_scale, color = 'red', rotation_order = 'zxy',name = 'right_sp_no_bone')
    right_wing_root = Joint([0,-0.34,-0.05],[0.0,0,0], parent = right_sp_no_bone, end_joint_of_bone = False, scale = skeleton_scale, color = 'red',rotation_order = 'zxy',name = 'right_wing_root')
    right_wing_joint1 = Joint([0.05,-0.7,0],[0.0,0,0], parent = right_wing_root,  scale = skeleton_scale, color = 'red',rotation_order = 'zxy',name = 'right_wing_joint1')
    right_wing_joint2 = Joint([0.05,-0.9,0],[0.0,0,0], parent = right_wing_joint1,  scale = skeleton_scale, color = 'red',rotation_order = 'zxy',name = 'right_wing_joint2')
    right_wing_tip = Joint([-0.25,-0.6,0],[0.0,0,0], parent = right_wing_joint2, scale = skeleton_scale, color = 'red',rotation_order = 'zxy',name = 'right_wing_tip')

    # right_sp_no_bone = Joint([0,-0,0.5],[0.0,pitch_body,0],parent = root , scale = skeleton_scale, end_joint_of_bone = False, color = 'red', rotation_order = 'zxy',name = 'right_sp_no_bone')
    # right_wing_root = Joint([0,-0.5,-0],[0.0,0,0], parent = right_sp_no_bone, end_joint_of_bone = False, scale = skeleton_scale, color = 'red',rotation_order = 'zxy',name = 'right_wing_root')
    # right_wing_center1 = Joint([0.05,-0.5,0.],[0.0,0,0], parent = right_wing_root,  scale = skeleton_scale, color = 'red',rotation_order = 'zxy',name = 'right_wing_center1')
    # right_wing_center2 = Joint([0.05,-0.6,0],[0.0,0,0], parent = right_wing_center1,  scale = skeleton_scale, color = 'red',rotation_order = 'zxy',name = 'right_wing_center2')
    # right_wing_tip = Joint([-0.25,-0.8,0],[0.0,0,0], parent = right_wing_center2, scale = skeleton_scale, color = 'red',rotation_order = 'zxy',name = 'right_wing_tip')
    
    
    left_sp_no_bone = Joint([0,0,0.34],[0.0,pitch_body,0], parent = root, end_joint_of_bone = False, scale = skeleton_scale, color = 'blue',rotation_order = 'zxy',name = 'left_sp_no_bone')
    left_wing_root = Joint([0,0.34,-0.05],[0.0,0,0],parent = left_sp_no_bone, end_joint_of_bone = False, scale = skeleton_scale, color = 'blue',rotation_order = 'zxy',name = 'left_wing_root')
    left_wing_joint1 = Joint([0.05,0.7,0],[0.0,0,0], parent = left_wing_root,  scale = skeleton_scale, color = 'blue',rotation_order = 'zxy',name = 'left_wing_joint1')
    left_wing_joint2 = Joint([0.05,0.9,0],[0.0,0,0], parent = left_wing_joint1,  scale = skeleton_scale, color = 'blue',rotation_order = 'zxy',name = 'left_wing_joint2')
    left_wing_tip = Joint([-0.25,0.6,0],[0.0,0,0], parent =left_wing_joint2, scale = skeleton_scale, color = 'blue',rotation_order = 'zxy',name = 'left_wing_tip')

    # left_sp_no_bone = Joint([0,0,0.5],[0.0,pitch_body,0], parent = root, end_joint_of_bone = False, scale = skeleton_scale, color = 'blue',rotation_order = 'zxy',name = 'left_sp_no_bone')
    # left_wing_root = Joint([-0.,0.5,-0],[0.0,0,0],parent = left_sp_no_bone, end_joint_of_bone = False, scale = skeleton_scale, color = 'blue',rotation_order = 'zxy',name = 'left_wing_root')
    # left_wing_joint1 = Joint([0.05,0.5,0],[0.0,0,0], parent = left_wing_root,  scale = skeleton_scale, color = 'blue',rotation_order = 'zxy',name = 'right_wing_center')
    # left_wing_joint2 = Joint([0.05,0.6,0],[0.0,0,0], parent = left_wing_joint1,  scale = skeleton_scale, color = 'blue',rotation_order = 'zxy',name = 'right_wing_center')
    # left_wing_tip = Joint([-0.25,0.8,0],[0.0,0,0], parent =left_wing_joint2, scale = skeleton_scale, color = 'blue',rotation_order = 'zxy',name = 'left_wing_tip')

    list_joints_pitch_update = [neck,right_sp_no_bone,left_sp_no_bone]

#right_wing_large_thin_y2
    # body = Skin(f'{path_to_mesh}/body_remesh.stl',scale = skin_scale,color = 'lime')
    # right_wing = Skin(f'{path_to_mesh}/right_wing_large_thin_y2.stl',scale = skin_scale,constant_weight = right_wing_root,color = 'crimson')
    # left_wing = Skin(f'{path_to_mesh}/left_wing_large_thin_y2.stl',scale = skin_scale, constant_weight = left_wing_root,color = 'dodgerblue')
    

    body = Skin(f'{path_to_mesh}/body_mos.stl',scale = skin_scale,color = 'lime')
    right_wing = Skin(f'{path_to_mesh}/right_wing.stl',scale = skin_scale,constant_weight = right_wing_root,color = 'crimson')
    left_wing = Skin(f'{path_to_mesh}/left_wing.stl',scale = skin_scale, constant_weight = left_wing_root,color = 'dodgerblue')
    
    # body = Skin(f'{path_to_mesh}/body_bee_v3.stl',scale = skin_scale,color = 'lime')
    # right_wing = Skin(f'{path_to_mesh}/right_wing_v3.stl',scale = skin_scale,constant_weight = right_wing_root,color = 'crimson')
    # left_wing = Skin(f'{path_to_mesh}/left_wing_v3.stl',scale = skin_scale, constant_weight = left_wing_root,color = 'dodgerblue')

    return root,body,right_wing,left_wing,list_joints_pitch_update


def build_skeleton(root,body,right_wing,left_wing,skin_translation = torch.tensor([-0.1/1000-1/1000,0,1/1000], device = 'cuda')):

    joints_of_bone = root.get_and_assign_bones()
    [skin.add_bones(joints_of_bone) for skin in  [body, right_wing,left_wing]]
    [skin.translate_ptcloud_skin(skin_translation) for skin in  [body, right_wing,left_wing]]
    body.calculate_weights_dist(body.bones[0:3])
    right_wing.calculate_weights_dist(right_wing.bones[3:6])
    left_wing.calculate_weights_dist(left_wing.bones[6:])

    # left_wing.calculate_weights_constant()
    joint_list = root.get_list_of_joints()


    skin = torch.vstack([body.ptcloud_skin,right_wing.ptcloud_skin,left_wing.ptcloud_skin])

    weights = torch.block_diag(body.weights, right_wing.weights, left_wing.weights).cuda()
    bones = body.bones
    return joint_list,skin,weights,bones

def transform_pose(points,weights,root_rotation,list_joints_pitch_update,joint_list,bones,translation,right_wing_angles,
                   left_wing_angles,right_wing_angles_joint1,left_wing_angles_joint1,
                   right_wing_twist_joint1,left_wing_twist_joint1,right_wing_angles_joint2,left_wing_angles_joint2,
                   right_wing_twist_joint2,left_wing_twist_joint2, thorax_ang):
    
    joint_list[0].set_local_translation(translation[0],translation[1],translation[2])
    joint_list[0].set_local_rotation(root_rotation[0],root_rotation[1],root_rotation[2])

    joint_list[4].set_local_rotation(torch.tensor(0.0).cuda(),thorax_ang,torch.tensor(0.0).cuda())

    [joint.set_local_rotation(root_rotation[0]*0,-root_rotation[1],root_rotation[0]*0) for joint in list_joints_pitch_update]
    # joint_list[6].set_local_translation(right_wing_location,torch.tensor(-0).cuda(),torch.tensor(0.3/1000).cuda(),)
    joint_list[7].set_local_rotation(right_wing_angles[0],right_wing_angles[1],right_wing_angles[2])
    joint_list[8].set_local_rotation(torch.tensor(0.0).cuda(),right_wing_twist_joint1,right_wing_angles_joint1)
    joint_list[9].set_local_rotation(torch.tensor(0.0).cuda(),right_wing_twist_joint2,right_wing_angles_joint2)

    # joint_list[10].set_local_translation(left_wing_location,torch.tensor(0.).cuda(),torch.tensor(0.3/1000).cuda())
    joint_list[12].set_local_rotation(left_wing_angles[0],left_wing_angles[1],left_wing_angles[2])
    joint_list[13].set_local_rotation(torch.tensor(0.0).cuda(),left_wing_twist_joint1,left_wing_angles_joint1)
    joint_list[14].set_local_rotation(torch.tensor(0.0).cuda(),left_wing_twist_joint2,left_wing_angles_joint2)

    [joint.update_rotation() for joint in joint_list]

    points_homo = torch.column_stack([points,torch.ones(points.shape[0], device = 'cuda')])
    rotated_points = [joint.rotate_to_new_position(weight[:,None],points_homo) for weight,joint in zip(weights.T,bones)]
    skin_rotated = sum(rotated_points)[:,0:3]
    return skin_rotated





    


if __name__ == "__main__":
    path_to_mesh = 'D:/Documents/model_gaussian_splatting/model/mesh'
    skin_translation = torch.tensor([-0.1-1,0,1])*1/1000
    cm_translation = torch.tensor([-0.00134725,  0.00580915,  0.00811845])
    pitch = -25

    root,body,right_wing,left_wing,list_joints_pitch_update = initilize_skeleton_and_skin(path_to_mesh,skeleton_scale=1/1000)
    joint_list,skin,weights,bones = build_skeleton(root,body,right_wing,left_wing)

    


    # update skin position

    # root.set_local_translation(cm_translation*1/1000)
    # root.set_local_rotation([0,pitch,0])
    # [joint.set_local_rotation([0,-pitch,0]) for joint in list_joints_pitch_update]
    # [joint.update_rotation() for joint in joint_list]

    # points_homo = torch.column_stack([skin,torch.ones(skin.shape[0])])
    # rotated_points = [joint.rotate_to_new_position(weight[:,None],points_homo) for weight,joint in zip(weights.T,bones)]
    # skin_rotated = sum(rotated_points)[:,0:3]
















