import numpy as np
from model.Bone import Bone
import torch

class Joint:
    def __init__(self, translation, rotation,  parent = None, end_joint_of_bone = True, rotation_order = 'zyx', scale = 1,color = 'green',name = None):
        self.child = []
        self.parent = parent
        self.neme = name
        rotation = torch.tensor(rotation,device='cuda')
        self.local_angles = rotation
        self.local_translation = torch.tensor(translation,device='cuda')*scale
        self.rotation_order = list(rotation_order)
        self.local_rotation = self.rotation_matrix(rotation[0],rotation[1],rotation[2])
        self.translation_from_parent = torch.tensor(translation,device='cuda')*scale
        self.local_transformation = self.transformation_matrix()
        self.global_transformation = self.get_global_transformation(rest_bind = True)
        self.end_joint_of_bone = end_joint_of_bone
        self.get_global_point()
        self.bone = None
        self.color = color
        self.scale = scale
        self.update_child()
        # self.parent.child.append(self) if self.parent != None else []
        


    def update_child(self):
        if self.parent == None:
            return
        self.parent.update_child()
        
        if self not in self.parent.child:
            self.parent.child.append(self)
       

    def get_and_assign_bones(self, visited = None):
        visited = visited or set()
            
        if self in visited:
            return []
        visited.add(self)
        if self.end_joint_of_bone:
            self.parent.bone = Bone(self.parent,self )
        bones = [self.parent] if self.end_joint_of_bone else []
        for child in self.child: 
            bones += child.get_and_assign_bones(visited)
        return bones
    

    def get_list_of_joints(self, visited = None,joints = None):
        visited = visited or set()
        joints = joints or []
        joints.append(self)
        visited.add(self)

        for child in self.child: 
            if child not in visited:
                child.get_list_of_joints(visited,joints)
        return joints




    def set_local_rotation(self,yaw,pitch,roll):
        # angles (z,y,x) (yaw, pitch, roll)
        self.local_rotation = self.rotation_matrix(yaw,pitch,roll)
        self.local_transformation = self.transformation_matrix()

    
    def set_local_translation(self,x,y,z):
        self.local_translation = torch.stack([x,y,z])
        self.local_transformation = self.transformation_matrix()

    def set_local_transformation(self):
        self.local_transformation = self.transformation_matrix()

    def get_global_transformation(self, rest_bind = False):
        if self.parent == None:
            return self.local_transformation
        self.global_transformation = self.parent.get_global_transformation(rest_bind = rest_bind)
        self.global_transformation = torch.matmul(self.global_transformation.float(),self.local_transformation)
        if rest_bind == True:
            self.bind_transformation = self.global_transformation
        return self.global_transformation
    
    def get_global_point(self,point = torch.tensor([0,0,0,1.0],device='cuda')):
        if (point == torch.tensor([0,0,0,1],device='cuda')).all():
            self.global_origin = torch.matmul(self.global_transformation,point)[0:3]
        return torch.matmul(self.global_transformation,point)[0:3]
        

    def rotate_to_new_position(self,weight,points_homo):
        transformation_rest = torch.inverse(self.bind_transformation)
        rotated_points = torch.matmul(transformation_rest,points_homo.T)
        return weight*torch.matmul(self.global_transformation,rotated_points).T

    def update_rotation(self):
        self.get_global_transformation()
        self.get_global_point()


    def rotate_normal_to_new_position(self,weight,normal):
        transformation_rest = np.linalg.inv(self.bind_transformation)
        transformation_rest_to_global = np.dot(self.global_transformation,transformation_rest).T
        rotated_points_inv = np.linalg.inv(transformation_rest_to_global)
        return weight*np.dot(rotated_points_inv,normal.T).T





    def rotation_matrix(self,yaw,pitch,roll):
        
        roll = roll[None]*torch.pi/180
        pitch = pitch[None]*torch.pi/180
        yaw = yaw[None]*torch.pi/180

        tensor_0 = torch.zeros(1,device='cuda')
        tensor_1 = torch.ones(1,device='cuda')

        mat = {}
        mat['x'] = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                    torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

        mat['y'] = torch.stack([
                    torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)

        mat['z'] =torch.stack([
                    torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                    torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)
        rotation_matrix = mat[self.rotation_order[0]] @ mat[self.rotation_order[1]]  @ mat[self.rotation_order[2]] 
        return rotation_matrix
    
    def transformation_matrix(self):
        return  torch.vstack((torch.column_stack((self.local_rotation,self.local_translation)),torch.tensor([0,0,0,1],device='cuda')))
