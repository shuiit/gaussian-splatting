import cv2
import numpy as np
from PIL import Image
from Camera import Camera
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
class Frame(Camera):
    def __init__(self,path,frame,frame_num, camera_path = None,frames_dict= None):
        """
        Initialize a Frame object by loading the corresponding image and voxels and processing pixel data.

        Args:
            path (str): The directory path to the images.
            im_name (str): The name of the image file.
            idx (int): Unique identifier for the image.
        """
      

        self.path = path
        

        super().__init__()
        if frames_dict:
            self.image_id = list(frames_dict[frame][0].keys())[frame_num]
            self.load_from_dict(frames_dict,frame,self.image_id,frame_num)  
            self.image_name = frames_dict[frame][0][list(frames_dict[frame][0].keys())[frame_num]]['name'].split('.')[0]
            
            self.frame = frame
        else: 
            self.image_id = frame_num
            self.image_name = frame
            self.load_from_text(camera_path,int(frame.split('CAM')[1])-1)
            self.load_and_crop_image()
            
            self.frame = int(frame.split('CAM')[0].split('P')[1])
        self.im = Image.open(f'{self.path}/images/{self.image_name}.jpg').convert('L')
        self.camera_number = frame_num
        y,x = np.where((np.array(self.im) < 250) )
        self.pixels = np.vstack([y,x]).T
        self.cm = np.mean(self.pixels,0)

    
    def load_image(self):
   
        im = scipy.io.loadmat(f'{self.path}images/{self.image_name.split(".jpg")[0]}.mat')['im']
        bg = np.array((scipy.io.loadmat(f'{self.path}images/bg.mat')['bg']//255).astype(np.uint16))
        white_bg = bg*0 + 255
        image = Image.fromarray(np.array((im * 255).astype(np.uint8)), mode="L")
        return image,white_bg,bg
    
    def erode_and_add_bf(self,image,bg,kernel = np.ones((2, 2), np.uint8)):
        
        eroded_image = np.array(cv2.erode(np.array(image), kernel))
        image_with_bg = np.array(bg)
        image_with_bg[eroded_image > 0] = eroded_image[eroded_image > 0]
        return Image.fromarray(image_with_bg),Image.fromarray(eroded_image)
    
    def calculate_bounding_box(self,cm,delta_xy):
        top_left = np.array([max(0,cm[0] - delta_xy), max(0,cm[1]-delta_xy)])
        bottom_right =  np.array([min(self.image_size[0],cm[0] + delta_xy), min(self.image_size[1],cm[1] + delta_xy)])

        if (bottom_right != 0).any():
            top_left = np.minimum(bottom_right - delta_xy*2,top_left) 
        return np.hstack((top_left,bottom_right))
        
        # bottom_right = np.array([min(self.image_size[0],bottom_right[0]),min(self.image_size[1],bottom_right[1])])


        # self.bounding_box = [max(0,cm[1] - delta_xy), max(0,cm[0]-delta_xy), max(0,cm[1] - delta_xy) + delta_xy*2 , max(0,cm[0]-delta_xy) + delta_xy*2] # [top left, bottom right]


    def load_and_crop_image(self,delta_xy = 80):
        """
        Crop the image around the mean pixel coordinates.

        Args:
            delta_xy (int, optional): The half-width of the cropping area. Default is 80.
        """
        image,white_bg,bg = self.load_image()
        image,image_no_bg = self.erode_and_add_bf(image,white_bg)

        pixels = np.vstack(np.where(np.array(image_no_bg) > 0)).T
        cm = np.mean(pixels,0).astype(int)
        self.bounding_box = self.calculate_bounding_box(cm,delta_xy)
        self.top_left = self.bounding_box[0:2] # to-left
        self.crop_size = delta_xy*2
        self.image = image.crop([self.bounding_box[1],self.bounding_box[0],self.bounding_box[3],self.bounding_box[2]])

        self.camera_calibration_crop(self.top_left, self.image.size) 

        self.bg  = bg[self.bounding_box[0]:self.bounding_box[2],self.bounding_box[1]:self.bounding_box[3]]
        self.image_no_bg = image_no_bg.crop([self.bounding_box[1],self.bounding_box[0],self.bounding_box[3],self.bounding_box[2]])
        self.image_size = self.image.size
        image_mask = (np.array(self.image)  == 255 )
        self.image_with_bg = self.bg*image_mask + np.array(self.image_no_bg)
        self.pixels = pixels - self.top_left
        self.cm = np.mean(self.pixels,0)
        self.save_croped_images()



    # def save_croped_images(self):
       
    #     image_rgb = self.image_with_bg.convert('RGB')

    #     im_np = np.array(image_rgb)
    #     idx = np.where(im_np[:,:,0] == 0)
    #     im_np[idx[0],idx[1],:] = 255
    #     image = Image.fromarray(im_np)
    #     image.save(f'{self.path}/images/{self.image_name}.jpg', format='JPEG', subsampling=0, quality=100)


    def map_3d_2d(self,points_3d, use_zbuff = True):
        """
        Map 3D voxel positions to 2D pixel coordinates and store relevant data.

        Args:
            croped_image (bool, optional): Whether to use cropped image pixels. Default is False.
        """
        # voxels,pixels_of_voxels = self.z_buffer(croped_camera_matrix = croped_image) if use_zbuff == True else self.map_no_zbuff(croped_camera_matrix = croped_image)
        pixels_of_voxels = self.project_on_image(points_3d)

        original_projected_pixels = np.vstack((pixels_of_voxels,np.fliplr(self.pixels ))) # project pixels
        [non_intersect_pixels,cnt] = np.unique(original_projected_pixels,axis = 0,return_counts=True) # identify non intersecting pixels
        non_intersect_pixels = non_intersect_pixels[cnt == 1,:] 

        all_pixels = np.vstack((pixels_of_voxels, non_intersect_pixels)) if use_zbuff == True else pixels_of_voxels
        all_3d_idx = np.full(all_pixels.shape[0], -1)
        all_3d_idx[0:points_3d.shape[0]] = points_3d[:,3]

        self.pixel_with_idx = np.column_stack((all_pixels, all_3d_idx))
        self.voxels_with_idx = np.column_stack((points_3d,np.full(pixels_of_voxels.shape[0],self.image_id),np.arange(pixels_of_voxels.shape[0])))

        # determine the color of every pixel that has a mapping, 
        idx = self.pixel_with_idx[:,2] != -1
        pixels = self.pixel_with_idx[idx,0:3].astype(int)
        self.color_of_pixel =  np.array(np.array(self.image))[pixels[:,1],pixels[:,0]]

    
    def map_no_zbuff(self,croped_camera_matrix = False):
        
        voxels_cam = np.matmul(self.world_to_cam, self.points_in_ew_frame_homo.T).T
        projected = self.project_on_image(self.points_in_ew_frame_homo,croped_camera_matrix)
        pxls = np.round(projected) 
        
        idx_sorted_by_z = voxels_cam[:,2].argsort()
        voxels_sorted_by_z = self.points_in_ew_frame[idx_sorted_by_z,:]
        return voxels_sorted_by_z,pxls[idx_sorted_by_z,:]

    def z_buffer(self,croped_camera_matrix = False):
        """
        Compute the z-buffer for 3D points, projecting them onto the 2D image plane.

        Args:
            croped_camera_matrix (bool, optional): Whether to use a cropped camera matrix. Default is False.

        Returns:
            tuple: A tuple containing sorted voxel positions and pixel coordinates.
        """
        voxels_cam = np.matmul(self.world_to_cam, self.points_in_ew_frame_homo.T).T
        projected = self.project_on_image(self.points_in_ew_frame_homo,croped_camera_matrix)
        pxls = np.round(projected) 
        idx_sorted_by_z = voxels_cam[:,2].argsort()
        voxels_sorted_by_z = self.points_in_ew_frame[idx_sorted_by_z,:]
        [pixels,idx] = np.unique(pxls[idx_sorted_by_z,:], axis=0,return_index=True)
        return voxels_sorted_by_z[idx,:],pixels
    

    

    def save_croped_images(self):
        image_rgb = self.image.convert('RGB')
        im_np = np.array(image_rgb)
        idx = np.where(im_np[:,:,0] == 0)
        im_np[idx[0],idx[1],0] = 255
        image = Image.fromarray(im_np)
        image.save(f'{self.path}/images/{self.image_name}.jpg', format='JPEG', subsampling=0, quality=100)


    
    def generate_base_image(self):

        return {'id' :self.image_id, 'qvec' : self.qvec.copy(), 'tvec' : self.t.T[0].copy(),
                            'camera_id': self.camera_number, 'name' : f'{self.image_name}.jpg'
                            }


    

    def intersection_per_cam(self,ptcloud_volume):    
        ptsv = self.homogenize_coordinate(ptcloud_volume)
        pt2dv = self.project_on_image(ptsv)
        pt2dv = np.fliplr(pt2dv)
        pts_for_unique = np.vstack((self.pixels,np.unique(pt2dv.astype(int),axis = 0)))
        v,cnt = np.unique(pts_for_unique,return_counts = True,axis = 0)
        projected_on_image = v[cnt > 1]
        all_indices = np.vstack(np.argwhere(np.all(pt2dv.astype(int) == repeated_group, axis=1)) for repeated_group in projected_on_image)
        return ptcloud_volume[all_indices[:,0]]
    

    


    





