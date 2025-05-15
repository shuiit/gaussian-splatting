
import numpy as np
import scipy.io
import math 
import Utils

class Camera():
    def __init__(self):
        """Initialize the camera object with parameters, intrinsic and extrinsic matrices."""

    def load_from_text(self,path,camera_number, cam = False, image_size = [800,1280]):
        """Initialize the camera object with parameters, intrinsic and extrinsic matrices."""

        self.path = path
        cam,get_cam_mat = (scipy.io.loadmat(f'{path}/camera_KRX0.mat'),camera_number) if cam == False else (cam,0)
        self.K = cam['camera'][0:3,0:3,get_cam_mat]
        self.R = cam['camera'][0:3,3:6,get_cam_mat]
        self.X0 = cam['camera'][0:3,6:7,get_cam_mat]
        self.t = -np.matmul(self.R,self.X0)
        self.camera_number = camera_number + 1
        self.K[1,2]  = self.K[1,2] - 1  
        self.K[0,2]  = self.K[0,2] - 1
        self.fx = self.K[0,0]
        self.fy = self.K[1,1]
        self.cx = self.K[0,2]
        self.cy = self.K[1,2]
        self.znear = 0.000000000001
        self.zfar = 1000
        self.world_to_cam = np.hstack([self.R,self.t])
        self.camera_matrix = np.hstack([np.matmul(self.K,self.R),np.matmul(self.K,self.t)])
        self.image_size = image_size
        self.rotmat2qvec()
        self.getProjectionMatrix(image_size)

    def load_from_dict(self,frames_dict,frame,cam_id,cam_num):
        self.cam_id = cam_num
        image = frames_dict[frame][0]
        camera = frames_dict[frame][1]

        qvec = image[cam_id]['qvec']
        params = camera[cam_id]['params']
        self.R = Utils.qvec2rotmat(qvec)
        self.t = image[cam_id]['tvec'][:,np.newaxis]
        self.X0 = -np.matmul(np.linalg.inv(self.R),self.t)

        self.fx = params[0]
        self.fy = params[1]
        self.cx = params[2] +1
        self.cy = params[3] +1
        self.K = np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]])
        width = camera[cam_id]['width']
        hight = camera[cam_id]['height']
        self.image_size = [hight,width]
        self.znear = 0.000000001
        self.zfar = 100
        self.world_to_cam = np.hstack([self.R,self.t])
        self.camera_matrix = np.hstack([np.matmul(self.K,self.R),np.matmul(self.K,self.t)])
        self.rotmat2qvec()
        self.getProjectionMatrix(self.image_size)

        

    def camera_calibration_crop(self,crop_pixels, image_size):
        """updates the intrinsic K matrix for croped images

        Args:
            crop_pixels (np array): loaction of top left pixel 
        """
        self.image_size = image_size
        self.K = self.K.copy()
        self.K[0,2] = self.K[0,2] - crop_pixels[1]
        self.K[1,2] = self.K[1,2] - (crop_pixels[0])

        self.fx = self.K[0,0]
        self.fy = self.K[1,1]
        self.cx = self.K[0,2]
        self.cy = self.K[1,2]
        self.camera_matrix = np.hstack([np.matmul(self.K,self.R),np.matmul(self.K,self.t)])
        self.rotmat2qvec()
        self.getProjectionMatrix(self.image_size)
        
    def camera_center_to_pixel_ray(self,pixels):
        
        ray_ndc = [(pixels[1] - self.cx)/self.fx,(pixels[0] - self.cy)/self.fy,1]
        return  np.dot(self.R.T,ray_ndc) + self.X0.T
        
    def rotation_matrix_from_vectors(self,vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    
    def project_on_image(self,points):
        """project 3d points on 2d image

        Args:
            points (np array): 3d points in camera axes
            cam_matrix (np array): camera calibration matrix [K[R|T]]

        Returns:
            pixels (x/u,y/v): pixels in image plane
        """
        points_2d = np.matmul(self.camera_matrix,points.T)
        points_2d = (points_2d[:-1, :] / points_2d[-1, :]).T
        return points_2d
    

    def rotate_world_to_cam(self,points):
        """Rotate points from world coordinates to camera coordinates.

        Args:
        points (ndarray): Array of points in world coordinates (shape: [n, 3]).

        Returns:
            ndarray: Array of points in camera coordinates (shape: [n, 3]).
        """
        return np.matmul(self.world_to_cam , points).T
    
    
    def rotmat2qvec(self):
        """Convert a rotation matrix to a quaternion vector
        Taken from colmap loader (gaussian-splatting)-- probably taken from colmap 
        """
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = self.R.flat
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        self.qvec = np.round(np.array(qvec),7)
    
    def focal2fov(self,focal, pixels):
        return 2*math.atan(pixels/(2*focal))
    

    
    def getProjectionMatrix(self,im_size):
        """Compute the projection matrix for a given image size.

        Args:
            im_size (list): Size of the image [height, width].
        """
        fovy = self.focal2fov(self.fy, im_size[1])
        fovx = self.focal2fov(self.fx, im_size[0])
        tanHalfFovY = math.tan((fovy / 2))
        tanHalfFovX = math.tan((fovx / 2))

        top = tanHalfFovY * self.znear
        bottom = -top
        right = tanHalfFovX * self.znear
        left = -right

        P = np.zeros((4, 4))

        z_sign = 1.0

        P[0, 0] = 2.0 * self.znear / (right - left)
        P[1, 1] = 2.0 * self.znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.zfar / (self.zfar - self.znear)
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        P[0, 2] = (right + left) / (right - left) + (2 * self.cx / im_size[0]) - 1
        P[1, 2] = (top + bottom) / (top - bottom) + (2 * self.cy / im_size[1]) - 1
        self.projection = P
        world_view_transform = np.vstack((self.world_to_cam,[0,0,0,1]))
        self.full_proj_transform = np.matmul(P,world_view_transform)  # Shape: (1, N, K)

    
    def proj_screen(self,pixel,s):
        """Translate pixel coordinates to normalized device coordinates (NDC).

        Args:
            pixel (np array): Pixel coordinates.
            s (float): Scaling factor.

        Returns:
            np array: Translated coordinates in NDC.
        """
        # translate to ndc space
        return ((pixel + 1)*s-1)*0.5
    

    

    def project_with_proj_mat(self, points):
        """
        Projects 3D points in world coordinates onto the 2D image plane 
        using the precomputed full projection matrix.

        Args:
            points (np.array): Array of 3D points in world coordinates (shape: [n, 3]).

        Returns:
            np.array: Array of 2D pixel coordinates in normalized device coordinates.
        """
        xyz_homo  = self.homogenize_coordinate(points)


        p_proj = np.matmul(self.full_proj_transform,xyz_homo.T).T
        p_proj = p_proj/p_proj[:,3:]
        pixels = self.proj_screen(p_proj,self.image_size[0])
        return pixels
    

    def homogenize_coordinate(self,points):
        """
        Converts 3D points to homogeneous coordinates by adding a fourth 
        dimension with value 1 to each point.

        Args:
            points (np.array): Array of 3D points (shape: [n, 3]).

        Returns:
            np.array: Array of points in homogeneous coordinates (shape: [4, n]).
        """
        return np.column_stack((points,np.ones((points.shape[0],1))))




    def cams_for_gs(self):

        params = self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]
        return {'id' : self.camera_number, 'model' : 'PINHOLE',
                                            'width' : self.image.size[0], 'height' : self.image.size[1],
                                            'params' : params}

