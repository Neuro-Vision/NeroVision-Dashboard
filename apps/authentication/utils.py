import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import imageio
import nibabel as nib
import pydicom as pdm
import nilearn as nl
import nilearn.plotting as nlplt
import h5py
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import io
import urllib
import base64
import cv2
import os

def handle_uploaded_file(f):  
    with open('apps/static/upload/'+f.name, 'wb+') as destination:  
        for chunk in f.chunks():  
            destination.write(chunk) 
    return f.name      

class Image3dToGIF3d:
    """
    Displaying 3D images in 3d axes.
    Parameters:
        img_dim: shape of cube for resizing.
        figsize: figure size for plotting in inches.
    """
    def __init__(self, 
                 img_dim: tuple = (55, 55, 55),
                 figsize: tuple = (15, 10),
                 binary: bool = False,
                 normalizing: bool = True,
                ):
        """Initialization."""
        self.img_dim = img_dim
        print(img_dim)
        self.figsize = figsize
        self.binary = binary
        self.normalizing = normalizing

    def _explode(self, data: np.ndarray):
        """
        Takes: array and return an array twice as large in each dimension,
        with an extra space between each voxel.
        """
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]),
                            dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    def _expand_coordinates(self, indices: np.ndarray):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z
    
    def _normalize(self, arr: np.ndarray):
        """Normilize image value between 0 and 1."""
        arr_min = np.min(arr)
        return (arr - arr_min) / (np.max(arr) - arr_min)

    
    def _scale_by(self, arr: np.ndarray, factor: int):
        """
        Scale 3d Image to factor.
        Parameters:
            arr: 3d image for scalling.
            factor: factor for scalling.
        """
        mean = np.mean(arr)
        return (arr - mean) * factor + mean
    
    def get_transformed_data(self, data: np.ndarray):
        """Data transformation: normalization, scaling, resizing."""
        if self.binary:
            resized_data = resize(data, self.img_dim, preserve_range=True)
            return np.clip(resized_data.astype(np.uint8), 0, 1).astype(np.float32)
            
        norm_data = np.clip(self._normalize(data)-0.1, 0, 1) ** 0.4
        scaled_data = np.clip(self._scale_by(norm_data, 2) - 0.1, 0, 1)
        resized_data = resize(scaled_data, self.img_dim, preserve_range=True)
        
        return resized_data
    
    def plot_cube(self,
                  cube,
                  title: str = '', 
                  init_angle: int = 0,
                  make_gif: bool = False,
                  path_to_save: str = 'filename.gif'
                 ):
        """
        Plot 3d data.
        Parameters:
            cube: 3d data
            title: title for figure.
            init_angle: angle for image plot (from 0-360).
            make_gif: if True create gif from every 5th frames from 3d image plot.
            path_to_save: path to save GIF file.
            """
        if self.binary:
            facecolors = cm.winter(cube)
            print("binary")
        else:
            if self.normalizing:
                cube = self._normalize(cube)
            facecolors = cm.gist_stern(cube)
            print("not binary")
            
        facecolors[:,:,:,-1] = cube
        facecolors = self._explode(facecolors)
        print("Line 118")
        filled = facecolors[:,:,:,-1] != 0
        x, y, z = self._expand_coordinates(np.indices(np.array(filled.shape) + 1))
        print("Line 121")

        with plt.style.context("dark_background"):
            print("Line 124")
            fig = plt.figure(figsize=self.figsize)
            ax = fig.gca(projection='3d')

            ax.view_init(30, init_angle)
            ax.set_xlim(right = self.img_dim[0] * 2)
            ax.set_ylim(top = self.img_dim[1] * 2)
            ax.set_zlim(top = self.img_dim[2] * 2)
            ax.set_title(title, fontsize=18, y=1.05)
            t1 = time.time()
            ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
            print(time.time() - t1)
            if make_gif:
                print("Line 137")
                images = []
                for angle in tqdm(range(0, 360, 5)):
                    ax.view_init(30, angle)
                    fname = str(angle) + '.png'
                    print(angle)
                    plt.savefig(fname, dpi=120, format='png', bbox_inches='tight')
                    images.append(imageio.imread(fname))
                    #os.remove(fname)
                imageio.mimsave(path_to_save, images)
                plt.close()

            else:
                plt.show()

def create_2d_plots(user_id, segmented, orignal_data) :

        path = "apps/static/2d_files/"

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if(not os.path.exists((BASE_DIR + "/static/2d_files/" + user_id))) :
            os.mkdir(BASE_DIR + "/static/2d_files/" + user_id)
        
        graph_plots = dict()
        start_slice = 60

        # Copy is Important

        edema = segmented.copy()
        edema[edema != 1]= 0

        enhancing = segmented.copy()
        enhancing[enhancing != 2] = 0

        core = segmented.copy()
        core[core != 3] = 0

        context = orignal_data

        plt.switch_backend("AGG")

        #for original image
        plt.figure(figsize=(8,5))
        plt.title("Orignal Image", fontsize=30)
        plt.imshow(cv2.resize(context[:,:,start_slice], (128,128)), cmap="gray", interpolation='none')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_png = buffer.getvalue()
        graph = base64.b64encode(img_png)
        graph = graph.decode('utf-8')
        buffer.close()
        graph_plots['original']=graph
        plt.savefig(path + user_id + '/orignal.png')
        plt.close()



        #for all classes image
        plt.figure(figsize=(8,5))
        plt.title("All Classes", fontsize=30)
        plt.imshow(segmented[:,:, start_slice], cmap="gray")
        print(segmented[:,:, start_slice].shape)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_png = buffer.getvalue()
        graph = base64.b64encode(img_png)
        graph = graph.decode('utf-8')
        buffer.close()
        graph_plots['all']=graph
        plt.savefig(path + user_id + '/all_classes.png')
        plt.close()

        #for edema image
        plt.figure(figsize=(8,5)) 
        plt.title("Edema Image", fontsize=30)
        plt.imshow(edema[:, :, start_slice], cmap="gray")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_png = buffer.getvalue()
        graph = base64.b64encode(img_png)
        graph = graph.decode('utf-8')
        buffer.close()
        graph_plots['edema']=graph
        plt.savefig(path + user_id + '/edema.png')
        plt.close()

        #for core image
        plt.figure(figsize=(8,5))
        plt.title("Core Image", fontsize=30)
        plt.imshow(core[:,:, start_slice], cmap="gray")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_png = buffer.getvalue()
        graph = base64.b64encode(img_png)
        graph = graph.decode('utf-8')
        buffer.close()
        graph_plots['core']=graph
        plt.savefig(path + user_id + '/core.png')
        plt.close()

        #for enhancing image
        plt.figure(figsize=(8,5))
        plt.title("Enhancing Image", fontsize=30)
        plt.imshow(enhancing[:,:, start_slice], cmap="gray")
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_png = buffer.getvalue()
        graph = base64.b64encode(img_png)
        graph = graph.decode('utf-8')
        buffer.close()
        graph_plots['enhancing']=graph
        plt.savefig(path + user_id + '/enhancing.png')
        plt.close()

        return graph_plots
    
def find_tumor_location(segmented):

    max_axial = 0 # saves max non zero value
    max_slice = 0 # saves max non zero slice number
    output = {}

    for i in range(0,155):
        total_tumor_density =  np.count_nonzero(segmented[:, :, i])
        if total_tumor_density > max_axial :
            max_axial = total_tumor_density
            max_slice = i

    middle_range = segmented.shape[2] * 0.2
    rest_range = segmented.shape[2] * 0.4

    if max_slice <  rest_range : 
        output['Axial']  = "bottom"
    elif max_slice > rest_range and max_slice < (rest_range + middle_range) :
        output['Axial']  = "middle"
    else :
        output['Axial']  = "top"


    # for Coronal (Front to Back)
    max_coronal = 0
    max_slice = 0

    for i in range(0,240) :
        total_tumor_density =  np.count_nonzero(np.rot90(segmented[:, i, :]))

    if total_tumor_density > max_coronal :
        max_coronal = total_tumor_density
        max_slice = i


    middle_range = segmented.shape[1] * 0.2
    rest_range = segmented.shape[1] * 0.4

    if max_slice <  rest_range : 
        output['Coronal']  = "Front"
    elif max_slice > rest_range and max_slice < (rest_range + middle_range) :
        output['Coronal']  = "Middle"
    else :
        output['Coronal']  = "Back"


    # For Saggital (Right to Left)
    max_saggital = 0
    max_slice = 0

    for i in range(0,240) :
        total_tumor_density =  np.count_nonzero(np.rot90(segmented[i, :, :]))
    if total_tumor_density > max_saggital :
        max_saggital = total_tumor_density
        max_slice = i

    middle_range = segmented.shape[0] * 0.2
    rest_range = segmented.shape[0] * 0.4

    if max_slice <  rest_range : 
        output['Saggital']  = "right"
    elif max_slice > rest_range and max_slice < (rest_range + middle_range) :
        output['Saggital']  = "middle"
    else :
        output['Saggital']  = "left"

    return output

def occupancy(label_array, image_data) :
    density = dict()

    density["tumor_density"] = round((np.count_nonzero(label_array) / np.count_nonzero(image_data)) * 100, 2)

    enhancing = label_array[label_array == 2]
    density["enhancing"] = round((np.count_nonzero(enhancing) / np.count_nonzero(image_data)) * 100, 2)

    edema = label_array[label_array == 1]
    density["edema"] = round((np.count_nonzero(edema) / np.count_nonzero(image_data)) * 100, 2)

    core = label_array[label_array == 3]
    density["core"] = round((np.count_nonzero(core) / np.count_nonzero(image_data)) * 100, 2)

    return density