import os
import json

from torch.utils.data import Dataset
from torchvision import transforms as T

from .ray_utils import *

class BlenderDataset(Dataset):
    """
        datadir: data directory
        split: #TODO
        downsample: Shrink the size of the input image?
        is_stack: #TODO: ??
        N_vis: #TODO: ??
    """
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):
        
        # What is this parameter?
        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(800/downsample), int(800/downsample))
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        
        #TODO: Implement this two functions 
        self.read_meta()
        self.define_proj_mat()

    #TODO: functionality??
    def read_meta(self):
        
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)
        
        w, h = self.img_wh

        # original focal length
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  
        # modify focal length to match size self.img_wh
        self.focal *= self.img_wh[0] / 800

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal, self.focal]) # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal, 0, w / 2], [0, self.focal, h / 2], [0, 0, 1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.downsample = 1.0

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'): #img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            

    #TODO: Do we really need this function?
    def define_transforms(self):
        self.transform = T.ToTensor()
