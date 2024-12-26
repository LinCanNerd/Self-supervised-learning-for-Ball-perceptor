import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from copy import deepcopy
from collections import namedtuple
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import os
import random
from math import floor
import cv2
from skimage import io
Box = namedtuple("Box", ["x_min", "x_max", "y_min", "y_max"])
ATTEMPTS = 50 # How many times it attempts to get a patch
DEBUG = False

class BallDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        
        self.dataframe["contains_ball"] = self.dataframe["class"] == "ball" 

        def process_group(group):
            if group["contains_ball"].any():  # If there's a ball in the group
                return group[group["contains_ball"]]  # Keep only the ball annotations
            else:
                return group.sample(n=1) 

        self.dataframe = self.dataframe.groupby("filename").apply(process_group).reset_index(drop=True)
        
        self.img_dir = img_dir
        self.transform = transform
        self.scales = [32,64,128]

        '''
            P(showBall | containBall)  P(containBall) = P(showBall and containsBall) = 50
            P(showBall | containBall) = 50/P(containBall)
        '''
        self.pShowBall = max(min(0.5 / self.dataframe["contains_ball"].mean(), 1.), 0.)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        id, width, height, xmin, xmax, ymin, ymax, contains_ball = self.dataframe.loc[idx, ["filename", "width", "height", "xmin", "xmax", "ymin", "ymax", "contains_ball"]]        # id, contains_ball, bbx
        
        bbx = Box(xmin, xmax, ymin, ymax)

        img_path = os.path.join(self.img_dir, id)
        
        # Read the image
        image = io.imread(img_path)
        image = Image.fromarray(image)

        if DEBUG and contains_ball:
            imgDraw = ImageDraw.Draw(image)
            imgDraw.rectangle([bbx.x_min, bbx.y_min, bbx.x_max, bbx.y_max], outline="red", width=3)

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)

        # Estrai patch con palla - Estrai patch senza palla

        scaleBuffer = deepcopy(self.scales)

        while True:
            scale_idx = random.randint(0, len(scaleBuffer) - 1)
            currentScale = scaleBuffer.pop(scale_idx)
            lastChance = len(scaleBuffer) == 1
            
            if contains_ball:
                # randomly pick either ball or not ball patch
                if random.random() < self.pShowBall:
                    res = self.showBall(currentScale, bbx, image)
                    if res["is_valid"]:
                        return {
                            "original_img": image,
                            "patch": res["patch"],
                            "bbx": res["scaledBbx"],
                            "x": res['xyr']['x'],
                            "y": res['xyr']['y'],
                            "r": res['xyr']['r'],
                            "contains_ball": True,
                        }
                    elif lastChance:
                        res = self.hideBall(currentScale, bbx, image)
                        if res["is_valid"]:
                            return {
                                "original_img": image,
                                "patch": res["patch"],
                                "bbx": None,
                                'x':None,
                                'y':None,
                                'r':None,
                                "contains_ball": False
                            }
                else:
                    for _ in range(ATTEMPTS):
                        res = self.hideBall(currentScale, bbx, image)
                        if res["is_valid"] and not self.isPatchGreen(res['patch']):
                            return {
                                "original_img": image,
                                "patch": res["patch"],
                                "bbx": None,
                                'x':None,
                                'y':None,
                                'r':None,
                                "contains_ball": False
                            }
            else:
                for _ in range(ATTEMPTS):
                    # return no ball patch
                    x = random.randint(0, width - currentScale - 1)
                    y = random.randint(0, height - currentScale - 1)
                    patch = image[:, y:y+currentScale, x:x+currentScale]
                    
                    # Resize the patch to 32x32
                    patch = F.interpolate(patch.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)

                    if not self.isPatchGreen(patch, edge_threshold=0.05):
                        return {
                            "original_img": image,
                            "patch": patch,
                            "bbx": None,
                            'x':None,
                            'y':None,
                            'r':None,
                            "contains_ball": False
                        }
                return {
                            "original_img": image,
                            "patch": patch,
                            "bbx": None,
                            'x':None,
                            'y':None,
                            'r':None,
                            "contains_ball": False
                        }

    def showBall(self, scale, bbx, image, minPixels = 5):
        
        targetPatchSize = 32
        
        scaleRatio = targetPatchSize / scale
        assert scaleRatio <= 1.

        bbxWidth = bbx.x_max - bbx.x_min
        bbxHeight = bbx.y_max - bbx.y_min

        _, height, width = image.shape 

        scaledBbxWidth = floor(bbxWidth * scaleRatio)
        scaledBbxHeight = floor(bbxHeight * scaleRatio)

        if scaledBbxWidth < minPixels or scaledBbxHeight < minPixels or scaledBbxHeight > targetPatchSize or scaledBbxWidth > targetPatchSize:
            return {"is_valid": False, "patch": None, "scaledBbx": None} 
        
        bbxCenterx = bbx.x_min + bbxWidth//2
        bbxCentery = bbx.y_min + bbxHeight//2

        #print(f"{bbxCenterx=} {bbxCentery=}")

        scaledBbx = Box(
            x_min=bbx.x_min * scaleRatio,
            x_max=bbx.x_max * scaleRatio,
            y_min=bbx.y_min * scaleRatio,
            y_max=bbx.y_max * scaleRatio,
        )

        # Randomly pick top left corner among the feasible values
        # center - scale <= rand <= center
        minRandx = max(0, bbxCenterx - scale)
        maxRandx = min(bbxCenterx, width - scale) 

        xPatch = random.randint(minRandx, maxRandx)

        minRandy = max(0, bbxCentery - scale)
        maxRandy = min(bbxCentery, height - scale) 

        yPatch = random.randint(minRandy, maxRandy)


        # Get image
        patch = image[:, yPatch:yPatch+scale, xPatch:xPatch+scale]
        assert patch.shape[1] == patch.shape[2], f"MEEK! {patch.shape=} given {scale=} and the top left corner ({yPatch}, {xPatch})" 

        # Resize the patch to 32x32
        patch = F.interpolate(patch.unsqueeze(0), size=(targetPatchSize, targetPatchSize), mode='bilinear', align_corners=False).squeeze(0)

        #ADD XYR
        x = (bbxCenterx - xPatch) * (32/scale)
        y =(bbxCentery - yPatch) * (32/scale)
        r = ((bbx.x_max - bbx.x_min) / 2) * (32 / scale)


        return {"is_valid": True, "patch": patch, "scaledBbx": scaledBbx, 'xyr':{'x':x,'y':y,'r':r}} 
    
    def showBall2(self, scale, bbx, image, minPixels = 5):
        
        targetPatchSize = 32
        
        scaleRatio = targetPatchSize / scale
        assert scaleRatio <= 1.

        bbxWidth = bbx.x_max - bbx.x_min
        bbxHeight = bbx.y_max - bbx.y_min

        _, height, width = image.shape 

        scaledBbxWidth = floor(bbxWidth * scaleRatio)
        scaledBbxHeight = floor(bbxHeight * scaleRatio)

        if scaledBbxWidth < minPixels or scaledBbxHeight < minPixels or scaledBbxHeight > targetPatchSize or scaledBbxWidth > targetPatchSize:
            return {"is_valid": False, "patch": None, "scaledBbx": None} 
        
        bbxCenterx = bbx.x_min + bbxWidth//2
        bbxCentery = bbx.y_min + bbxHeight//2

        #print(f"{bbxCenterx=} {bbxCentery=}")

        scaledBbx = Box(
            x_min=bbx.x_min * scaleRatio,
            x_max=bbx.x_max * scaleRatio,
            y_min=bbx.y_min * scaleRatio,
            y_max=bbx.y_max * scaleRatio,
        )

        mean_x = bbxCenterx
        mean_y = bbxCentery

        minRandx = max(0, bbxCenterx - scale)
        maxRandx = min(bbxCenterx, width - scale) 
        minRandy = max(0, bbxCentery - scale)
        maxRandy = min(bbxCentery, height - scale) 

        # Standard deviation as a fraction of the scale
        sigma = scale / 3.0  

        # Generate Gaussian random points and clip them to valid ranges
        xPatch = int(np.clip(np.random.normal(mean_x, sigma), minRandx, maxRandx))
        yPatch = int(np.clip(np.random.normal(mean_y, sigma), minRandy, maxRandy))

        # Get image
        patch = image[:, yPatch:yPatch+scale, xPatch:xPatch+scale]
        assert patch.shape[1] == patch.shape[2], f"MEEK! {patch.shape=} given {scale=} and the top left corner ({yPatch}, {xPatch})" 

        # Resize the patch to 32x32
        patch = F.interpolate(patch.unsqueeze(0), size=(targetPatchSize, targetPatchSize), mode='bilinear', align_corners=False).squeeze(0)

        return {"is_valid": True, "patch": patch, "scaledBbx": scaledBbx} 
    

    def hideBall(self, scale, bbx, image):
        # Must contain boxes
        regions = []

        _, height, width = image.shape 

        # Top
        if bbx.y_min > scale :
            regions.append(Box(0, width - scale, 0, bbx.y_min - 1 - scale))
        # Bottom
        if bbx.y_max + scale < height:
            regions.append(Box(0, width - scale, bbx.y_max + 1, height - scale))
        # Left
        if bbx.x_min > scale:
            regions.append(Box(0, bbx.x_min - 1 - scale, 0, height - scale))
        # Right
        if bbx.x_max + scale < width:
            regions.append(Box(bbx.x_max + 1, width - scale, 0, height - scale ))

        if len(regions) == 0:
            return {"is_valid": False, "patch": None} 

        region = random.choice(regions)

        xPatch = random.randint(region.x_min, region.x_max)
        yPatch = random.randint(region.y_min, region.y_max)
        
        patch = image[:, yPatch:yPatch+scale, xPatch:xPatch+scale]
        # Resize the patch to 32x32
        patch = F.interpolate(patch.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)

        return {"is_valid": True, "patch": patch} 

    def isPatchGreen(self, patch, edge_threshold = 0.01):
        # Ensure the patch is in uint8 format
        patch = patch.permute(1, 2, 0).numpy()
        patch = (patch * 255).clip(0, 255).astype("uint8")
        
        # Apply Canny edge detection
        edges = cv2.Canny(patch, threshold1=50, threshold2=200)
        
        # Count the number of edge pixels
        edge_pixel_count = np.sum(edges > 0)
        total_pixels = patch.shape[0] * patch.shape[1]

        # If the proportion of edge pixels is below the threshold, return True
        return edge_pixel_count / total_pixels < edge_threshold

