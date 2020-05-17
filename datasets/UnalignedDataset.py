from torch.utils.data.dataset import Dataset
import os
import random
from PIL import Image
import glob

class UnalignedDataset(Dataset):
    
    def __init__(self, root, folder_names, limit=-1, transform=None):
        super(UnalignedDataset, self).__init__()
        
        self.root = root
        self.transform = transform
        self.folder_names = folder_names
        self.A_paths = glob.glob(os.path.join(root, folder_names[0]) + '/**/*.jpg', recursive=True)
        self.B_paths = glob.glob(os.path.join(root, folder_names[1]) + '/**/*.jpg', recursive=True)
        if limit == -1:
            self.A_size = len(self.A_paths)
            self.B_size = len(self.B_paths)
        else:
            self.A_size = min(len(self.A_paths), limit)
            self.B_size = min(len(self.B_paths), limit)
        self.A_paths = self.A_paths[:self.A_size]
        self.B_paths = self.B_paths[:self.B_size]
        #print(self.A_paths)
        
    def __len__(self):
        return max(self.A_size, self.B_size)
    
    def __getitem__(self, idx):
        idx_A = idx % self.A_size
        idx_B = random.randint(0, self.B_size - 1)
        
        #image_A = Image.open(os.path.join(self.root, self.folder_names[0], self.A_paths[idx_A])).convert("RGB")
        image_A = Image.open(self.A_paths[idx_A]).convert("RGB")
        #image_B = Image.open(os.path.join(self.root, self.folder_names[1], self.B_paths[idx_B])).convert("RGB")
        image_B = Image.open(self.B_paths[idx_B]).convert("RGB")
        if self.transform is not None:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            
        return {"A": image_A, "B": image_B}