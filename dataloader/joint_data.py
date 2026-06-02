import torch
import pathlib
import cv2
import kornia
from typing import Callable, Tuple

class JointTrainData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    Supports: .bmp, .jpg, .tiff, .png image formats
    """
    def __init__(self, 
                 ir_folder: pathlib.Path, 
                 vi_folder: pathlib.Path, 
                 ir_mask: pathlib.Path, 
                 vi_mask: pathlib.Path,
                 crop: Callable = lambda x: x):
        super(JointTrainData, self).__init__()
        
        # Supported image extensions
        self.supported_ext = ['.bmp', '.jpg', '.jpeg', '.tiff', '.tif', '.png']
        
        # Get sorted file lists with assertion
        self.ir_list = self._get_sorted_files(ir_folder)
        self.vi_list = self._get_sorted_files(vi_folder)
        self.ir_mask_list = self._get_sorted_files(ir_mask)
        self.vi_mask_list = self._get_sorted_files(vi_mask)
        
        # Verify all lists have same length
        assert len(self.ir_list) == len(self.vi_list) == len(self.ir_mask_list) == len(self.vi_mask_list), \
            "Mismatch in number of files between input folders"
        
        self.crop = crop

    def _get_sorted_files(self, folder: pathlib.Path) -> list:
        """Get sorted list of image files with supported extensions"""
        return sorted([x for x in folder.glob('*') 
                      if x.suffix.lower() in self.supported_ext])

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], 
                                         Tuple[str, str]]:
        # Get paths
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]
        ir_mask_path = self.ir_mask_list[index]
        vi_mask_path = self.vi_mask_list[index]

        # Read images
        ir = self._read_image(ir_path)
        vi = self._read_image(vi_path)
        ir_mask = self._read_image(ir_mask_path)
        vi_mask = self._read_image(vi_mask_path)
        
        # Apply cropping
        ir = self.crop(ir)
        vi = self.crop(vi)
        ir_mask = self.crop(ir_mask)
        vi_mask = self.crop(vi_mask)
        
        return (ir, vi, ir_mask, vi_mask), (str(ir_path), str(vi_path))

    def __len__(self) -> int:
        return len(self.ir_list)

    @staticmethod
    def _read_image(path: pathlib.Path) -> torch.Tensor:
        """
        Read image with auto-detect for TIFF 16-bit/32-bit images
        Returns normalized FloatTensor in [0,1] range
        """
        # Special handling for TIFF
        if path.suffix.lower() in ['.tiff', '.tif']:
            img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
            if img.dtype == 'uint16':
                img = img.astype('float32') / 65535.0
            elif img.dtype == 'uint32':
                img = img.astype('float32') / 4294967295.0
            else:
                img = img.astype('float32') / 255.0
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read image: {str(path)}")
            img = img.astype('float32') / 255.0
        
        return kornia.utils.image_to_tensor(img).type(torch.FloatTensor)



class JointTestData(torch.utils.data.Dataset):
    """
    Load test dataset with infrared and visible image pairs
    Supports: .bmp, .jpg/.jpeg, .tiff/.tif, .png image formats
    """

    def __init__(self, 
                 ir_folder: pathlib.Path, 
                 vi_folder: pathlib.Path,
                 crop: Callable = lambda x: x):
        super(JointTestData, self).__init__()
        
        # Supported image extensions (case insensitive)
        self.supported_ext = ['.bmp', '.jpg', '.jpeg', '.tiff', '.tif', '.png']
        
        # Get sorted file lists with validation
        self.ir_list = self._get_validated_files(ir_folder)
        self.vi_list = self._get_validated_files(vi_folder)
        
        # Verify matching filenames
        self._validate_filename_pairs()
        
        self.crop = crop

    def _get_validated_files(self, folder: pathlib.Path) -> list:
        """Get sorted list of image files with supported extensions"""
        files = sorted([x for x in folder.glob('*') 
                       if x.suffix.lower() in self.supported_ext])
        if not files:
            raise ValueError(f"No valid images found in {folder}. Supported formats: {self.supported_ext}")
        return files

    def _validate_filename_pairs(self):
        """Verify all IR and VI images have matching filenames"""
        if len(self.ir_list) != len(self.vi_list):
            raise ValueError(f"File count mismatch: IR({len(self.ir_list)}) vs VI({len(self.vi_list)})")
        
        for ir_path, vi_path in zip(self.ir_list, self.vi_list):
            if ir_path.stem != vi_path.stem:  # Compare filenames without extensions
                raise ValueError(
                    f"Filename mismatch:\n"
                    f"IR: {ir_path.name}\n"
                    f"VI: {vi_path.name}\n"
                    f"All pairs must have identical basenames (e.g. 'scene1_rgb.jpg' and 'scene1_nir.tiff')"
                )

    def __getitem__(self, index) -> Tuple[Tuple[torch.Tensor, torch.Tensor], 
                                        Tuple[str, str]]:
        """Returns ((ir_tensor, vi_tensor), (ir_path, vi_path))"""
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        # Read and process images
        ir = self._read_image(ir_path)
        vi = self._read_image(vi_path)
        
        return (ir,vi), (str(ir_path), str(vi_path))

    def __len__(self) -> int:
        return len(self.ir_list)

    @staticmethod
    def _read_image(path: pathlib.Path) -> torch.Tensor:
        """
        Robust image reading with TIFF support
        Returns:
            torch.FloatTensor in [0,1] range
        """
        # Special handling for multi-channel TIFF
        if path.suffix.lower() in ('.tiff', '.tif'):
            img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if img is None:
                raise RuntimeError(f"Failed to read TIFF image: {path}")
                
            # Normalize based on bit depth
            if img.dtype == 'uint16':
                img = img.astype('float32') / 65535.0
            elif img.dtype == 'uint32':
                img = img.astype('float32') / 4294967295.0
            else:  # 8-bit
                img = img.astype('float32') / 255.0
                
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            img = img.astype('float32') / 255.0

        return kornia.utils.image_to_tensor(img).type(torch.FloatTensor)
