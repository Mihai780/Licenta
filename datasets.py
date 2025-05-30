import os
import json
import h5py
import torch
from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):
    """
    PyTorch Dataset for loading image-caption pairs from preprocessed HDF5 and JSON files.
    """
    def __init__(self, data_folder, data_name, phase, proc_transform=None):
        """
        data_folder: directory containing data files
        data_name: common identifier for dataset files
        phase: one of 'TRAIN', 'VAL', 'TEST'
        proc_transform: optional image transformation pipeline
        """
        assert phase in {'TRAIN', 'VAL', 'TEST'}, f"Invalid phase: {phase}"
        self.phase = phase

        # Open HDF5 archive with images
        h5_path = os.path.join(data_folder, f"{phase}_IMAGES_{data_name}.h5")
        self.h5_file = h5py.File(h5_path, 'r')
        self.images_ds = self.h5_file['images']
        self.caps_per_img = self.h5_file.attrs['captions_per_image']

        # Load encoded captions and their lengths
        captions_path = os.path.join(data_folder, f"{phase}_ENCODEDCAPS_{data_name}.json")
        with open(captions_path, 'r') as cj:
            self.encoded_caps = json.load(cj)
        lengths_path = os.path.join(data_folder, f"{phase}_CAPLENGTHS_{data_name}.json")
        with open(lengths_path, 'r') as lj:
            self.cap_lengths = json.load(lj)

        self.transform = proc_transform
        self.num_samples = len(self.encoded_caps)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # Determine which image this caption belongs to
        img_index = idx // self.caps_per_img
        img_array = self.images_ds[img_index] / 255.0
        img_tensor = torch.FloatTensor(img_array)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        cap_seq = torch.LongTensor(self.encoded_caps[idx])
        cap_len = torch.LongTensor([self.cap_lengths[idx]])

        if self.phase == 'TRAIN':
            return img_tensor, cap_seq, cap_len
        else:
            #Adds all the captions for an image to compare it to a generated caption
            start = img_index * self.caps_per_img
            end = start + self.caps_per_img
            all_caps = torch.LongTensor(self.encoded_caps[start:end])
            return img_tensor, cap_seq, cap_len, all_caps
