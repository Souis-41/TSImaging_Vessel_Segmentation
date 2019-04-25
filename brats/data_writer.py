#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:24:21 2019

@author: sure
"""

from PIL import Image
import tables
import numpy as np
import os
from keras.utils.np_utils import to_categorical

class data_writer(object):
    
    def __init__(self, input_txt_path, output_hdf5_path):
        
        self.input_txt_path = input_txt_path
        self.output_hdf5_path = output_hdf5_path
        self.images_dir = './carotid_images/'
        self.img_width = 288
        self.img_height = 288
        self.img_slice = 16
        self.modality_num = 4
        self.n_label = 6
    
    def create_data_file(self):
        
        hdf5_file = tables.open_file(self.output_hdf5_path, mode='w')
        filters = tables.Filters(complevel=0)
        data_shape = tuple([0, self.modality_num, self.img_width, self.img_height, self.img_slice])
        truth_shape = tuple([0, self.n_label, self.img_width, self.img_height,  self.img_slice])
        data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                               filters=filters)
        mask_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                                filters=filters)
        
        return hdf5_file, data_storage, mask_storage


    def add_data_to_storage(self, data_storage, mask_storage, subject_example, mask_labels_example, modality_num):
        
        data_storage.append(np.asarray(subject_example)[np.newaxis])
        mask_storage.append(np.asarray(mask_labels_example)[np.newaxis])
        
        
    def slice_fixed_subject(self, subject_example, img_slice): ##(?,4,288,288)

        
        subject_example = np.transpose(subject_example, (1,0,2,3))  ## expected (4,?,288,288)
        modality_num, _, width, height= np.shape(subject_example)
        
        subject_assert_example = np.zeros((modality_num, width, height, img_slice))
        
        for i in range(modality_num):
            
            mod_img = subject_example[i] ## (?,288,288)
        
            if len(mod_img) == img_slice:
                pass
            else:
                mod_img = self.Resize(mod_img, target_slice = img_slice, interpolation="linear") ##(16,288,288)
            
            mod_img = np.transpose(mod_img, (1,2,0)) ##(288,288,16)
            
            subject_assert_example[i]= mod_img ##(4,288,288,16)
            
        
        assert np.shape(subject_assert_example) == (self.modality_num, self.img_width, self.img_height, self.img_slice)
        return subject_assert_example
    
    def slice_fixed_mask(self, mask_example, img_slice): ##(?,288,288)
        
        if len(mask_example) == img_slice:
            pass
        else:
            mask_example = self.Resize(mask_example, target_slice = img_slice, interpolation="nearest") ##(16,288,288)
            
        mask_labels = to_categorical(mask_example, num_classes = self.n_label)  ##(16,288,288,6)
        mask_labels = np.transpose(mask_labels,(3,1,2,0)) ##(6,288,288,16)
        
        assert np.shape(mask_labels) == (self.n_label, self.img_width, self.img_height,  self.img_slice)
        return mask_labels
    
    def save_hdf5(self):
        
        txt_path = self.input_txt_path
        images_dir = self.images_dir
        img_width = self.img_width
        img_height = self.img_height
        img_slice = self.img_slice
        modality_num = self.modality_num
        subject_id_list = list()
        subject_example = list()
        mask_example = list()
        id_num = 0
        
        hdf5_file, data_storage, mask_storage = self.create_data_file()
        
        with open(txt_path) as f:
            lines = f.readlines()
            
        for line_id in range(len(lines)):
            
            img = np.zeros((modality_num, img_width, img_height))
            line = lines[line_id].strip().split(' ')
            
            vessel = Image.open(images_dir+line[4]).resize((img_width,img_height))
            mask = Image.open(images_dir+line[5]).resize((img_width,img_height))
            mask = np.array(mask, dtype=np.uint8)
            
            for modality_id in range(modality_num):
                
                img[modality_id] = Image.open(images_dir+line[modality_id]).resize((img_width,img_height))
                # multiply vessel information
                img[modality_id] = np.multiply(np.array([modality_id], dtype=np.float32), np.array(vessel, dtype=np.float32))
                
                
            example_ids = line[0].split('_')
            example_id = example_ids[0]
            example_side = example_ids[1]
            example_special = str(example_id + '_' + example_side)
            
            if line_id == 0:
                
                subject_id_list.append(example_special)
                id_num += 1
                print(str(id_num) + '*************' + example_special+ '****************')
                subject_example.append(img)
                mask_example.append(mask)
            
            elif example_special in subject_id_list:
                
                subject_example.append(img)
                mask_example.append(mask)
                
            else:
                
                subject_id_list.append(example_special)
                id_num += 1
                print(str(id_num) + '*************' + example_special + '****************')
                subject_example = self.slice_fixed_subject(subject_example, img_slice)##(4,288,288,16)
                mask_labels_example = self.slice_fixed_mask(mask_example, img_slice) ## (6,288,288,16)
                self.add_data_to_storage(data_storage, mask_storage, subject_example, mask_labels_example, modality_num)
                
                ## clear list, continue append operation
                subject_example = list()
                subject_example.append(img)
                mask_example = list()
                mask_example.append(mask) ##(?, 288,288)
                
        
        subject_example = self.slice_fixed_subject(subject_example, img_slice)##(4,288,288,16)
        mask_labels_example = self.slice_fixed_mask(mask_example, img_slice) ## (6,288,288,16)
        self.add_data_to_storage(data_storage, mask_storage, subject_example, mask_labels_example, modality_num)
                
        hdf5_file.close()
                
                
writer = data_writer('./imageList_trval.txt', './data_storage.h5')
writer.save_hdf5()