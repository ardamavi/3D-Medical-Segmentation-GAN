# Arda Mavi

import os
import pydicom
import numpy as np
import dicom_numpy
from os import listdir
from scipy.misc import imread, imresize, imsave
from sklearn.model_selection import train_test_split

def get_np(np_file_path):
    # Getting numpy array:
    if not os.path.exists(np_file_path):
        print('Numpy array file not exists!')
        return
    np_array = np.load(np_file_path)
    return np_array

def get_scan(dicom_path):
    # Getting DICOM images from path:
    if not os.path.exists(dicom_path):
        print('DICOM files not exists!')
        return
    dicom_files = listdir(dicom_path)
    dicom_files.sort()
    voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices([pydicom.read_file(dicom_path+'/'+dcm_file, force=True) for dcm_file in dicom_files])
    return voxel_ndarray

def get_img(data_path, img_size):
    # Getting image array from path:
    img = imread(data_path, flatten = True)
    img = imresize(img, img_size[0:2])
    img = img.reshape(img_size)
    img = np.fliplr(np.rot90(img, 3))
    return img

def get_seg_img(images_path, img_size):
    # Getting segmented images from path:
    if not os.path.exists(images_path):
        print('Segmented images not exists!')
        return

    images = []
    images_files = listdir(images_path)
    images_files.sort()
    for one_img in reversed(images_files):
        img = get_img(images_path+'/'+one_img, img_size)
        if images == []:
            images = img
        else:
            images = np.dstack((images, img))

    return images

def save_seg_imgs(seg_imgs, save_path):
    for i in range(0, seg_imgs.shape[-1]):
        imsave(save_path+'/SegImg_'+str(i)+'.png', seg_imgs[:,:,i])
    print('Segmentated images saved into the ' + save_path)

def scan_pading(scan, seg_img, section_size = 16):
    # For easly split:
    pad_size = section_size - (scan.shape[-1] % section_size)
    if pad_size != 16:
        padded_scan = np.pad(scan, ((0,0),(0,0),(0,pad_size)), 'constant')
        try:
            padded_seg_img = np.pad(seg_img, ((0,0),(0,0),(0,pad_size)), 'constant')
        except:
            padded_seg_img = None
    else:
        padded_scan = scan
        padded_seg_img = seg_img
    return padded_scan, padded_seg_img


def split_scans_imgs(scans, seg_img, section_size = 16):
    # Split with sliding window:

    splitted_scans = []
    for i in range(0, scans.shape[-1]-15):
        splitted_scans.append(scans[:,:,i:i+16])

    splitted_seg_img = []
    for i in range(0, seg_img.shape[-1]-15):
        splitted_seg_img.append(seg_img[:,:,i:i+16])

    splitted_scans = np.array(splitted_scans)
    splitted_seg_img = np.array(splitted_seg_img)
    return splitted_scans, splitted_seg_img

def get_dataset(dataset_path, dicom_file = 'DICOM_anon', ground_file = 'Ground', section_size = (512, 512, 16), test_size = 0.2, save_npy = True, dataset_save_path = 'Data/npy_dataset'):
    # Create dateset:

    scans, seg_imgs = [], []
    samples = listdir(dataset_path)
    for sample_id in samples:
            print('Reading dataset: '+sample_id+' ...')
            sample_path = dataset_path+'/'+sample_id

            scan = get_scan(sample_path+'/'+dicom_file)
            seg_img = get_seg_img(sample_path+'/'+ground_file, img_size = section_size[0:2]+(1,))

            scan, seg_img = scan_pading(scan, seg_img, section_size = 16)
            scan, seg_img = split_scans_imgs(scan, seg_img, section_size = 16)

            for one_scan in scan:
                scans.append(one_scan)
            for one_seg_img in seg_img:
                seg_imgs.append(one_seg_img)

    scans = np.array(scans, dtype='float32')

    scans = (scans-np.min(scans))/(np.max(scans)-np.min(scans)) # Normalization
    seg_imgs = np.array(seg_imgs).astype('float32')/255

    scans = scans.reshape((scans.shape[0],)+section_size+(1,))
    seg_imgs = seg_imgs.reshape((seg_imgs.shape[0],)+section_size+(1,))

    print('Scan Data Shape: ' + str(scans.shape))
    print('Segmantation Data Shape: ' + str(seg_imgs.shape))

    if save_npy:
        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path)
        np.save(dataset_save_path+'/scans.npy', scans)
        np.save(dataset_save_path+'/seg.npy', seg_imgs)
        print('NPY dataset saved!')

    X, X_test, Y, Y_test = train_test_split(scans, seg_imgs, test_size=test_size, random_state=42)
    print('Train Data Shape: ' + str(X.shape[0]))
    print('Test Data Shape: ' + str(X_test.shape[0]))
    return X, X_test, Y, Y_test

def read_npy_dataset(npy_dataset_path, test_size = 0.2):
    X = np.load(npy_dataset_path+'/scans.npy')
    Y = np.load(npy_dataset_path+'/seg.npy')
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    print('Train Data Shape: ' + str(X.shape[0]))
    print('Test Data Shape: ' + str(X_test.shape[0]))
    return X, X_test, Y, Y_test

if __name__ == '__main__':
    X, X_test, Y, Y_test = get_dataset(dataset_path = 'Data/Dataset', dicom_file = 'DICOM_anon', ground_file = 'Ground', section_size = (512, 512, 16), test_size = 0.2, save_npy = True, dataset_save_path = 'Data/npy_dataset')
