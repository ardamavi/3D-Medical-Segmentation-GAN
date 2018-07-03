# Arda Mavi

import sys
import numpy as np
from get_dataset import get_scan, scan_pading, save_imgs
from keras.models import model_from_json

def predict(model, dicom_path):
    X = get_scan(dicom_path)
    X, _ = scan_pading(scan, None, section_size = 16)

    splitted_scans = []
    for i in range(0, scans.shape[-1]-15):              # TODO
        splitted_scans.append(scans[:,:,i:i+16])

    X = np.array(splitted_scans, dtype='float64')
    X = (X-np.min(X))/(np.max(X)-np.min(X))
    Y = model.predict(X)*255.
    Y = np.rot90(np.fliplr(Y))
    return Y

if __name__ == '__main__':
    dicom_path = sys.argv[1]

    # Getting model:
    with open('Data/Model/model.json', 'r') as model_file:
        model = model_file.read()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")

    Y = predict(model, dicom_path)
    save_imgs(Y, 'mask.png') # TODO: save_imgs
    print('Segmentated image saved as '+name)
