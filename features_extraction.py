"""
@author: Carmel WENGA
This python file define useful function to extract image features with class FeaturesExtraction.
For more details about these functions, please refer to the tutorial's notebook of this repository.
"""

from googlenet import GoogLeNet
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import json
import os

class FeaturesExtraction:

    def __init__(self):

        # output directory where to save extracted features
        self.output_dir = "images_features"

    def extract_image_features(self, item, output_dir):
        """
        Extract image's features of a product, save the corresponding features in a .npz format and add the attribute 
        'image_features' in metadata of the current product
        :param
            - item : metadatas of the product for which we extract image features
            - output_dir: folder where to saved the extracted features
        """
        
        # get the image path of the product
        im_path = os.path.join('dataset', item['imPath'])
        
        # initialize image features to an empty list
        img_features = []
        
        try:
            # open an preprocess image of the current product
            img = Image.open(im_path)
            img = img.convert('RGB')
            img = np.array(img)
            
            with tf.compat.v1.Session() as session:
                # extract image feature from the pool_3 node of the googlenet graph
                pool_3 = session.graph.get_tensor_by_name('pool_3:0')
                img_features = session.run(pool_3, {'DecodeJpeg:0': img})
                img_features = np.squeeze(img_features)
        except ValueError as verror:
            print('could not process image {}.\n ValueError : {}'.format(im_path, verror))
            
        # save extracted features in the folder 'image_features' with the .npz format
        fname = str(item['ID']) + ".npz"
        fpath = os.path.join(output_dir, fname)
        np.savetxt(fname=fpath, X=img_features, delimiter=',')
        
        # update metadata of this item by adding attribute 'image_features' to metadata of this product
        item['image_features'] = fpath
        
        with open(os.path.join('items metadata', str(item['ID']) + '.json'), 'w') as out:
            json.dump(item, out)

    def extract_features(self, items):
        """
        Extract image's features for all products in our database
        
        :param items: list of metadatas of all products in our database
        """
        
        # load the googlenet model into the tensorflow session
        model = GoogLeNet()
        model.load_model()
        
        # create the output directory if it not already exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # loop over items to extract image's features
        print('[INFO] extract images features of the entire database ...')
        
        with tqdm(total=len(items)) as progressbar:
            for item in items:
                FeaturesExtraction.extract_image_features(self, item, self.output_dir)
                progressbar.update(1)