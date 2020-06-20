"""
@author: Carmel WENGA
This python file define useful function to compute similarities between images.
For more details about these functions, please refer to the tutorial's notebook of this repository.
"""

from tqdm import tqdm
from ibrs_utils import get_all_items

import json
import numpy as np
import nmslib
import os

class NearestNeighbors:

    def __init__(self):
        self.index_name = "index.bin"
        self.index_dir = "nmslib_index"
        self.index_path = os.path.join(self.index_dir, self.index_name)

    def create_nmslib_index(self, items, output_dir):
        """
        Loop over all our products and add their corresponding image features to an nmslib index and saved the index.
        
        :param 
            - items: list of metadatas of our products
            - output_dir: folder where to save the created index
        """
        print("[INFO] create nmslib index ...")
        
        # Initialize the index as description earlier. nmslib.DataType.DENSE_VECTOR just specifies that values of our 
        # image features are mostly non-zero values
        index = nmslib.init(method='hnsw', space='cosinesimil', data_type=nmslib.DataType.DENSE_VECTOR)
        
        # load our data points of image's features into the index. Note that we are setting the id of each data point 
        # to be the id of the corresponding product. 
        for item in items:
            
            # load the image features of the current item, located at item['image_features']. Recall that 
            # item['image_features'] returns the path to the image features of this item
            data_point = np.loadtxt(item['image_features'])
            
            # add the data point to the index
            index.addDataPoint(id=item['ID'], data=data_point)
        
        # create the index
        index.createIndex({'post': 2}, print_progress=True)
            
        # create the output directory if it doesn't exists    
        os.makedirs(self.index_dir, exist_ok=True)
        
        # save the index
        index.saveIndex(self.index_path, save_data=True)


    def load_nmslib_index(self):
        """
        load the nmslib index
        
        :return index: nmslib index
        """
        print("[INFO] load nmslib index")
        
        # initialize and load the index
        index = nmslib.init(method='hnsw', space='cosinesimil', data_type=nmslib.DataType.DENSE_VECTOR)
        index.loadIndex(self.index_path, load_data=True)

        return index

    def compute_nearest_neighbors(self, item, index, k=10):
        """
        Compute nearest neighbors of item and add attribute IBSP (Image Based Similar Product) its metadata
        
        :param
            - item : metadata of the item for which we wish to compute nearest neighbors
            - index : nmslib index where image's features are saved as data points
            - k : number of nearest neighbors to recommend
        """
        
        # load image's features of the product
        item_features = np.loadtxt(item['image_features'])
        
        # compute the nearest neigbors of this image features.
        # ids : list of ids of similar products
        # distances : distances for each id in ids
        ids, distances = index.knnQuery(vector=item_features, k=k+1)
        
        # initialize nearest neighbors of the current product
        nearest_neighbors = []
        
        # normalize distances 
        for iid, distance in zip(ids, distances):
            
            # lets not consider the similarity between this product and it self
            if iid != item['ID']:
                similarity = 1 - int((distance * 10000.0)) / 10000.0
                nearest_neighbors.append({
                    'id': int(iid),
                    'sim': similarity
                })
        
        # save nearest neighbors of the current product by adding attribute IBSP to its metadata
        item['IBSP'] = nearest_neighbors
        
        with open(os.path.join('items metadata', str(item['ID']) + '.json'), 'w') as out:
            json.dump(item, out)

    
    def nearest_neighbors(self):
        """
        Compute nearest neighbors for all products in our database
        """
        
        # get metadata of all products
        items = get_all_items()
        
        # create the nmslib index if it doesn't already exists
        if not os.path.exists(self.index_path):
            NearestNeighbors.create_nmslib_index(self, items, self.index_dir)
        
        # load the nmslib index. The nmslib index has been saved in the folder 'nmslib index'
        index = NearestNeighbors.load_nmslib_index(self)
        
        print('[INFO] computing nearest neighbors for all images ...')
        # loop over all items to compute their nearest neighbors    
        with tqdm(total=len(items)) as progressbar:
            for item in items:
                NearestNeighbors.compute_nearest_neighbors(self,item=item, index=index)
                progressbar.update(1)