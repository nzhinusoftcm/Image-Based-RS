"""
@author: Carmel WENGA
This python file contains useful function for image base recommender system.
For further details about these functions, please refer to the tutorial notebook.
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tarfile
import zipfile
import random
import urllib
import json
import sys
import os


def generate_database(metadata, folder):
    """
    This function reads metadata from metadata.json, creates a separate metadata file for each product and saves
    them in the folder specified by folder
    
    :param 
        - metadata: json file in which each line represents metadata for a product
        - folder: path where to save metadata files
    """
    
    # create the folder if not exist
    os.makedirs(folder, exist_ok=True)
    
    # initialize the list of products
    items = []
    
    print('[INFO] generating metadata file for each item ...')
    with open(metadata, 'r') as file :
        line = file.readline()
        while line :
            
            # read metadata of the current item
            item = eval(line.strip())
            
            # the file name is defined the ID of the product
            name = str(item['ID']) + '.json'
            path = os.path.join(folder, name)
            
            # save metadata file of the current item
            with open(path, 'w') as out:
                json.dump(item, out)
            
            # add ID of the current item in the list of items
            items.append(item['ID'])
            
            # read the next line (next item)
            line = file.readline()


def get_item(ID):
    """    
    :param ID: id of a product
    :return item: dictionary of metadata of the corresponding product
    """
    
    folder = 'items metadata'
    file_name = str(ID) + '.json'
    
    with open(os.path.join(folder, file_name),'r') as file:
        item = eval(file.readline().strip())
        
    return item


def get_all_items():
    """
    
    :return items: list of dictionaries of product's metadata
    """
    
    folder = 'items metadata'
    items_path = os.listdir(folder)
    items = []
    
    for path in items_path:
        with open(os.path.join(folder, path),'r') as file:
            items.append(eval(file.readline().strip()))
        
    return items


def visualize_random_items(items, n=50):
    """
    Randomly choose n items in the database and display them in a grib
    
    :param 
        - items: list of item's IDs
        - n : number of items to display. Default value = 50
    """
    # root folder where images data are stored
    root = "dataset"
    
    # randomly choose 50 items of products
    random_items = random.choices(items, k=n)
    
    # load images data of randomly choosed items
    images = []
    
    for item in random_items:
        images.append(mpimg.imread(os.path.join(root, item['imPath'])))
    
    # defining the grid
    cols = 10
    rows = n // cols
    
    # size of images in the grid
    fig = plt.figure(figsize = (10,10))

    axis = []

    for i, img  in enumerate(images):
        axis.append(fig.add_subplot(rows,cols,i+1))
        axis[i].imshow(img)
        plt.axis("off")
    
    # display the grid
    plt.show()


def recommend_items(item):
    """
    Display or recommend the 10 most image-based similar products for the given item
    
    :param item: metadata of the product for which we wish to make recommendation
    """
    # root directory where images of products are saved
    root = "dataset"
    
    # size of images in the grid
    fig = plt.figure(figsize = (16,3))
    
    # get image of the referenced product and display it in the matplotlib grid
    img_ref_product = mpimg.imread(os.path.join(root, item['imPath']))
    
    axis = []
    axis.append(fig.add_subplot(2,10,1))
    axis[0].imshow(img_ref_product)
    plt.axis("off")
    
    # get similar product of the referenced product. Recall that attribute IBSP in metadata of a product contains
    # the list of similar items with their corresponding similarities.
    similar_products = item['IBSP']
    
    # retrieve similar images and store them in variable img_sim_products
    # initialize list of similar images 
    img_sim_products = []
    
    # loop over similar items and get their corresponding images
    for sim_product in similar_products:
        metadata = get_item(sim_product['id'])
        image = os.path.join(root, metadata['imPath'])
        img_sim_products.append(mpimg.imread(image))
    
    # display similar images in the grid
    for i, img  in enumerate(img_sim_products):
        axis.append(fig.add_subplot(2,10,i+11))
        axis[i+1].imshow(img)
        plt.axis("off")
    
    # display the grid
    plt.show()

    
def unzip_data():
    zip_data = "dataset.zip"

    print("[INFO] unzip the dataset ...")
    with zipfile.ZipFile(zip_data, 'r') as data:
        data.extractall()