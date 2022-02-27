import pandas as pd
import numpy as np
import pickle
import os
import gc
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class DataProcessor:
    def __init__(self, data_dir:str):
        self.base = data_dir # data diectory
    
    def _load_raw_data(self) -> dict:
        """Load original raw data

        Returns:
            dict: raw data dictionary
        """
        articles  = pd.read_csv(self.base+'articles.csv')
        customers = pd.read_csv(self.base+'customers.csv')
        trans     = pd.read_csv(self.base+'transactions_train.csv')

        return {'item':articles, 'user':customers, 'trans':trans}
    
    def _encode_id(self, data:dict, map_dir:str) -> dict:
        """Encode user and item id as integers

        Args:
            data (dict): raw data dictionary, keys: 'item', 'user', 'trans'
            map_dir (str): relative directory to store index-id-maps

        Returns:
            dict: data dictionary
        """
        if not os.path.isdir(self.base+map_dir):
            os.mkdir(self.base+map_dir)

        user_id2index_path = self.base + map_dir + 'user_id2index.pkl'
        user_index2id_path = self.base + map_dir + 'user_index2id.pkl'
        item_id2index_path = self.base + map_dir + 'item_id2index.pkl'
        item_index2id_path = self.base + map_dir + 'item_index2id.pkl'

        user_id2index_dict = dict(zip(data['user']['customer_id'], data['user'].index+1))
        user_index2id_dict = dict(zip(data['user'].index+1, data['user']['customer_id']))
        item_id2index_dict = dict(zip(data['item']['article_id'], data['item'].index+1))
        item_index2id_dict = dict(zip(data['item'].index+1, data['item']['article_id']))
        pickle.dump(user_id2index_dict, open(user_id2index_path, 'wb'))
        pickle.dump(user_index2id_dict, open(user_index2id_path, 'wb'))
        pickle.dump(item_id2index_dict, open(item_id2index_path, 'wb'))
        pickle.dump(item_index2id_dict, open(item_index2id_path, 'wb'))
        
        data['trans']['customer_id'] = data['trans']['customer_id'].map(user_id2index_dict)
        data['trans']['article_id']  = data['trans']['article_id'].map(item_id2index_dict)
        data['user']['customer_id']  = data['user']['customer_id'].map(user_id2index_dict)
        data['item']['article_id']   = data['item']['article_id'].map(item_id2index_dict)

        return data
    
    def _transform_feats(self, data:dict) -> dict:
        """Transform features (label encode and change dtypes)

        Args:
            data (dict): data dictionary, keys: 'item', 'user', 'trans'

        Returns:
            dict: data dictionary
        """
        trans = data['trans']
        user = data['user'].fillna(-1)
        item = data['item']

        # Transactions
        trans['price'] = trans['price'].astype('float32')
        trans['sales_channel_id'] = trans['sales_channel_id'].astype('int8')

        # Customers
        user_sparse_feats = [x for x in user.columns if x not in ['age']]
        for feat in tqdm([x for x in user_sparse_feats if x!='customer_id'], 'Encode User Sparse Feats'):
            lbe = LabelEncoder()
            user[feat] = lbe.fit_transform(user[feat].astype(str)) + 1
            user[feat] = user[feat].astype('int32')
        
        # Articles
        item_sparse_feats = ['article_id', 'product_code', 'product_type_no', 'product_group_name', 
                             'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 
                             'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 
                             'section_no', 'garment_group_no']
        for feat in tqdm([x for x in item_sparse_feats if x!='article_id'], 'Encode Item Sparse Feats'):
            lbe = LabelEncoder()
            item[feat] = lbe.fit_transform(item[feat].astype(str)) + 1
            item[feat] = item[feat].astype('int32')
        
        data['trans'] = trans
        data['user'] = user
        data['item'] = item[item_sparse_feats]

        return data
    

    def save_data(self, data:dict, name:str):
        """Save data dictionary as parquet

        Args:
            data (dict): data dictionary, keys: 'item', 'user', 'trans'
            name (str): name of the data dict (data versioning)
        """
        path = self.base+name+'/'
        if not os.path.exists(path):
            os.mkdir(path)
        data['user'].to_parquet(path+'user.pqt')
        data['item'].to_parquet(path+'item.pqt')
        data['trans'].to_parquet(path+'trans.pqt')
    
    def load_data(self, name:str) -> dict:
        """Load data dictionary

        Args:
            name (str): name of data dict

        Raises:
            OSError: invalid data version

        Returns:
            dict: loaded data dictionary
        """
        path = self.base+name+'/'
        if not os.path.exists(path):
            raise OSError
        data = {}
        data['user'] = pd.read_parquet(path+'user.pqt')
        data['item'] = pd.read_parquet(path+'item.pqt')
        data['trans'] = pd.read_parquet(path+'trans.pqt')

        return data
    
    def preprocess_data(self, save:bool=True, name:str='encoded_full') -> dict:
        """Preprocess raw data

        Args:
            save (bool, optional): whether to save the preprocessed data. Defaults to True.
            name (str, optional): version name of the data to be saved

        Returns:
            dict: preprocessed data
        """
        data = self._load_raw_data()
        data = self._encode_id(data, 'index_id_map/')
        data = self._transform_feats(data)
        if save:
            self.save_data(data, name)
        return data
