from carla.self_explaining_model.catalog.vcnet.library.utils import *
from torch import nn
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder
import pathlib  
from carla.data.catalog import DataCatalog

# Load configuration and prepare data  
class Load_dataset_carla(nn.Module):
    '''
    
    Class to load the CARLA framework data for VCnet fitting
    
    Parameters
    ----------
    data : carla.data.Datacatalog
    Dataset 
    model_config_dict : Dict 
    VCnet training hyperparameters 

    Methods 
    --------
    prepare_data:
    Preprocess and split the data for VCnet training 
    
    '''
    def __init__(self, data_catalog : DataCatalog , model_config_dict : dict, return_data_loader=True):
        super().__init__()
        
        self.return_data_loader = return_data_loader
        self.name = data_catalog.name
        self.continous_cols = data_catalog.continuous
        self.discret_cols = data_catalog.categorical
        self.class_size = 2 
        self.feature_size = data_catalog.df_train.shape[1] - 1
        self.data_catalog = data_catalog
        
        # load training and model configs from model_config_dict
        for param in model_config_dict.keys() :
            setattr(self,param,model_config_dict[param])
        

    # Preprocessing and splitting 
    def prepare_data(self):
        
        # target 
        self.target= self.data_catalog.target
        # Train and test 
        self.train = self.data_catalog.df_train 
        self.test = self.data_catalog.df_test
        self.val = self.data_catalog.df_val
        # preprocessing 
        self.normalizer = self.data_catalog.scaler
        self.encoder = self.data_catalog.encoder
        
        # Categories is second column for each categorical feature (drop first)
        cat_arrays = self.encoder.categories_ if self.discret_cols else []
        
        def convert_drop_first(cat_arrays):
            cat_arrays_drop_first = [] 
            for e in cat_arrays : 
                cat_arrays_drop_first.append(np.array([e[1]]))
            return(cat_arrays_drop_first)
        
        cat_arrays = convert_drop_first(cat_arrays)
        #cat_arrays = self.data_catalog.encoder.get_feature_names(self.data_catalog.categorical) if self.discret_cols else []
        #print("ENCODER CATEGORIES: ",self.encoder.categories_)
        # Number of continious variables 
        cont_shape = len(self.data_catalog.continuous)


        train_X = self.train.drop(columns = [self.target]).to_numpy()
        train_y = self.train[self.target].to_numpy()
        
        val_X = self.val.drop(columns = [self.target]).to_numpy()
        val_y = self.val[self.target].to_numpy()

        test_X = self.test.drop(columns = [self.target]).to_numpy()
        test_y = self.test[self.target].to_numpy()
        

       
        
        self.train_dataset = NumpyDataset(train_X, train_y)
        self.test_dataset = NumpyDataset(test_X, test_y)
        self.val_dataset = NumpyDataset(val_X, val_y)
        
        
        if self.return_data_loader : 
            #Dataloaders 
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,pin_memory=True, shuffle=True, num_workers=0)
            val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,pin_memory=True, shuffle=True, num_workers=0)
            test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,pin_memory=True, shuffle=False, num_workers=0)
            loaders = { 'train' : train_loader, "test" : test_loader, "val" : val_loader }
        
            return(loaders,cat_arrays,cont_shape)
        else : 
            return(cat_arrays,cont_shape)