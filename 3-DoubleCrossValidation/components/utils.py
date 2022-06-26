import pickle
import os
import glob
import logging

def save_cache(data:object, path:str, filename:str='data.pickle'):
    if(os.path.exists(path) == False):
        os.mkdir(path)

    file_path = os.path.join(path, filename)
    with open(f"{file_path}", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_cache(path:str, filename:str='data.pickle'):
    file_path = os.path.join(path, filename)

    if(os.path.exists(file_path) == False):
        raise FileNotFoundError(f"The file {file_path} is not exists.")

    with open(f"{file_path}",'rb') as handle:
        data = pickle.load(handle)
    return data 

def clear_cache(path:str, filename:str=None): # type: ignore
    if(filename == None):
        file_paths = glob.glob(os.path.join(path, "*.pickle"))
    else:
        file_paths = [os.path.join(path, filename)]
    
    for file_path in file_paths:
        os.remove(file_path)
        logger = logging.getLogger('main-logger')
        logger.info(f"cache:{file_path} is removed.")