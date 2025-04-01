import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
import tensorflow as tf
import requests
from io import BytesIO
import cv2
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle

from dotenv import load_dotenv
load_dotenv()

print(tf.__version__)
print(np.__version__)
print(keras.__version__)

# -- Leer el archivo .env
try:
    MODEL_NAME = os.getenv('model_name')
    PREDICTOR_NAME = os.getenv('predictor_name')
    BATCH_SIZE = int(float(os.getenv('batch_size')))
    EPOCHS = int(float(os.getenv('epochs')))
    IMAGE_SIZE = int(float(os.getenv('image_size')))
except Exception as e:
    raise ValueError(f"Error: {e}")


class CNNModel(object):
    IMAGE_SIZE = IMAGE_SIZE
    BATCH_SIZE = BATCH_SIZE
    EPOCHS = EPOCHS

    def __init__(self):
        pass
        #self.directorios = ["train_altogrado", "train_ascus", "train_bajogrado", "train_benigna"]
        #self.source_folders = [("./train_altogrado", "./dataset/altogrado"), ("./train_ascus", "./dataset/ascus"), ("./train_bajogrado", "./dataset/bajogrado"), ("./train_benigna","./dataset/benigna")]

    @staticmethod
    @tf.keras.utils.register_keras_serializable()
    def preprocess_input_lambda_efficientnetv2(x):
        return tf.keras.applications.efficientnet_v2.preprocess_input(x)

    @staticmethod
    def load_model(path:str):
        '''
        Cargar el modelo desde el archivo SavedModel
        '''
        #import tensorflow as tf
        return tf.keras.models.load_model(path)

    @staticmethod
    def categorizador_web(model:object, url:str) -> int:
        res = requests.get(url)
        img = Image.open(BytesIO(res.content))
        img = np.array(img)
        img = cv2.resize(img, (CNNModel.IMAGE_SIZE, CNNModel.IMAGE_SIZE))
        img = img.astype(np.uint8)
        prediccion = model.predict(img.reshape(-1, CNNModel.IMAGE_SIZE, CNNModel.IMAGE_SIZE, 3))
        return np.argmax(prediccion[0], axis=-1), max(prediccion[0])


    @staticmethod
    def categorizador_local(model:object, path:str) -> int:
        img = Image.open(path)
        img = np.array(img)
        img = cv2.resize(img, (CNNModel.IMAGE_SIZE, CNNModel.IMAGE_SIZE))
        img = img.astype(np.uint8)
        prediccion = model.predict(img.reshape(-1, CNNModel.IMAGE_SIZE, CNNModel.IMAGE_SIZE, 3))
        return np.argmax(prediccion[0], axis=-1) , max(prediccion[0])


    def save_predictor(self, path:str):
        '''
        Almacena el predictor en un archivo
        '''
        labels = {0: 'altogrado', 1: 'ascus', 2: 'bajogrado', 3: 'benigna'}
        with open(path, 'wb') as f:
            pickle.dump(labels, f)

    def get_predictor(path:str):
        '''
        Obtiene el predictor desde un archivo
        '''

        with open(path, 'rb') as f:
            return pickle.load(f)


def diagnosticar(modelo:object, predictor:dict, file_path:str):
    '''
    Categorizar la imagen con el modelo
    '''
    print('Categorizando' + fr'{file_path}' + '...')
    # -- Realizar la predicción
    try:
        cat_id, max_val = CNNModel.categorizador_local(model=modelo, path=fr'{file_path}')
        return {'prediccion': predictor[cat_id].lower(), 'probabilidad':max_val, 'status':True, 'message':'OK'}
    except Exception as e:
        return {'prediccion': 'Ocurrió un error al procesar la imagen', 'probabilidad':0, 'status': False, 'message':str(e)}


if __name__ == '__main__':
    #cnn = CNNModel()
    #cnn.save_predictor(path=PREDICTOR_NAME)
    print(MODEL_NAME)
    predictor = CNNModel.get_predictor(path=PREDICTOR_NAME)
    modelo = CNNModel.load_model(path=MODEL_NAME)

    print(f'Predictor: {predictor}')
    print(f'Modelo: {modelo}')

    result = diagnosticar(modelo, predictor, "D:\\Celulas\\entrenamiento\\altogrado\\altogrado.00061.tiff")
    print(result)
    """
    urls = ['./Celulas/test/altogrado/altogrado.00097.tiff',
            './Celulas/test/ascus/ascus.00066.tiff',
            './Celulas/test/bajogrado/bajogrado.00044.tiff',
            './Celulas/test/benigna/benigna.00391.tiff']
    
    for url in urls:
        result = diagnosticar(modelo, predictor, url)
        print(result)
    """