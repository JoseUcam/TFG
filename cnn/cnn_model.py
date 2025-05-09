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

#print(tf.__version__)
#print(np.__version__)
#print(keras.__version__)

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


    #@staticmethod
    #def categorizador_local(model:object, path:str) -> int:
    #    img = Image.open(path)
    #    img = np.array(img)
    #    img = cv2.resize(img, (CNNModel.IMAGE_SIZE, CNNModel.IMAGE_SIZE))
    #    img = img.astype(np.uint8)
    #    prediccion = model.predict(img.reshape(-1, CNNModel.IMAGE_SIZE, CNNModel.IMAGE_SIZE, 3))
    #    return np.argmax(prediccion[0], axis=-1) , max(prediccion[0])

    @staticmethod
    def categorizador_local(model:object, path:str):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)  # Leer imagen
        image = tf.image.resize(image,  (CNNModel.IMAGE_SIZE, CNNModel.IMAGE_SIZE))    # Cambiar tamaño
        image = tf.cast(image, tf.uint8)
        image = tf.expand_dims(image, axis=0)
        prediccion = model.predict(image)
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


    @staticmethod
    def cargar_imagenes(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)  # Leer imagen
        image = tf.image.resize(image, (CNNModel.IMAGE_SIZE, CNNModel.IMAGE_SIZE))    # Cambiar tamaño
        image = tf.cast(image, tf.uint8)
        #image = tf.expand_dims(image, axis=0)
        return image

    @staticmethod
    def crear_dataset(rutas_imagenes, batch_size):
        # Cargar imágenes
        dataset = tf.data.Dataset.from_tensor_slices(rutas_imagenes)

        # Cargar datos
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = dataset.map(CNNModel.cargar_imagenes, num_parallel_calls=AUTOTUNE)

        # Particionar data en lotes y uso de prefetch para acelerar el cargado de datos
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    @staticmethod
    def categorizador_lotes(model, paths, batch_size=128):
        test_ds = CNNModel.crear_dataset(paths, batch_size)
        predictions = model.predict(test_ds)
        return predictions.argmax(axis=-1) , predictions.max(-1)


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

def diagnosticar_lotes(modelo:object, predictor:dict, paths:str, batch_size=64):
    '''
    Categorizar imágenes con el modelo
    '''
    print('Categorizando imágenes ...')
    # -- Realizar la predicción
    try:
        cat_ids, max_vals = CNNModel.categorizador_lotes(modelo, rutas, batch_size)
        predictions = []
        for cat_id, max_val in zip(cat_ids, max_vals):
            predictions.append({'prediccion': predictor[cat_id].lower(), 'probabilidad':max_val, 'status':True, 'message':'OK'})
        return predictions
    except Exception as e:
        return [{'prediccion': 'Ocurrió un error al procesar las imagenes', 'probabilidad':0, 'status': False, 'message':str(e)}]


if __name__ == '__main__':
    #cnn = CNNModel()
    #cnn.save_predictor(path=PREDICTOR_NAME)
    print(MODEL_NAME)
    #print(numpy.__file__)
    predictor = CNNModel.get_predictor(path=PREDICTOR_NAME)
    modelo = CNNModel.load_model(path=MODEL_NAME)
    #modelo = CNNModel.load_model(path="")
    

    print(f'Predictor: {predictor}')
    print(f'Modelo: {modelo}')

    #result = diagnosticar(modelo, predictor, "C:\\Users\\migue\\Dropbox\\PC\\Desktop\\1 CELULAS DETECTADAS\\celulas_2025-04-22_12-08-06\\celula_0.jpeg")
    #print(result)
    
    rutas = ["C:\\Users\\migue\\Dropbox\\PC\\Desktop\\1 CELULAS DETECTADAS\\celulas_2025-04-22_12-08-06\\celula_0.jpeg"]
    result = diagnosticar_lotes(modelo, predictor, rutas)
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