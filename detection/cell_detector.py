import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

class CellDetector:

    @staticmethod
    def preprocess_image(img, alpha=1.2, beta=20, kernel_size=(3,3)):
        ### Ajustar brillo y contraste
        alpha = 1.2  # Aumentar contraste
        beta = 20    # Aumentar brillo

        adjusted_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        adjusted_gray = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)

        ### Desenfoque gaussiano
        blur = cv2.GaussianBlur(adjusted_gray, kernel_size, 0)

        ### Binarización
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 30)
        #_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return thresh

    @staticmethod
    def get_markers(img, thresh, kernel_size=(3,3), distance_threshold=0.1):
        ### Eliminar objetos pequeños y obtener fondo con operadores morfológicos
        kernel = np.ones(kernel_size, np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        ### Transformada de distancia para obtener objetos de interés
        # Calcular la transformada de distancia
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, distance_threshold * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        ### Encontrar componentes conectados
        ret, markers = cv2.connectedComponents(sure_fg)

        return markers

    @staticmethod
    def extract_region_rescale(image, bbox, standard_size=(224, 224), min_scale=0.5):
        x, y, w, h = bbox
        center_x, center_y = x + w // 2, y + h // 2

        # Calcular factor de escala
        scale = max(w / standard_size[0], h / standard_size[1])

        if scale < min_scale:
            # Bounding box pequeño, zoom out
            new_w, new_h = int(standard_size[0] / min_scale), int(standard_size[1] / min_scale)
        else:
            new_w, new_h = w, h

        # Calcular nuevo bounding box
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)
        new_x2 = min(image.shape[1], new_x + new_w)
        new_y2 = min(image.shape[0], new_y + new_h)

        # Extraer region
        region = image[new_y:new_y2, new_x:new_x2]

        return region, (new_x, new_y, new_x2, new_y2) #(xmin, ymin, xmax, ymax)


    @staticmethod
    def get_bounding_boxes_from_markers(img, markers, cell_img_size, min_scale=1):

        # Listas para almacenar las imágenes de las células, sus puntuaciones y las coordenadas de sus cajas delimitadoras
        scores = []
        boxes = []

        for label in np.unique(markers):

            # Ignorar el fondo (etiqueta 0) y las regiones desconocidas (etiqueta 1)
            if label <= 1:
                continue

            # Crear una máscara para la región actual
            mask = np.zeros(img.shape[:-1], dtype='uint8')
            mask[markers == label] = 255

            # Encontrar los contornos en la máscara
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Si se encuentran contornos, procesar la región
            if len(cnts) > 0:

                # Obtener la caja delimitadora del primer contorno
                bbox = cv2.boundingRect(cnts[0])

                # Extraer la región de la imagen y redimensionarla al tamaño deseado
                resized_region, coord = CellDetector.extract_region_rescale(img, bbox, standard_size=cell_img_size, min_scale=min_scale)

                # Calcular la puntuación de confianza como el área de la caja delimitadora
                scores.append(bbox[-2] * bbox[-1])

                # Almacenar las coordenadas de la nueva caja delimitadora
                boxes.append(coord)

        return boxes, scores


    @staticmethod
    def plot_detected_cells(img, selected_boxes):
        output = img.copy()
        for (xmin, ymin, xmax, ymax) in selected_boxes:
            # Dibujar la caja delimitadora en la imagen de salida
            cv2.rectangle(output, (xmin, ymin), (xmax,ymax), (0, 255, 0), 1)

        # Mostrar la imagen de salida con las cajas delimitadoras dibujadas
        #cv2_imshow(output)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)  # Convertir a RGB
        #cv2.imwrite(os.path.join(output_folder, f'celula_{num_cell}.png'), output)
        #cv2.imshow('Celulas detectadas', output)
        #cv2.waitKey(0)  # Esperar para presionar una tecla
        #cv2.destroyAllWindows()  # Cerrar todas las ventadnas
        plt.imshow(output)
        plt.axis("off")
        plt.show()


    @staticmethod
    def save_detected_cells(img, selected_boxes, filepath='celulas_detectadas.jpeg'):
        output = img.copy()
        for (xmin, ymin, xmax, ymax) in selected_boxes:
            # Dibujar la caja delimitadora en la imagen de salida
            cv2.rectangle(output, (xmin, ymin), (xmax,ymax), (0, 255, 0), 1)

        # Guardar la imagen de salida con las cajas delimitadoras dibujadas
        #output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)  # Convertir a RGB
        #plt.imshow(output)
        cv2.imwrite(filepath, output)


    @staticmethod
    def non_max_suppression(boxes, scores, iou_threshold=0.5):
        if len(boxes) == 0:
            return []

        # Convertir a numpy
        boxes = np.array(boxes)
        scores = np.array(scores)

        # Calcular las áreas de todos los cuadros delimitadores
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Ordenar cuadros por puntuaciones en orden descendente
        order = np.argsort(scores)[::-1]

        keep = []  # Lista para almacenar índices de cuadros seleccionados

        while len(order) > 0:
            i = order[0]  # Índice de cuadro con mayor puntuación
            keep.append(i)

            # Calcular el IoU entre el cuadro seleccionado y todos los cuadros restantes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union

            # Suprimir cuadros con IoU mayor que el umbral
            order = order[np.where(iou <= iou_threshold)[0] + 1]

        # Seleccionar las cajas delimitadoras que pasaron la NMS
        selected_boxes = [boxes[i] for i in keep]

        return selected_boxes

    @staticmethod
    def extract_regions_interest(img, selected_boxes, cell_size=None):
        # Extraer las regiones de interés después de aplicar la NMS
        extracted_regions = []

        # Iterar sobre las cajas seleccionadas
        for box in selected_boxes:
            # Obtener las coordenadas de la caja delimitadora
            xmin, ymin, xmax, ymax = box

            # Extraer la región de la imagen original
            region = img[ymin:ymax, xmin:xmax]

            # Redimensionar la región al tamaño estándar
            if cell_size:
                region = cv2.resize(region, cell_size)

            # Almacenar la región redimensionada
            extracted_regions.append(region)

        return extracted_regions

    @staticmethod
    def detect_cells(image, alpha=1.2, beta=20,
                    kernel_size_blur=(3,3), kernel_size_morph=(3,3),
                    distance_threshold=0.1, cell_bbox_size = (50, 50),
                    iou_threshold=None):
        
        binarized = CellDetector.preprocess_image(image, alpha, beta, kernel_size_blur)
        markers = CellDetector.get_markers(image, binarized, kernel_size_morph, distance_threshold)
        boxes, scores = CellDetector.get_bounding_boxes_from_markers(image, markers, cell_bbox_size)
        print(f'Se detectaron {len(boxes)} células')

        # Non-Maximum Suppression (NMS)
        if iou_threshold:
            selected_boxes = CellDetector.non_max_suppression(boxes, scores, iou_threshold)
            print(f'Se detectaron {len(selected_boxes)} células luego de aplicar NMS')
            return selected_boxes
        return boxes


def generar_nombre_carpeta(base_path):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder_name = f"celulas_{timestamp}"
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

if __name__ == '__main__':
    alpha=1.2
    beta=20
    kernel_size_blur=(3,3)
    kernel_size_morph=(3,3)
    distance_threshold=0.1
    cell_bbox_size = (60, 60)
    iou_threshold = 0.3

    # Leer imagen
    image_path = 'C:\\Users\\Usuario\\Desktop\\CITOLOGIA 1.jpg'
    image = cv2.imread(image_path)

    # Detectar celulas
    selected_boxes = CellDetector.detect_cells(image, alpha, beta, kernel_size_blur,
                                  kernel_size_morph, distance_threshold,
                                  cell_bbox_size, iou_threshold)


    CellDetector.plot_detected_cells(image, selected_boxes)

    # Guardar cuadros delimitadores en carpeta
    cell_output_size = (224, 224)
    extracted_regions = CellDetector.extract_regions_interest(image, selected_boxes, cell_output_size)
    #extracted_regions = CellDetector.extract_regions_interest(image, selected_boxes)

    #output_folder = 'uploads/segregadas/prueba/'
    #output_folder = 'C:\\Users\\Usuario\\Desktop\\celulas_segregadas\\prueba'
    #os.makedirs(output_folder, exist_ok=True)

    #detected_filepath = os.path.join(output_folder, f'celulas_detectadas.jpeg')
    #CellDetector.save_detected_cells(image, selected_boxes, detected_filepath)

    # Guardar imágenes en carpeta generada
    #folder_name = generar_nombre_carpeta('uploads/segregadas')
    folder_name = generar_nombre_carpeta('C:\\Users\\Usuario\\Desktop\\celulas_segregadas')
    print(folder_name)
    for num_cell, img_region in enumerate(extracted_regions):
        cv2.imwrite(os.path.join(folder_name, f'celula_{num_cell}.jpeg'), img_region)