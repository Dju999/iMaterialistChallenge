"""
    Модуль для детекции изображений
"""
import os

import matplotlib.pyplot as plt

plt.switch_backend('agg')
from object_detection.utils import label_map_util


class ObjectDetector:

    def __init__(self, dest_dir):
        # ссылки на различные модели тут: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
        self.PATH_TO_LABELS = os.path.join(dest_dir, 'models/research/object_detection/data', 'mscoco_label_map.pbtxt')
        self.dest_dir = dest_dir
        self.init_category()

    def download_model(self, MODEL_NAME='faster_rcnn_resnet101_coco_2018_01_28'):
        """Скачиваем и распаковываем модель Object Detection Zoo"""
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        self.PATH_TO_CKPT = os.path.join(self.dest_dir, MODEL_NAME) + '/frozen_inference_graph.pb'
        # скачиваем модель
        urllib.request.urlretrieve(DOWNLOAD_BASE + MODEL_FILE, os.path.join(self.dest_dir, MODEL_FILE))
        tar_file = tarfile.open(os.path.join(self.dest_dir, MODEL_FILE))
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, self.dest_dir)

    def init_category(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def _prepare_img_arrays(self):
        img_dir = os.path.join(self.dest_dir, 'raw_img_data')
        img_array = []
        for batch_num in np.unique([i['batch'] for i in self.image_dir_decription])[:10]:
            for filename in [i['filename'] for i in self.image_dir_decription if i['batch'] == batch_num]:
                image = PIL_IMAGE.open(os.path.join(img_dir, filename))
                (img_width, img_height) = image.size
                image_np = np.array(image.getdata()).reshape((img_width, img_height, 3)).astype(np.uint8)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                img_array.append({'img_name': filename, 'img_array': image_np_expanded})
        print("Длина массива с рекомендациями %s" % len(img_array))
        return img_array

    def object_detection(self, image_dir_decription):
        """Запускаем детектор картинок"""
        self.image_dir_decription = image_dir_decription
        categories, probabilities = [], []
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        img_detections = []
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                for image in self._prepare_img_arrays():
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image['img_array']})
                    for index, value in enumerate(classes[0]):
                        img_detections.append({
                            'category_name': self.category_index.get(value)['name'],
                            'category_id': self.category_index.get(value)['id'],
                            'category_proba': scores[0, index],
                            'category_box': boxes[0, index],
                            'img_array': image['img_array'],
                            'img_name': image['img_array']
                        })
        # возвращаем результат
        self.img_detections = img_detections

    def crop_bounding_box(self, img_name):
        """Делаем crop изображения"""
        # TODO: избавиться от сканирования списка, перейти к словарю
        img_meta = [i for i in self.img_detections if i['img_name'] == img_name][0]
        image_np = img_meta['img_array'].copy()
        # vis_util.draw_bounding_boxes_on_image_array(image_np, boxes)
        # переходим от относительных координта к абсолютным
        im_width, im_height = image_np.shape[:2]
        ymin, xmin, ymax, xmax = img_meta['category_box']
        (xminn, xmaxx, yminn, ymaxx) = (
        int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
        cropped_image = tf.image.crop_to_bounding_box(image_np, yminn, xminn, ymaxx - yminn, xmaxx - xminn)
        with tf.Session() as sess:
            image_np = cropped_image.eval()
        return image_np
