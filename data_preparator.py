"""
    Модуль для подготовки данных
"""
import os
from subprocess import call
import json

import shutil


class DataPreparator:
    """Работа с данными: создаём структуру директорий, скачиваем из интернетов"""

    def __init__(self, root_dir, api_token, fs):
        self.root_dir = root_dir
        self.api_token = api_token
        # Поля, которые будут инициализированы в процессе работы
        self.img_urls = None
        self.challenge_data = None
        self.img_dir_name = os.path.join(self.root_dir, 'raw_img_data')
        self.fs = fs

    def prepare_data(self, challenge):
        self.prepare_working_dir()
        self.download_challenge_data(challenge)
        self.get_url_from_challenge_data('train')
        self.get_images()

    def prepare_working_dir(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        # сохраняем конфиг с токеном - директория .kaggle будет создана автоматически
        kaggle_utils_dir = os.path.join(self.root_dir, '.kaggle')
        if not os.path.exists(kaggle_utils_dir):
            os.mkdir(kaggle_utils_dir)
        with open(os.path.join(kaggle_utils_dir, 'kaggle.json'), 'w') as file:
            json.dump(self.api_token, file)
        # устанавливаем права на  файл с конфигом
        call('chmod 600 /content/.kaggle/kaggle.json', shell=True)
        call('kaggle config set --name path --value \/content', shell=True)

    def download_challenge_data(self, challenge):
        # директория competitions создаётся утилитой kaggle автоматически
        self.competition_dir = os.path.join(self.root_dir, 'competitions', challenge)
        if os.path.exists(self.competition_dir) and len(os.listdir(self.competition_dir)) > 0:
            self.read_challenge_data()
            return
        # скачиваем набор данных для соревнования
        call('kaggle competitions download -c %s' % challenge, shell=True)
        for file in os.listdir(self.competition_dir):
            if file[-3:] == 'zip':
                self.fs.unzip_file(self.root_dir, os.path.join(self.competition_dir, file))

        print('Downloaded files: %s into %s' % (sorted(os.listdir(self.competition_dir)), self.competition_dir))
        self.read_challenge_data()
        self.get_url_from_challenge_data('train')

    def image_dir_description(self, dir_name, batch_size=50):
        """Формируем JSON c файлами внутри директории - включая разбиение по батчам"""
        # формируем список изображений
        file_list = [i for i in os.listdir(dir_name) if i[-3:] == 'jpg']
        batches = (np.arange(len(file_list)) / batch_size).astype(np.uint16)
        return [
            {
                'filename': file_list[file_num],
                'batch': batches[file_num],
                'id': file_num
            }
            for file_num in np.arange(batches.size)
        ]

    def get_images(self):
        # проверяем, есть ли в директории какие-то данные
        if os.path.exists(self.img_dir_name) and len(os.listdir(self.img_dir_name)) > 0:
            return
        # проверяем, есть ли файл c архивом фоток на google_drive
        zip_filename = 'raw_img_data.zip'
        drive_file_id = self.fs.drive_file_id(zip_filename)
        if drive_file_id is not None:
            # скачиваем файл из облака только если заранее не скачали
            if not os.path.exists(os.path.join(self.root_dir, zip_filename)):
                drive_file = self.fs.load_from_drive(self.root_dir, zip_filename, drive_file_id)
            self.fs.unzip_file(self.root_dir, zip_filename)
            # эта строчка тут только потому, что файл изначально плохо сжали =[
            shutil.move('/content/content/competitions/imaterialist-challenge-fashion-2018/raw_img_data',
                        '/content/raw_img_data')
        else:
            # если файла с архивом на диске нет - загружаем фотки из интернетов
            self.fs.multithread_img_downloader(self.img_urls)
            # загружаем на диск архив с фотками (500 мб)
            self.fs.make_zip('raw_img_data')
            # исходные данные тоже пригодятся
            self.fs.load_to_drive('train.json.zip')
            self.fs.load_to_drive('test.json.zip')
            self.fs.load_to_drive('validation.json.zip')
        print("Загрузили %s изображений" % len(os.listdir(self.img_dir_name)))

    def read_challenge_data(self):
        """Читаем данные json из рабочей директории"""
        train_path = os.path.join(self.root_dir, 'train.json')
        test_path = os.path.join(self.root_dir, 'test.json')
        valid_path = os.path.join(self.root_dir, 'validation.json')

        train_inp = open(train_path).read()
        train_inp = json.loads(train_inp)

        test_inp = open(test_path).read()
        test_inp = json.loads(test_inp)

        valid_inp = open(valid_path).read()
        valid_inp = json.loads(valid_inp)
        self.challenge_data = {
            'train': train_inp, 'test': test_inp, 'validation': valid_inp
        }

    def get_url_from_challenge_data(self, dataset, _max=10000):
        """
            parse the dataset to create a list of tuple containing absolute path and url of image
            :param _dataset: dataset to parse
            :param _outdir: output directory where data will be saved
            :param _max: maximum images to download (change to download all dataset)
        :return: list of tuple containing absolute path and url of image
        """
        _fnames_urls = []
        data = self.challenge_data[dataset]
        for image in data["images"]:
            url = image["url"]
            fname = os.path.join(self.img_dir_name, "%s.jpg" % image["imageId"])
            _fnames_urls.append((fname, url))
        self.img_urls = _fnames_urls[:_max]
