"""
    Модуль для скачивания данных с Kaggle
"""
import os
import urllib3
import multiprocessing


from tqdm import tqdm
from urllib3.util import Retry
from IPython.display import Image
import zipfile
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


class FS:
    def __init__(self):
        # 1. Authenticate and create the PyDrive client.
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def zipdir(self, path, ziph):
        """Сжимаем все файлы в директории и добавляем их в архив"""
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file))

    def load_to_drive(self, file_name):
        """загружаем на GoogleDrive"""
        file1 = self.drive.CreateFile({'title': file_name})
        file1.SetContentFile(os.path.join(working_dir, file_name))
        file1.Upload()

    def make_zip(self, dir_name):
        """создаём архив

        :param str dir_name: полный путь до директории, которую хотим сжать
        """
        zip_file = zipfile.ZipFile(dir_name + '.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir(dir_name, zip_file)
        zip_file.close()

        load_to_drive(dir_name + '.zip')

    def drive_file_id(self, filename):
        """Проверка файла на существование"""
        file_list = self.drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        for file1 in file_list:
            if file1['title'] == filename:
                return file1['id']
        return None

    def load_from_drive(self, dest_dir, filename, drive_file_id=None):
        """Загружаем файл с Гугл-диска, если он отсутствует локально"""
        if drive_file_id is None:
            drive_file_id = self.drive_file_id(filename)
        # если файл есть на гугл драйв - скачиваем оттуда
        if drive_file_id is not None:
            drive_file = self.drive.CreateFile({'id': drive_file_id})
            drive_file.GetContentFile(os.path.join(dest_dir, filename))
        print("загрузка {} завершена".format(filename))

    def unzip_file(self, root_dir, file_path):
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(root_dir)
        zip_ref.close()
        print("Распаковали %s в %s" % (file_path, root_dir))

    def multithread_img_downloader(self, img_urls):
        """Загрузка данных в несколько потоков"""

        def image_downloader(img_info):
            """
              download image and save its with 90% quality as JPG format
              skip image downloading if image already exists at given path
              :param fnames_and_urls: tuple containing absolute path and url of image
            """
            fname, url = img_info
            if not os.path.exists(fname):
                http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
                response = http.request("GET", url)
                image = Image.open(io.BytesIO(response.data))
                image_rgb = image.convert("RGB")
                image_rgb.save(fname, format='JPEG', quality=90)
            return None

        # download data
        pool = multiprocessing.Pool(processes=12)
        with tqdm(total=len(img_urls)) as progress_bar:
            for _ in pool.imap_unordered(image_downloader, img_urls):
                progress_bar.update(1)
        print('all images loaded!')
