import os
import shutil
from PIL import Image

# Путь до директории с исходными данными
data_dir = 'G:\dataset'

# Путь до директории для сохранения "сломанных" файлов
broken_dir = os.path.join(data_dir, 'broken')
if not os.path.exists(broken_dir):
    os.mkdir(broken_dir)

# Проход по всем поддиректориям и файлам внутри
for root, dirs, files in os.walk(data_dir):
    for file in files:
        # Получение полного пути к файлу
        file_path = os.path.join(root, file)
        try:
            # Попытка открыть изображение
            img = Image.open(file_path)
            width, height = img.size
            # Если ширина или высота меньше или равна 180 и 240 соответственно
            if width <= 180 or height <= 240:
                # Вывод информации о найденном файле
                print(f"Найден файл {file_path} с размерами {width}x{height}")
                # Перемещение файла в директорию broken
                shutil.move(file_path, os.path.join(broken_dir, file))
                os.remove(file_path)
        except:
            # Если файл не является изображением или не удается его открыть, пропускаем его
            pass
