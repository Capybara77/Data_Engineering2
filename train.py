import numpy as np
from pathlib import Path
from shutil import copyfile
import os
import xml.etree.ElementTree as ET
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Список классов, которые мы будем использовать для аннотации
classes = ["helmet", "head", "person"]

# Функция для преобразования аннотации в формат, который мы будем использовать для обучения модели
def convert_annot(size, box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    dw = np.float32(1.0 / int(size[0]))
    dh = np.float32(1.0 / int(size[1]))

    w = x2 - x1
    h = y2 - y1
    x = x1 + (w / 2)
    y = y1 + (h / 2)

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]

# Функция для сохранения аннотаций в формате txt
def save_txt_file(img_jpg_file_name, size, img_box):
    save_file_name = "/Dataset/labels/" + img_jpg_file_name + ".txt"

    with open(save_file_name, "a+") as file_path:
        for box in img_box:
            cls_num = classes.index(box[0])
            new_box = convert_annot(size, box[1:])
            file_path.write(
                f"{cls_num} {new_box[0]} {new_box[1]} {new_box[2]} {new_box[3]}\n"
            )

# Функция для получения аннотаций из xml-файла
def get_xml_data(file_path, img_xml_file):
    img_path = file_path + "/" + img_xml_file + ".xml"

    tree = ET.parse(img_path)
    root = tree.getroot()

    img_size = root.find("size")
    img_w = int(img_size.find("width").text)
    img_h = int(img_size.find("height").text)

    img_box = []
    for box in root.findall("object"):
        cls_name = box.find("name").text
        x1 = int(box.find("bndbox").find("xmin").text)
        y1 = int(box.find("bndbox").find("ymin").text)
        x2 = int(box.find("bndbox").find("xmax").text)
        y2 = int(box.find("bndbox").find("ymax").text)

        img_box.append([cls_name, x1, y1, x2, y2])

    save_txt_file(img_xml_file, [img_w, img_h], img_box)

# Получение списка xml-файлов из директории /annotations
files = os.listdir("/annotations")
for file in tqdm(files, total=len(files)):
    file_xml = file.split(".")
    get_xml_data("/annotations/", file_xml[0])

# Разделение списка изображений на тренировочный, валидационный и тестовый наборы
image_list = os.listdir("/annotations")
train_list, test_list = train_test_split(
    image_list, test_size=0.2, random_state=42)
val_list, test_list = train_test_split(
    test_list, test_size=0.5, random_state=42)


# Вывод информации о размерах наборов данных
print("total =", len(image_list))
print("train :", len(train_list))
print("val   :", len(val_list))
print("test  :", len(test_list))


def copy_data(file_list, img_labels_root, imgs_source, mode):
    dataset_root = Path("/Dataset/images/")

    images_path = dataset_root / "images" / mode
    labels_path = dataset_root / "labels" / mode

    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    for file in tqdm(file_list, desc=f"Copying {mode} data"):
        base_filename = file.replace(".png", "")

        img_src_file = Path(imgs_source) / (base_filename + ".png")
        label_src_file = Path(img_labels_root) / (base_filename + ".txt")

        img_dest_file = images_path / (base_filename + ".png")
        label_dest_file = labels_path / (base_filename + ".txt")

        copyfile(img_src_file, img_dest_file)
        copyfile(label_src_file, label_dest_file)


copy_data(train_list, "/Dataset/labels", "/Dataset/images", "train")
copy_data(val_list, "/Dataset/labels", "/Dataset/images", "val")
copy_data(test_list, "/Dataset/labels", "/Dataset/images", "test")

config = {
    "train": "/Dataset/images/train",
    "val": "/Dataset/images/val",
    "test": "/Dataset/images/test",
    "nc": 3,
    "names": ["helmet", "head", "person"],
}

with open("data.yaml", "w") as file:
    yaml.dump(config, file, default_flow_style=False)
