from PIL import Image
import random

with open("HW1_dataset/train.txt", 'r') as f:
    for img in f:
        image = Image.open('HW1_dataset/images/' + img.strip())

        flipped = False
        if random.uniform(0, 1) <= 0.3:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped = True

        labels = []
        label_file_name = img.strip().split('.')[0] + '.txt'
        with open('HW1_dataset/labels/' + label_file_name, 'r') as l:
            for obj in l:
                obj = obj.strip()
                class_index, xc, yc, w, h = obj.split()
                if flipped:
                    class_index = int(class_index) - 1 if int(class_index) % 2 else int(class_index) + 1  # Switching RH and LH classifications
                labels.append(f"{class_index} {1 - float(xc) if flipped else xc} {yc} {w} {h}")

        image.save('HW1_dataset/images/train/' + img.strip())
        with open('HW1_dataset/labels/train/' + label_file_name, 'w') as l:
            l.write("\n".join(labels))

with open("HW1_dataset/valid.txt", 'r') as f:
    for img in f:
        image = Image.open('HW1_dataset/images/' + img.strip())

        label_file_name = img.strip().split('.')[0] + '.txt'
        with open('HW1_dataset/labels/' + label_file_name, 'r') as l:
            labels = l.read()

        image.save('HW1_dataset/images/val/' + img.strip())
        with open('HW1_dataset/labels/val/' + label_file_name, 'w') as l:
            l.write(labels)

with open("HW1_dataset/test.txt", 'r') as f:
    for img in f:
        image = Image.open('HW1_dataset/images/' + img.strip())

        label_file_name = img.strip().split('.')[0] + '.txt'
        with open('HW1_dataset/labels/' + label_file_name, 'r') as l:
            labels = l.read()

        image.save('HW1_dataset/images/test/' + img.strip())
        with open('HW1_dataset/labels/test/' + label_file_name, 'w') as l:
            l.write(labels)
