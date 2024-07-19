import os
from glob import glob
from sklearn.model_selection import train_test_split


name = 'busi'
root = r'./data/' + name

img_ids = glob(os.path.join(root, 'images', '*.png'))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

count = 1
for i in [1, 2, 3]:
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.3, random_state=i)
    filename = root + '/train_{}.txt'.format(count)
    with open(filename, 'w') as file:
        for i in train_img_ids:
            file.write(i + '\n')

    filename = root + '/val_{}.txt'.format(count)
    with open(filename, 'w') as file:
        for i in val_img_ids:
            file.writelines(i + '\n')

    count += 1
