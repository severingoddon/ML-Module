import os
import csv
import skimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm

image_list = []
i = 1
for image in os.listdir('/Users/sevi/Desktop/Docs/PycharmProjects/MLModule/PVA1/newImages'):
    image_path = '/Users/sevi/Desktop/Docs/PycharmProjects/MLModule/PVA1/newImages/' + image
    img = skimage.io.imread(image_path)
    img_gr = skimage.io.imread(image_path, as_gray=True)
    img_re = skimage.transform.resize(img_gr, (10, 10))
    imgplot = plt.imshow(img_re, cmap=cm.Greys_r)
    finalImage = [image]
    for x in img_re:
        for y in x:
            print(y)
            if 100*y>0.1: # normalize data
                finalImage.append(1)
            else:
                finalImage.append(0)
    image_list.append(finalImage)
    print('created ' + str(i / 50 * 100) + '%')
    i += 1

header = ['filename']
for i in range(100):
    header.append('Pixel ' + str(i + 1))

with open('singleImage.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(image_list)
