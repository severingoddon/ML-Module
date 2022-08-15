import os
import csv
import skimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm

image_list = []
i = 1
for image in os.listdir('C:\\Users\\sever\\Desktop\\Documents\\PycharmProjects\\MLMODULE\\PVA1\\images'):
    image_path = 'C:\\Users\\sever\\Desktop\\Documents\\PycharmProjects\\MLMODULE\\PVA1\\images\\' + image
    img = skimage.io.imread(image_path)
    img_gr = skimage.io.imread(image_path, as_gray=True)
    img_re = skimage.transform.resize(img_gr, (10, 10))
    imgplot = plt.imshow(img_re, cmap=cm.Greys_r)
    finalImage = [image]
    for x in img_re:
        for y in x:
            finalImage.append(y)
    image_list.append(finalImage)
    print('created ' + str(i / 50 * 100) + '%')
    i += 1

header = ['filename']
for i in range(100):
    header.append('Pixel ' + str(i + 1))

with open('symbole.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(image_list)
