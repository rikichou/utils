import os
import cv2

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

def save_lists_to_txtfile(infos, file_path):
    with open(file_path, 'w') as fp:
        for info in infos:
            strinfo = [str(x) for x in info]
            fp.write(';'.join(strinfo)+'\n')

def read_lists_from_txtfile(file_path):
    infos = []
    with open(file_path, 'r') as fp:
        for line in fp.readlines():
            info = [float(x) for x in line.strip().split(';')]
            infos.append(info)
    return infos

def show_facerect_with_txtfile(image_path, save_path=False):
    # get infos from txt file
    txtfile_path = image_path.rsplit('.', maxsplit=1)[0]+'.facerect'
    infos = read_lists_from_txtfile(txtfile_path)

    # show
    image = cv2.imread(image_path)
    for info in infos:
        showinfo = [int(x) for x in info]
        cv2.rectangle(image, (showinfo[0], showinfo[1]), (showinfo[2], showinfo[3]), (0,0,255), 1)
    if save_path:
        cv2.imwrite(save_path, image)
    else:
        cv2.imshow(image)