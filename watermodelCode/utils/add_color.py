import cv2
import numpy as np
def mask_color_img(img, mask, color=[255, 0, 255], alpha=0.7):
    '''
    在img上的mask位置上画color
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
    alpha: float [0, 1].
    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = img.copy()  # 深拷贝，重新开辟内存，操作不影响img，也算浅拷贝？

    if mask.shape[:2] != img.shape[:2]:
        mask = np.rot90(mask)
    # print('shape0:',np.shape(img_layer))
    # print('shape1:', np.shape(mask))
    img_layer[mask] = color
    # plt.imshow(img_layer)
    # plt.show()
    out = cv2.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)  # 0是gamma修正系数，不需要修正设置为0，out输出图像保存位置
    return out.astype(np.uint8)