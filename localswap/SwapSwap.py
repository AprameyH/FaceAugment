import cv2
import matplotlib.pyplot as plt

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640,640))

swapper = insightface.model_zoo.get_model('../models/inswapper_128.onnx', download=False, download_zip=False)

# main_img: File Path to AI generated background image (with one person)
# face_image: File path to mugshot/celebrity face
def swapswap(main_img, face_image, show_steps=False):
    img2 = cv2.imread(main_img)
    
    if show_steps:
        plt.imshow(img2[:,:,::-1])
        plt.title('Background AI image')
        plt.show()

    img_face = cv2.imread(face_image)
    if show_steps:
        plt.imshow(img_face[:,:,::-1])
        plt.title('Mugshot image')
        plt.show()

    face_out = app.get(img2)[0]

    face_in = app.get(img_face)[0]

    # bbox = face_out['bbox']
    # bbox = [int(b) for b in bbox]
    
    # if show_steps:
    #     plt.imshow(img2[bbox[1]:bbox[3],bbox[0]:bbox[2], ::-1])
    #     plt.axis('off')
    #     plt.show()

    # bbox = face_in['bbox']
    # bbox = [int(b) for b in bbox]
    # if show_steps:
    #     plt.imshow(img_face[bbox[1]:bbox[3],bbox[0]:bbox[2], ::-1])
    #     plt.axis('off')
    #     plt.show()

    res2 = img2.copy()

    res2 = swapper.get(res2, face_out, face_in, paste_back=True)

    return res2



# res2 = swapswap('../images/test1.jpeg', '../images/test1-real.png')

# plt.imshow(res2[:,:,::-1])
# plt.show()

res2 = swapswap('../images/bp-out.jpeg', '../images/pitt.png', True)
plt.title("Swapped image")
plt.imshow(res2[:,:,::-1])
plt.show()