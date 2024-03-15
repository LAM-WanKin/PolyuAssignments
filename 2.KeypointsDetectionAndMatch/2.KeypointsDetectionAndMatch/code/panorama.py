import cv2
import numpy as np
from tqdm import tqdm
from sift import keypoint_match, draw_match, transform
from utils import read_video_frames, write_and_show, destroyAllWindows, imshow

video_name = 'image/winter_day.mov'
images, fps = read_video_frames(video_name)
n_image = len(images)

# TODO: init panorama
h,w=images[0].shape[0],images[0].shape[1]
H=h
W=w*7 # need wide enough
h_start= H-h
w_start= W-w
panorama = np.zeros([H,W,3])
panorama[h_start:h_start+h,w_start:w_start+w,0:3]=images[0]

trans_sum = np.zeros([H,W,3])
cnt = np.ones([H,W,1])*1e-10
for img in tqdm(images[::4], 'processing'):
    # TODO: stitch img to panorama one by one
    key1,key2,match = keypoint_match(panorama,img,max_n_match=1000,draw=False)
    key1 = np.array([key1[m.queryIdx].pt for m in match])
    key2 = np.array([key2[m.trainIdx].pt for m in match])
    trans_img = transform(img,key2,key1,H,W)

    trans_sum += trans_img
    cnt += (trans_img != 0).any(2, keepdims=True)
    panorama = trans_sum / cnt

    # show
    # imshow('results/2_panorama.jpg', panorama)
    # write_and_show('results/2_panorama.jpg', panorama)


# panorama = algined.mean(0)
write_and_show('results/2_panorama.jpg', panorama)

destroyAllWindows();
