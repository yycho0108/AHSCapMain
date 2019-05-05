import cv2
import numpy as np
from utils import order_points, four_point_transform

class AlignGUI(object):
    def __init__(self):
        self.win_ = cv2.namedWindow('win')
        self.img_ = cv2.imread('ref4.png')
        cv2.setMouseCallback('win', self.mouse_cb)

        self.src_ = []
        self.dst_ = [] 
        self.m0_  = np.load('m.npy')
        np.save('m0.npy', self.m0_)
        #self.src_ = [(6, 19), (368, 34), (381, 296), (11, 273)]
        #self.dst_ = [(23, 18), (380, 24), (400, 293), (33, 275)]
        #self.src_ = [(9, 46), (247, 64), (234, 206), (4, 209)]
        #self.dst_ = [(25, 38), (256, 52), (245, 199), (21, 202)]

    def align(self):
        print self.src_
        print self.dst_

        src = np.float32(self.src_)
        dst = np.float32(self.dst_)
        M = four_point_transform(self.img_, src, dst)
        np.save('m.npy', M * self.m0_)

    def key_cb(self, k):
        # TODO : handle pen-height calibration, etc.
        if k in [27, ord('q')]:
            return False

        if k in [ord('a')]:
            self.align()

        return True

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.src_.append( (x,y) )
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.dst_.append( (x,y) )

    def run(self):
        while True:
            frame = self.img_

            cv2.imshow('win', self.img_)
            k = cv2.waitKey(1)
            if not self.key_cb(k):
                break
        cv2.destroyAllWindows()

def main():
    app = AlignGUI()
    app.run()

if __name__ == '__main__':
    main()
