import numpy as np
import cv2
import axi
from threading import Thread
from collections import deque

def upcontrast(img):
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def upcontrast2(img):
    x = img/255.0 - 0.5
    x = 1.0 / (1.0 + np.exp(-x))
    x = (x * 255.0).astype(np.uint8)
    return x

def renormalize(img):
    x = img
    mnx = np.min(x).astype(np.float32)
    mxx = np.max(x).astype(np.float32)
    x = (x - mnx)/(mxx - mnx) * 255.0
    return x.astype(img.dtype)

class AHSGui(object):
    def __init__(self):

        # AxiDraw
        try:
            self.p_axi_ = np.load('axi_parameters.npy').item()
        except Exception as e:
            self.p_axi_ = dict(
                    steps_per_unit = axi.device.STEPS_PER_INCH,
                    pen_up_position = axi.device.PEN_UP_POSITION,
                    pen_up_speed = axi.device.PEN_UP_SPEED,
                    pen_up_delay = axi.device.PEN_UP_DELAY,
                    pen_down_position = axi.device.PEN_DOWN_POSITION,
                    pen_down_speed = axi.device.PEN_DOWN_SPEED,
                    pen_down_delay = axi.device.PEN_DOWN_DELAY,
                    acceleration = axi.device.ACCELERATION,
                    max_velocity = axi.device.MAX_VELOCITY,
                    corner_factor = axi.device.CORNER_FACTOR,
                    jog_acceleration = axi.device.JOG_ACCELERATION,
                    jog_max_velocity = axi.device.JOG_MAX_VELOCITY
                    )
        print self.p_axi_

        self.axi_ = axi.Device(**self.p_axi_)
        self.axi_.enable_motors()
        self.axi_.pen_up()
        self.axi_.home()

        # Camera
        self.params_ = np.load('camera_parameters.npz')
        self.align_  = np.load('m.npy')
        self.cam_ = cv2.VideoCapture(0)
        self.img_ = None

        # Drawing Data
        self.q_ = []#deque(maxlen=128)
        self.t_ex_ = None
        self.data_ = {
                'drw' : False, # drawing
                'pts' : [],
                'up'  : False,
                'down': False
                }
        self.drw_ = np.full(
                shape=(334,437,3),
                fill_value=255,
                dtype=np.uint8)

    def p2w(self, x, y):
        x = np.clip(x, 0, 437.0)
        y = np.clip(y, 0, 334.0)
        w_x = 11.0 - (x * (11.0 / 437.0))
        w_y = y * (8.5 / 334.0)
        return w_x, w_y

    def goto(self, x, y):
        w_x, w_y = self.p2w(x, y)
        self.axi_.goto(w_x, w_y)

    def _move_cb(self, x, y):
        try:
            self.goto(x, y)
            cv2.line(self.drw_,
                    self.data_['pts'][-1], (x,y),
                    (0,0,0), 2)
            self.data_['pts'].append( (x, y) )
            cv2.circle(self.drw_, (x,y), 2, (0,0,0), -1)
        except Exception as e:
            print('Exception while moving : {}'.format(e))

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.t_ex_ is not None:
                self.t_ex_.join()
                self.t_ex_ = None
            self.q_ = []
            self.data_['drw'] = True
            self.data_['pts'] = [(x, y)]
            self.data_['up']  = False
            self.axi_.pen_up()
            self.data_['down'] = True # set  down-request
            self.q_.append((x,y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.data_['drw'] = False
            self.data_['up'] = True # set up-request
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.data_['drw']:
                self.q_.append((x,y))

    def key_cb(self, k):
        # TODO : handle pen-height calibration, etc.
        if k in [27, ord('q')]:
            return False

        # pen up-down
        if k in [ord('u')]:
            self.axi_.pen_up()
        if k in [ord('d')]:
            self.axi_.pen_down()

        # home
        if k in [ord('h')]:
            self.axi_.home()
        
        # tweak params
        if k in [ord('l')]: # lower
            self.p_axi_['pen_down_position'] -= 1
            self.axi_.pen_down_position = self.p_axi_['pen_down_position']
            self.axi_.configure()
        if k in [ord('r')]: # raise
            self.p_axi_['pen_down_position'] += 1
            self.axi_.pen_down_position = self.p_axi_['pen_down_position']
            self.axi_.configure()

        return True

    def get_frame(self):
        # unroll
        cam = self.cam_
        params = self.params_
        if self.img_ is None:
            ret, img = cam.read()
            self.img_ = img
        else:
            ret, _ = cam.read(self.img_)
            img = self.img_

        img = cv2.undistort(img,
                params.get('arr_0'),
                params.get('arr_1'),
                newCameraMatrix=params.get('arr_2'))
        img = img[92:92+334, 73:73+437]
        img = upcontrast(img)
        # flip l-r
        img = img[:, ::-1]
        return img
    
    def run(self):
        cv2.namedWindow('window', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
        cv2.setMouseCallback('window', self.mouse_cb)

        while True:
            # real-life feedback
            frame = self.get_frame()
            #warp  = frame
            warp  = cv2.warpPerspective(frame, self.align_,
                    tuple(self.drw_.shape[:2][::-1]))
            #viz = np.concatenate([self.drw_, frame], axis=1)
            #viz = cv2.addWeighted(self.drw_, 0.5,
            #        warp, 0.5, 0.0)
            viz = cv2.addWeighted(warp, 0.8,
                    self.drw_, 0.2, 0.0)

            cv2.imshow('window', viz)
            k = cv2.waitKey(1)
            if not self.key_cb(k):
                break

            # handle mouse-move queue
            if self.t_ex_ is not None:
                if not self.t_ex_.is_alive():
                    self.t_ex_.join()
                    self.t_ex_ = None

            if self.t_ex_ is None and len(self.q_) > 0:
                prox = cv2.approxPolyDP(np.float32(self.q_), 4.0, False)
                prox = np.reshape(prox, (-1, 2))
                self.q_ = list(prox)
                t = Thread(
                        target=self._move_cb,
                        args=self.q_.pop(0)
                        )
                t.start()
                self.t_ex_ = t

            if self.data_['down']:
                if self.t_ex_ is not None:
                    self.t_ex_.join()
                    self.t_ex_ = None
                self.data_['down'] = False
                self.axi_.pen_down()

            if self.data_['up'] and len(self.q_) <= 0:
                # done; reset position
                if self.t_ex_ is not None:
                    self.t_ex_.join()
                    self.t_ex_ = None
                self.data_['up'] = False
                self.axi_.pen_up()
                #self.axi_.home()

        cv2.destroyAllWindows()

        # reset position
        self.axi_.pen_up()
        self.axi_.home()
        self.axi_.disable_motors()

        np.save('axi_parameters.npy', self.p_axi_)

        # save images
        cv2.imwrite('/tmp/drw.png', self.drw_)
        cv2.imwrite('/tmp/cam.png', self.img_)

def main():
    app = AHSGui()
    app.run()

if __name__ == '__main__':
    main()
