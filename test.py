import axi
import numpy as np

def uv2xy(uv, s=1.0):
    u, v = uv[...,0], uv[...,1]
    x, y = u / s, v / s
    return np.stack([x,y], axis=-1)

def main():
    dev = axi.Device(
            pen_down_position=25
            )
    # coord specification = (right(+x), down(+y))
    #path = [
    #        (0,0),
    #        (1,1),
    #        (1,2),
    #        (2,1),
    #        (1,1)
    #        ]
    dev.enable_motors()
    dev.pen_up()
    dev.home()
    dev.pen_down()
    dev.goto(1, 2)
    #dev.run_path(path)
    dev.pen_up()
    dev.home()
    dev.disable_motors()

if __name__ == '__main__':
    main()
