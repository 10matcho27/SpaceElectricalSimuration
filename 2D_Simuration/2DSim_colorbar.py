#png→Gif
#https://photocombine.net/gifanime/

import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation

def init_stat(t, alfa, τ0):
    num = math.exp(-1.0 * (alfa * (t - τ0)) ** 2.0)
    return num

def calc_Ez(i, j, Ez_t1, cv, dt, dx, dy, By_t2, Bx_t2):
    return Ez_t1[i][j] + (cv**2) * dt * ((By_t2[i][j] - By_t2[i][j-1]) / dx - (Bx_t2[i][j] - Bx_t2[i-1][j]) / dy)

def calc_By(i, j, By_t1, Ez_t2, dt, dx):
    return By_t1[i][j] + (dt/dx) * (Ez_t2[i][j+1] - Ez_t2[i][j])

def calc_Bx(i, j, Bx_t1, Ez_t2, dt, dy):
    return Bx_t1[i][j] - (dt/dy) * (Ez_t2[i+1][j] - Ez_t2[i][j])

def main():
    Nx, Ny, cv, dx, dy =300, 300, 5.0, 2.0, 2.0
    dt = dx / 10.0    #dt < dx
    τ0 = dt * 400.0 #τ0 >> dt
    alfa = (2.0/τ0)
    total_step = int((dx*Nx)/(cv*dt)) * 2

    #setup for Ez, By, Bx
    Ez_t1, Ez_t2, By_t1, By_t2, Bx_t1, Bx_t2 = \
        [[0 for _ in range(Nx)] for __ in range(Nx)], [[0 for _ in range(Nx)] for __ in range(Nx)],\
            [[0 for _ in range(Nx)] for __ in range(Nx)], [[0 for _ in range(Nx)] for __ in range(Nx)],\
                [[0 for _ in range(Nx)] for __ in range(Nx)], [[0 for _ in range(Nx)] for __ in range(Nx)]
    Ez_t1, Ez_t2, By_t1, By_t2, Bx_t1, Bx_t2 = \
        np.array(Ez_t1), np.array(Ez_t2), np.array(By_t1), np.array(By_t2), np.array(Bx_t1), np.array(Bx_t2)
    Ez_t1, Ez_t2, By_t1, By_t2, Bx_t2, Bx_t2 = \
        np.float_(Ez_t1), np.float_(Ez_t2), np.float_(By_t1), np.float_(By_t2), np.float_(Bx_t1), np.float_(Bx_t2)

    fig, ax = plt.subplots()
    image_list = []

    for step_num in tqdm(range(total_step)): #loop of time x,y方向
        Ez_t1[1][1] = init_stat(step_num, alfa, τ0) #格子番号1のEzに初期値(時間関数)を与える
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                Ez_t2[i][j] = calc_Ez(i, j, Ez_t1, cv, dt, dx, dy, By_t2, Bx_t2)

        for j in range(Nx):
            Ez_t2[0][i] = 0
            Ez_t2[Ny-1][i] = 0
        for i in range(Ny):
            Ez_t2[i][0] = 0
            Ez_t2[i][Nx-1] = 0

        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                By_t2[i][j] = calc_By(i, j, By_t1, Ez_t2, dt, dx)

        for j in range(1, Nx-1):
            for i in range(1, Ny-1):
                Bx_t2[i][j] = calc_Bx(i, j, Bx_t1, Ez_t2, dt, dy)

        im = plt.imshow(Ez_t1, animated=True, vmin=0, vmax=0.1)
        if(step_num == 0):
            bar = fig.colorbar(im, orientation="vertical")
            bar.mappable.set_clim(0,0.1)
        image_list.append([im])

        Ez_t1 = Ez_t2
        By_t1 = By_t2
        Bx_t1 = Bx_t2
        plt.savefig("./fig_colorbar/fig" + str(step_num) + ".png")

    ani = animation.ArtistAnimation(fig, image_list, interval=1)
    ani.save("test2_colorbar.gif", writer="pillow")
    plt.show()

if __name__ == '__main__':
    main()