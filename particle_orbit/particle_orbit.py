import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class Particle():
    def __init__(self, pos = [0.0, 0.0, 0.0], v = [0.0, 0.0, 0.0], mass = 1.0, q = 1):
        self.pos = np.array(pos)
        self.v = np.array(v)
        self.mass = mass
        self.q = q #正の電荷
        self.x_all, self.y_all, self.z_all, self.energy_all = [], [], [], []

    def set(self, pos, v, mass, q):
        self.pos = pos
        self.v = v
        self.mass = mass
        self.q = q

    def append_pos(self, x, y, z):
        self.x_all.append(x)
        self.y_all.append(y)
        self.z_all.append(z)

    def force(self, v, E, B):
        # return self.q / self.mass *(E + np.cross(v, B))
        return \
            np.array([self.q / self.mass * (E[0] + v[1]*B[2] - v[2]*B[1]), \
                    self.q / self.mass * (E[1] + v[2]*B[0] - v[0]*B[2]), \
                    self.q / self.mass * (E[2] + v[0]*B[1] - v[1]*B[0])])

    def simply_update(self, dt, E, B):
        self.v = self.v + dt * self.force(self.v, E, B)
        self.pos = dt * self.v
        self.energy()

    def runge_kutta_update(self, dt, E, B):
        k1 = dt * self.force(self.v, E, B)
        k2 = dt * self.force((self.v + k1/2), E, B)
        k3 = dt * self.force((self.v + k2/2), E, B)
        k4 = dt * self.force((self.v + k3), E, B)

        k = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        self.v = self.v + k
        self.pos = dt * self.v
        self.energy()

    def energy(self):
        self.energy_all.append((1/2) * self.mass * (self.v)**2)

if __name__ == '__main__':
    E = np.array([0, 0, 0])
    B = np.array([0, 0, 1])
    time_step = 1000
    dt = 1 / (50)
    t = np.arange(0, time_step * dt, dt)
    init_pos = np.array([0, 0, 0])

    #generate particle
    p0, p1 = Particle(), Particle()
    p0.set(init_pos, np.array([1, 0, 0]), 1, 1)
    p1.set(init_pos, np.array([1, 0, 0]), 1, 1)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # ax = fig.add_subplot()
    # plt.title("Particle orbit")
    fig.suptitle("Particle orbit\n[Ex, Ey, Ez]=["+str(E[0])+","+str(E[1])+","+str(E[2])+"]\n[Bx, By, Bz]=["+str(B[0])+","+str(B[1])+","+str(B[2])+"]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid()
    ax.view_init(elev=80, azim=45)
    ax.set_aspect('auto')
    image_list = []

    for t_ in tqdm(range(0, time_step)):
        # if(t_ == 0):
        #     print("\n", "init position :", "\n", p0.pos ,"\n", "init v :", "\n", p0.v)
        p0.simply_update(dt, E, B)
        p1.runge_kutta_update(dt, E, B)

        p0.append_pos(p0.pos[0], p0.pos[1], p0.pos[2])
        p1.append_pos(p1.pos[0], p1.pos[1], p1.pos[2])

        im1 = plt.plot(p0.pos[0], p0.pos[1], p0.pos[2], 'o', color = 'red', label="Euler")
        im1_line = plt.plot(p0.x_all, p0.y_all, p0.z_all, '--', color="red")
        im2 = plt.plot(p1.pos[0], p1.pos[1], p1.pos[2], '*', color = "blue", label="Runge-Kutta")
        im2_line = plt.plot(p1.x_all, p1.y_all, p1.z_all, '', color = "blue")

        if(t_ == 0):
            ax.legend(loc=2)
            ax.set_xlim([-0.03, 0.03])
            ax.set_ylim([-0.03, 0.03])
            ax.set_zlim([0, 1])

        image_list.append(im1 + im2 + im1_line + im2_line)

    ani = animation.ArtistAnimation(fig, image_list, interval=5)
    plt.show()
    # ani.save("orbit_comp_3D.gif", writer='pillow')
    ani.save("orbit_comp_3D_0.mp4", writer='ffmpeg')