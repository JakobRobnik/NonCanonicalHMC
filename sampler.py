import numpy as np
from numba import njit  #I use numba, which makes the python evaluations significantly faster (as if it was a c code). If you do not want to download numba comment out the @njit decorators.

from scipy.stats import special_ortho_group

#from numba import jitclass          # import the decorator
#from numba import int32, float32    # import the types


#@jitclass
class Sampler():

    #@njit
    def trajectory(self, x0, p0, dt, steps):
        X = np.zeros((steps, len(x0)))
        P = np.zeros((steps, len(p0)))
        X[0], P[0] = x0, p0
        E = self.hamiltonian(X[0], P[0])
        for i in range(steps-1):
            X[i+1], P[i+1] = self.step(X[i], P[i], E, dt)

        return X, P

    #@njit
    def time_evolve(self, x0, p0, E, dt, steps):
        x, p = x0, p0
        for i in range(steps-1):
            x, p = self.step(x, p, E, dt)
        return x, p

    #@njit
    def billiard_evolve(self, x0, p0, E, num_free_steps, rot):
        dt = 0.00001
        x, p = x0, p0
        X = np.empty((len(rot)*num_free_steps, len(x0)))
        #X = np.empty((len(rot), len(x0)))
        for k in range(len(rot)): #number of bounces
            # bounce
            p = np.dot(rot[k], p)

            #evolve
            for i in range(num_free_steps):
                for j in range(100):
                    x, p = self.step(x, p, E, dt)

                X[k*len(rot)+i, :] = x

            #X[k, :] = x

        return X


    def sample(self, num_free_steps, num_bounces):
        x0, p0 = np.ones(self.d), np.ones(self.d)

        E = self.hamiltonian(x0, p0)
        rot = np.array([special_ortho_group.rvs(len(x0)) for k in range(num_bounces)])  #generate random rotations

        history = self.billiard_evolve(x0, p0, E, num_free_steps, rot)

        return history


#
# spec = [
#     ('d', int32),               # a simple scalar field
#     ('array', float32[:]),          # an array field
# ]



class Ruthless(Sampler):

    def __init__(self, negative_log_p, grad_negative_log_p, d, step):
        """Args:
             negative_log_p: - log p of the target distribution
             d: dimension of x"""

        self.nlogp = negative_log_p
        self.grad_nlogp = grad_negative_log_p
        self.d = d

        if step == 'Euler':
            self.step = self.symplectic_Euler_step


        if step == 'leapfrog':
            self.step = self.leapfrog


    #@njit
    def g(self, x):
        """inverse mass"""
        return np.exp(2 * self.nlogp(x) / self.d)

    #@njit
    def grad_g(self, x):
        """returns g and it's gradient"""
        gg = np.exp(2 * self.nlogp(x) / self.d)
        return gg, (2 * gg / self.d) * self.grad_nlogp(x)

    #@njit
    def hamiltonian(self, x, p):
        """"H = g(x) p^2"""
        return self.g(x) * np.sum(np.square(p))

    #@njit
    def symplectic_Euler_step(self, x, p, E, dt):
        gg, Dg = self.grad_g(x)
        pnew = p - dt * Dg * E / gg
        xnew = x + dt * 2 * gg * pnew
        return xnew, pnew

    #@njit
    def leapfrog(self, x, v, E, dt):
        gg, Dg = self.grad_g(x)
        #iterate to find vn
        vn, vnew = np.copy(v), np.copy(v)
        for i in range(30):
            vnew = v + 0.5*dt*(-E *Dg + np.dot(vn, Dg) * vn / gg)
            if np.sum(np.abs(vnew - vn)) < 1e-6:
                break
            else:
                vn = vnew

        vnew = 2 *vnew - v
        xnew = x + vnew * dt

        return xnew, vnew



class BI(Sampler):

    def __init__(self, negative_log_p, grad_negative_log_p, d):
        """Args:
             negative_log_p: - log p of the target distribution
             d: dimension of x"""

        self.nlogp = negative_log_p
        self.grad_nlogp = grad_negative_log_p
        self.d = d

    @njit
    def c_sq(self, x):
        """squared speed of light"""
        return np.exp(2 * self.nlogp(x) / self.d)

    @njit
    def grad_c_sq(self, x):
        """returns c^2 and it's gradient"""
        gg = np.exp(2 * self.nlogp(x) / self.d)
        return gg, (2 * gg / self.d) * self.grad_nlogp(x)

    @njit
    def hamiltonian(self, x, p):
        """"H = g(x) p^2"""
        cc = self.c_sq(x)
        return np.sqrt(cc(x) ** 2 + cc * np.sum(np.square(p)))


    @njit
    def symplectic_Euler_step(self, x, p, E, dt):
        cc, Dcc = self.grad_g(x)
        pnew = p - dt * 0.5 * Dcc * (E / (cc + 1e-8) + cc / E)
        xnew = x + dt * cc * pnew / E
        return xnew, pnew


class Canonical(Sampler):

    def __init__(self, negative_log_p, grad_negative_log_p, d, step):
        """Args:
             negative_log_p: - log p of the target distribution
             d: dimension of x"""

        self.nlogp = negative_log_p
        self.grad_nlogp = grad_negative_log_p
        self.d = d

        if step == 'Euler':
            self.step = self.symplectic_Euler_step

        if step == 'Yoshida':
            self.step = self.Yoshida_step

    #@njit
    def V(self, x):
        """potential"""
        return -np.exp(-2 * self.nlogp(x) / (self.d-2))

    #@njit
    def grad_V(self, x):
        """returns g and it's gradient"""
        v = -np.exp(-2 * self.nlogp(x) / (self.d-2))

        return v, (-2 * v / (self.d-2)) * self.grad_nlogp(x)

    #@njit
    def hamiltonian(self, x, p):
        """"H = g(x) p^2"""
        return 0.5* np.sum(np.square(p)) + self.V(x)

    #@njit
    def symplectic_Euler_step(self, x, p, E, dt):
        v, Dv = self.grad_V(x)
        pnew = p - dt * Dv
        xnew = x + dt * pnew
        return xnew, pnew


    def Yoshida_step(self, x0, p0, dt):
        x = np.copy(x0)
        p = np.copy(p0)
        cbrt_two = np.cbrt(2)
        w0, w1 = cbrt_two / (2 - cbrt_two), 1.0 / (2 - cbrt_two)
        c = [0.5*w1, 0.5*(w0+w1), 0.5*(w0+w1)]
        d = [w1, w0, w1]
        for i in range(3):
            x += c[i] * p * dt
            p -= d[i] * self.grad_V(x) * dt

        x += c[0] * p * dt

        return x, p

