import numpy as np


class magnetic_dipoles():

    def __init__(self,x0,y0,mx,my):

        self.x0 = x0 #postion of magnetic dipoles, can be a array.
        self.y0 = y0
        self.mx = mx #momentum can be a array, must have same length with x0.
        self.my =my
        self.mu = 1.2566*1e-6 #N/A^2
    def magnetic_field(self,x,y,x0,y0,mx,my):
        #given one magnetic dipole at (x0,y0) with momentum (mx,my), generate magnetic_field at position (x,y)
        rx = x - x0
        ry = y - y0
        r = np.sqrt(rx**2+ry**2)
        rx_unit = rx/r
        ry_unit = ry/r
        r_m = rx_unit*mx+ry_unit*my
        Bx = self.mu/(4*np.pi*r**3)*(4*rx_unit*r_m-mx)
        By = self.mu/(4*np.pi*r**3)*(4*ry_unit*r_m-my)
        return [Bx, By]
    def dipoles_field(self,x,y):
        # given magnetic dipoles at x0,y0(array of position) return superposed field.
        field = np.array([self.magnetic_field(x,y,x0[i],y0[i],mx[i],my[i]) for i in range(len(x0))])
        return np.sum(field,axis = 0)
    def derivative_field(self,x,y):
        dx = 1e-6
        dy = 1e-6
        dBx_dx = (self.dipoles_field(x+dx,y)[0]-self.dipoles_field(x-dx,y)[0])/(2*dx)
        dBy_dx = (self.dipoles_field(x+dx,y)[1]-self.dipoles_field(x-dx,y)[1])/(2*dx)
        dBx_dy = (self.dipoles_field(x,y+dy)[0]-self.dipoles_field(x,y-dy)[0])/(2*dy)
        dBy_dy = (self.dipoles_field(x,y+dy)[1]-self.dipoles_field(x,y-dy)[1])/(2*dy)
        return dBx_dx, dBx_dy, dBy_dx, dBy_dy
    def force_on_constant_dipole(self,x,y,mx,my):
        # the magnetic momentum of the object is independent of magnetic field.
        dBx_dx, dBx_dy, dBy_dx, dBy_dy = self.derivative_field(x,y)
        Fx = mx*dBx_dx+my*dBy_dx
        Fy = mx*dBx_dy+my*dBy_dy
        return Fx, Fy
    def force_on_paramagnetic(self,x,y,m):
        # the magnetic momentum of the object has same direction with magnetic field.
        dBx_dx, dBx_dy, dBy_dx, dBy_dy = self.derivative_field(x,y)
        field = self.dipoles_field(x,y)
        Bx = field[0]
        By = field[1]
        B = np.sqrt(Bx**2+By**2)
        Fx = m/B*(Bx*dBx_dx+By*dBy_dx)
        Fy = m/B*(Bx*dBx_dy+By*dBy_dy)
        return Fx, Fy
    