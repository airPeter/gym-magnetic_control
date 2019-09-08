import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import math
import numpy as np

class MagControlEnv(gym.Env):
    '''
    Description:
        at a square area, four magnetic dipoles are fixed at four courner. One paramagnetic ball can
        move inside the area without friction. The goal is to adjust the force of fixed dipoles exerted on
        the ball to make the ball stay at the target point.

    Observation:
        Type: Box(4)
        num     Observation         Min     Max
        0       ball position_x     -5      5
        1       ball position_y     -5      5
        2       ball velocity_x     -Inf    Inf
        3       ball velocity_y     -Inf    Inf
        4       target position_x   -5      5
        5       target position_y   -5      5

    Actions: change magetic momentum of the fixed dipoles.
        Type: Box(4)
        Num    action           
        0      increase m1y       
        1      increase m2y       
        2      increase m3y        
        3      increase m4y        
        4      decrease m1y       
        5      decrease m2y         
        6      decrease m3y     
        7      decrease m4y
        8       doing nothing       
    Reward:
        Reward is -L.  L is the distance between ball and target point. 
        Target point is randomly initialized for every section.

    Episode Termination:
        the ball position is more than 4.
        solved requirements
        consider solved when the average reward is close to zero over 100 consecutive trials
    '''


    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    
    def __init__(self):
        self.x0 = [-4.5,4.5,-4.5,4.5]
        self.y0 = [-4.5,-4.5,4.5,4.5]
        self.dipoles = None
        self.massball = 1.0*1e-9
        self.m_ball = 1.0 # magnetic momentum
        self.tau = 0.02 # secondes between state updates
        #self.target_postion = [0,0] #(x,y)
        self.threshold = 4
        self.step_size_m = 0.01
        high = np.array([5, 5, np.finfo(np.float32).max, np.finfo(np.float32).max, 5, 5])

        self.observation_space = spaces.Box(-high,high, dtype = np.float32)
        self.action_space = spaces.Discrete(9)
        self.magnetic_momentum = None

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
            args: action is a number from 0 to 7
            0 1 2 3 means increase
            4 5 6 7means decrease
            8 means doning nothing
        '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, y, vx, vy, tx, ty = state
        if action <=3:
            self.magnetic_momentum[action] = self.magnetic_momentum[action]+self.step_size_m
        elif action <=7:
            self.magnetic_momentum[action] = self.magnetic_momentum[action]-self.step_size_m

        mx = np.zeros(4) # this direction is currently set to zero.
        my = self.magnetic_momentum
        self.dipoles = magnetic_dipoles(self.x0, self.y0, mx, my)
        Fx, Fy = self.dipoles.force_on_paramagnetic(x,y,self.m_ball)
        ax = Fx/self.massball
        ay = Fy/self.massball

        # update:
        vx = vx+ax*self.tau
        vy = vy+ay*self.tau
        x =x+vx*self.tau+ax/2.0*self.tau**2
        y =y+vy*self.tau+ay/2.0*self.tau**2
        
        self.state = (x,y,vx,vy,tx,ty)

        done = x<-self.threshold or x>self.threshold or y<-self.threshold or y>self.threshold
        done = bool(done)
        # distance:
        L = np.sqrt((x-tx)**2+(y-ty)**2)
        if not done:
            reward = -L 
        elif self.steps_beyond_done is None:
            # ball rolls out of boundary
            self.steps_beyond_done = 0
            reward = -L
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        #x, y, vx, vy = self.np_random.uniform(low=-0.05,high=0.05, size=(4,))
        x, y, vx, vy = 0,0,0,0
        self.steps_beyond_done = None
        tx, ty = self.np_random.uniform(low=-self.threshold, high=self.threshold, size=(2,))
        self.state = (x,y,vx,vy,tx,ty)
        self.magnetic_momentum = np.zeros(4)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500
        world_width = 10
        scale = screen_width/world_width
        ball_radius = scale*0.4
        cx, cy = screen_width/2.0, screen_height/2.0
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.ball = rendering.make_circle(ball_radius)
            self.balltrans = rendering.Transform()
            self.ball.add_attr(self.balltrans)
            self.ball.set_color(.5,.5,.8)
            self.viewer.add_geom(self.ball)

            self.fixed_ball1 = rendering.make_circle(ball_radius)
            self.fixed_ball1trans = rendering.Transform(translation=(self.x0[0]*scale+cx,self.y0[0]*scale+cy))
            self.fixed_ball1.add_attr(self.fixed_ball1trans)
            self.fixed_ball1.set_color(.5,.5,.8)
            self.viewer.add_geom(self.fixed_ball1)

            self.fixed_ball2 = rendering.make_circle(ball_radius)
            self.fixed_ball2trans = rendering.Transform(translation=(self.x0[1]*scale+cx,self.y0[1]*scale+cy))
            self.fixed_ball2.add_attr(self.fixed_ball2trans)
            self.fixed_ball2.set_color(.5,.5,.8)
            self.viewer.add_geom(self.fixed_ball2)

            self.fixed_ball3 = rendering.make_circle(ball_radius)
            self.fixed_ball3trans = rendering.Transform(translation=(self.x0[2]*scale+cx,self.y0[2]*scale+cy))
            self.fixed_ball3.add_attr(self.fixed_ball3trans)
            self.fixed_ball3.set_color(.5,.5,.8)
            self.viewer.add_geom(self.fixed_ball3)

            self.fixed_ball4 = rendering.make_circle(ball_radius)
            self.fixed_ball4trans = rendering.Transform(translation=(self.x0[3]*scale+cx,self.y0[3]*scale+cy))
            self.fixed_ball4.add_attr(self.fixed_ball4trans)
            self.fixed_ball4.set_color(.5,.5,.8)
            self.viewer.add_geom(self.fixed_ball4)

            self.target_circle= rendering.make_circle(ball_radius/4)
            self.target_circletrans = rendering.Transform(translation=(self.state[4]*scale+cx,self.state[5]*scale+cy))
            self.target_circle.add_attr(self.target_circletrans)
            self.target_circle.set_color(.5,.0,.0)
            self.viewer.add_geom(self.target_circle)

        if self.state is None: return None

        #edit the ball
        ball = self.ball
        points_on_circle = []
        for i in range(30):
            ang = 2*np.pi*i/30
            points_on_circle.append((np.cos(ang)*ball_radius,np.sin(ang)*ball_radius))
        ball.v = points_on_circle

        x = self.state
        ballx = x[0]*scale+screen_width/2.0
        bally = x[1]*scale+screen_height/2.0
        self.balltrans.set_translation(ballx, bally)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')




    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

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
        field = np.array([self.magnetic_field(x,y,self.x0[i],self.y0[i],self.mx[i],self.my[i]) for i in range(len(self.x0))])
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
        if B == 0:
            Fx = 0
            Fy = 0
        else:
            Fx = m/B*(Bx*dBx_dx+By*dBy_dx)
            Fy = m/B*(Bx*dBx_dy+By*dBy_dy)
        return Fx, Fy
