Install:
    first have gym installed, them use command: pip install -e . 
    
An env for 2d paramagnetic_ball_control. Four magnets are at four corners. By increasing or decreasing the magnetic momentum, a paramagnetic_ball on the ground can be moved at target place.
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
