from scipy.spatial import cKDTree
from math import * 
from SingleTrack import *
import numpy as np
from graph import *

from matplotlib import pyplot as plt

class RRT:
    def __init__(self, step_size):
        self.start = np.zeros(3)
        self.goal = np.array([25.0,25.0,0.0])
        self.step_size = step_size
        self.car = SingleTrackVehicleModel()
        self.car.velocity = 1.0
        self.min = -1.0
        self.max = 30.0
        self.graph = Graph()
        self.graph.add_vertex(self.start)

    def steer(self, near, rand):
        line = rand - near
        heading = atan2(line[1], line[0])
        self.car.x = near[0]
        self.car.y = near[1]
        self.car.theta = near[2]
        steer_angle = heading - self.car.theta
        if abs(steer_angle) > self.car.max_steer:
            steer_angle = self.car.max_steer * steer_angle/abs(steer_angle)
        self.car.update(self.step_size, min(steer_angle, self.car.max_steer), 0)
        return self.car.get_pos_as_vec()


    def get_random_point(self):
        new = np.random.uniform(self.min, self.max, 3)
        return new

    def is_goal(self, point):
        d = self.goal - point
        d[2] = 0
        dist = np.linalg.norm(d)
        
        if dist < 1.0:
            print("distance to goal ", dist)
            print(point)
            
            return True
        return False

    def colliding(self,p):
        if p[0] > 5 and p[0] < 20:
            if p[1] > 5 and p[1] < 20:
                return True 
        return False

    def planning(self, max_steps=10000):
        for i in range(max_steps):
            rnd = self.get_random_point()
            nearest = self.graph.get_near(rnd)
            closest_d = np.inf
            new_pose = None
            near_node = None
            for near in nearest:
                p = self.steer(near.pose, rnd)
                if (self.colliding(p)): continue
                d = p - rnd
                d[2] = 0
                dist = np.linalg.norm(d) 
                if dist < closest_d:
                    closest_d = dist
                    new_pose = p
                    near_node = near
            if near_node == None: continue 
            new_node = self.graph.add_vertex(new_pose)
            self.graph.add_edge(near_node, new_node)
            if self.is_goal(p):
                print("found it in: ", i )
                return np.array(self.graph.get_path(new_node))
        print("DID NOT SUCCEED : (")


if __name__ == "__main__":
    planner = RRT(0.5)
    res = planner.planning()
    plt.plot(res[:,0], res[:,1])
    plt.show()






    
