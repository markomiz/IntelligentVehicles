from scipy.spatial import cKDTree
from math import * 
from SingleTrack import *
import numpy as np
from graph import *
import scipy.interpolate as interp
from matplotlib import pyplot as plt

class RRT:
    def __init__(self, step_size, star=False, geometric=False):
        self.start = np.zeros(3)
        self.goal = np.array([25.0,25.0,0.0])
        self.step_size = step_size
        self.car = SingleTrackVehicleModel()
        self.car.velocity = 1.0
        self.min = -1.0
        self.max = 30.0
        self.graph = Graph()
        self.graph.add_vertex(self.start)
        self.star = star
        self.end_node = None
        self.end_cost = np.inf
        self.geometric = geometric

    def steer_geom(self,near, rand):
        line = rand - near
        unit = line / np.linalg.norm(line[:2])
        pos = near + unit * self.step_size
        pos[2] = 0
        return pos


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

    def rewire(self, node): 
        nearest_set = self.graph.get_near(node.pose, self.step_size*self.car.velocity * 2)
        # rewire
        for n in nearest_set:
            if n == node: continue
            q = self.steer(n.pose, node.pose)
            if q is None: continue
            if np.linalg.norm(node.pose - n.pose) > self.car.velocity * self.step_size: continue
            if node.cost + 1 < n.cost:
                self.graph.add_edge(node, n, 1.0)
                print("rewire")

    def rewire_geom(self,node):
        nearest_set = self.graph.get_near(node.pose, self.step_size*2)
        # rewire
        for n in nearest_set:
            if n == node: continue
            dist = np.linalg.norm(node.pose[:2] - n.pose[:2])
            if dist > self.step_size * 2: continue
            cost_inc = dist/self.step_size
            if node.cost + cost_inc < n.cost:
                self.graph.add_edge(node, n, cost_inc)
                # print("rewire")

    def get_random_point(self):
        new = np.random.uniform(self.min, self.max, 3)
        return new

    def colliding(self,p):
        if p[0] > 5 and p[0] < 20:
            if p[1] > 5 and p[1] < 20:
                return True 
        return False

    def planning(self, max_steps=1000):
        for i in range(max_steps):
            rnd = self.get_random_point()
            nearest = self.graph.get_near(rnd)
            closest_d = np.inf
            new_pose = None
            near_node = None
            for near in nearest:
                if self.geometric: p = self.steer_geom(near.pose, rnd)
                else: p = self.steer(near.pose, rnd)
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
            self.graph.add_edge(near_node, new_node, 1.0)
            if self.star:
                if self.geometric: self.rewire_geom(new_node)
                else: self.rewire(new_node)
            if self.is_goal(new_node):
                print("found it in: ", i )
                if not self.star:
                    return np.array(self.graph.get_path(new_node))
        if self.end_node is None: print("DID NOT SUCCEED : (")
        return np.array(self.graph.get_path(self.end_node))
        

    def is_goal(self, point):
        d = self.goal - point.pose
        dist = np.linalg.norm(d)
        if dist < 1.0:
            if point.cost < self.end_cost:
                self.end_node = point
                self.end_cost = point.cost
                print("replaced end node")
            return True
        return False

    def interpolate_polyline(self, polyline, num_points):
        duplicates = []
        for i in range(1, len(polyline)):
            if np.allclose(polyline[i], polyline[i-1]):
                duplicates.append(i)
        if duplicates:
            polyline = np.delete(polyline, duplicates, axis=0)
        tck, u = interp.splprep(polyline.T, s=10)
        u = np.linspace(0.0, 1.0, num_points)
        return np.column_stack(interp.splev(u, tck))

if __name__ == "__main__":
    planner = RRT(0.5, True, False)
    res = planner.planning(100000)
    plt.plot(res[:,0], res[:,1], label="Raw ")
    res = planner.interpolate_polyline(res[:,:2], 1000)
    plt.plot(res[:,0], res[:,1], label="Interpolated")
    plt.legend()
    
    plt.show()






    
