from scipy.spatial import cKDTree
from math import * 
from SingleTrack import *
import numpy as np
from graph import *
import scipy.interpolate as interp
from matplotlib import pyplot as plt

class RRTStar:
    def __init__(self, step_size):
        self.start = np.zeros(3)
        self.goal = np.array([25.0,25.0,0.0])
        self.step_size = step_size
        self.car = SingleTrackVehicleModel()
        self.car.velocity = 2.0
        self.min = -1.0
        self.max = 30.0
        self.graph = Graph()
        self.graph.add_vertex(self.start)
        self.end_cost = np.inf
        self.end_node = None

    def steer(self, near, rand):
        line = rand - near
        heading = atan2(line[1], line[0])
        self.car.x = near[0]
        self.car.y = near[1]
        self.car.theta = near[2]
        steer_angle = heading - self.car.theta
        improve = True
        cost = 0
        c_pos = near * 1
        distance = np.linalg.norm(line)
        while improve:
            steer_angle = heading - self.car.theta
            if abs(steer_angle) > self.car.max_steer:
                steer_angle = self.car.max_steer * steer_angle/abs(steer_angle)
            self.car.update(self.step_size, min(steer_angle, self.car.max_steer), 0)
            # print("steering ", distance)
            cost +=1
            c_pos = self.car.get_pos_as_vec()
            new_distance = np.linalg.norm(c_pos[:2] - near[:2])
            # improve = False
            if new_distance > distance or self.colliding(c_pos):
                improve = False
            distance = new_distance * 1
        
        return c_pos , cost

    # def steer(self, near, rand):
    #     line = -(near - rand) * 1.0
    #     line[2] = 0
    #     distance = np.linalg.norm(line)
    #     unit = line * float(1.0/ float(distance))
    #     improve = True
    #     cost = 0
    #     c_pos = near * 1
    #     while improve:
    #     # while cost < 1:
    #         # print("steering ", distance)
    #         cost +=1
    #         c_pos += unit * self.step_size
    #         new_distance = np.linalg.norm(c_pos - near)
    #         # improve = False
    #         if new_distance > distance or self.colliding(c_pos):
    #             improve = False
    #         distance = new_distance * 1
    #     return c_pos, cost


    def get_random_point(self):
        new = np.random.uniform(self.min, self.max, 3)
        while self.colliding(new):
            new = np.random.uniform(self.min, self.max, 3)
        return new

    def is_goal(self, point):
        d = self.goal - point.pose
        
        dist = np.linalg.norm(d)
        if dist < 1.0:
            if point.cost < self.end_cost:
                self.end_node = point
                self.end_cost = point.cost
            return True
        return False

    def colliding(self,p):
        if p[0] > 5 and p[0] < 20:
            if p[1] > 5 and p[1] < 20:
                return True 
        return False

    def planning(self, max_steps=1000):
        for i in range(max_steps):
            rnd = self.get_random_point()
            nearest, _  = self.graph.get_single_near(rnd)
            p, cost = self.steer(nearest.pose, rnd)
            if (self.colliding(p)): continue
            new_node = self.graph.add_vertex(p)
            self.graph.add_edge(nearest, new_node, cost)
            nearest_set = self.graph.get_near(rnd, self.step_size*self.car.velocity * 2)
            # rewire
            for n in nearest_set:
                if n == new_node: continue
                q, kost = self.steer(n.pose, new_node.pose)
                if np.linalg.norm(q[:2] - p[:2]) > 0.5: continue
                if new_node.cost + kost < n.cost:
                    self.graph.add_edge(new_node, n, kost)

            if self.is_goal(new_node):
                print("found it in: ", i )
        print("done")
        return np.array(self.graph.get_path(self.end_node))
        print("DID NOT SUCCEED : (")

    

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

    def planning(self, max_steps=1000):
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
    planner = RRTStar(0.5)
    res = planner.planning(2000)
    plt.plot(res[:,0], res[:,1], label="Raw Geometric RRT*")
    res = planner.interpolate_polyline(res[:,:2], 1000)
    plt.plot(res[:,0], res[:,1], label="Interpolated")
    plt.legend()
    
    plt.show()






    
