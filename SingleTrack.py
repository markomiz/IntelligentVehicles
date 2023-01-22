from math import *
from PIL import Image, ImageDraw

from time import sleep
import numpy as np

import torch
from RRT import *

import matplotlib.pyplot as plt

class Simulator:
    def __init__(self):
        self.car = SingleTrackVehicleModel()
        self.p2m = 20 # pixel to meter
        self.images = []
        self.dt = 0.1 # s
        self.img_w = 1000
        self.img_h = 1000
        self.state = torch.zeros(4)
        self.nu = 25
        self.spot_center = np.zeros(2)
        # self.create_spot()
        self.last_cost = 0.0

    def step(self, action):
        self.process_action(action)
        self.car.update(self.dt, self.steer, self.acc)
        
        c = self.calculate_reward()
        self.last_cost = c
        self.draw()
        # terminal = self.out_of_bound()
        terminal = False
        if c < 0.001: terminal = True
        return self.state, c, terminal, 0, 0

    def draw(self):
        image = Image.new("RGB", (self.img_w, self.img_h), (100, 100, 100))
        # Draw a rectangle on the image
        draw = ImageDraw.Draw(image)
        W = self.car.W * self.p2m
        L = self.car.L * self.p2m
        # define the angle at which to draw the rectangle (in degrees)
        x_c = 10  + self.car.x * self.p2m
        y_c = 10 + self.img_h - self.car.y * self.p2m # flip because images have flipped y
        c_th = - self.car.theta # flip because images have flipped y
        x0 = x_c - L/2 * cos(c_th) - W/2 * sin(c_th)
        y0 = y_c - L/2 * sin(c_th) + W/2 * cos(c_th)
        x1 = x_c - L/2 * cos(c_th) + W/2 * sin(c_th)
        y1 = y_c - L/2 * sin(c_th) - W/2 * cos(c_th)
        x2 = x_c + L/2 * cos(c_th) + W/2 * sin(c_th)
        y2 = y_c + L/2 * sin(c_th) - W/2 * cos(c_th)
        x3 = x_c + L/2 * cos(c_th) - W/2 * sin(c_th)
        y3 = y_c + L/2 * sin(c_th) + W/2 * cos(c_th)

        # draw the rectangle
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # # draw spot
        # draw.polygon(self.spot, fill=(50, 200, 50))

        # cost_points = [(1,1), (int(self.last_cost* self.img_h), 1)]
 
        # draw.line(cost_points, fill ="red", width = 6)

        self.images.append(image)
    
    def save_gif(self, name=""):
        self.images[0].save('car'+ name +'.gif',
        save_all=True, append_images=self.images[1:], loop=0)

    def create_spot(self):
        c = np.random.randint(self.img_h/6 , self.img_h/2)
        h = int(self.car.W * self.p2m * 0.55)
        w = int(self.car.L * self.p2m * 0.55)
        p0 = (self.img_h/2+w,c+h)
        p1 = (self.img_h/2+w,c-h)
        p2 = (self.img_h/2-w,c-h)
        p3 = (self.img_h/2-w,c+h)
        self.spot = [p0,p1,p2, p3]
        self.spot_center[0] = self.car.y
        self.spot_center[1] = -(float(c) - self.img_h/2) / self.p2m

    def process_action(self,action):
        steer = -float(int(action / 5)) * self.car.max_steer / 2 + self.car.max_steer
        acc = - float(int(action % 5)) * self.car.max_acc/ 2  + self.car.max_acc
        # print("steer  " , steer)
        # print("acc  " , acc)
        self.steer = steer
        self.acc = acc

    def reset(self):
        self.car.__init__()
        self.create_spot()
        self.draw()
        self.images = []
        return self.state

    def update_state(self):
        self.state[0] = self.car.x - self.spot_center[0]
        self.state[1] = self.car.y - self.spot_center[1]
        self.state[2] = self.car.theta
        self.state[3] = self.car.velocity

    def calculate_reward(self):
        x_dist = (self.car.x - self.spot_center[0]) **2
        y_dist = (self.car.y - self.spot_center[1]) **2
        th_dist = self.car.theta ** 2
        v_dist = self.car.velocity**2
        c = x_dist + y_dist + 0.1*th_dist + 0.1* v_dist

        return c /1000

    def out_of_bound(self):
        if abs(self.car.x * self.p2m) > self.img_h/2 \
        or abs(self.car.y * self.p2m) > self.img_h/2:
            return True
        else: return False
        


class SingleTrackVehicleModel:
    def __init__(self):
        self.img_w = 1000
        self.img_h = 1000
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.velocity = 0.0
        self.max_velocity = 10.0
        self.steering_angle = 0.0
        self.L =  3.0 # wheelbase
        self.max_steer = pi / 4.0
        self.W = 1.5
        self.max_acc = 3.0
        self.p2m = 30 # pixel to meter
        self.images = []

    def get_pos_as_vec(self):
        return np.array([self.x,self.y, self.theta])

    def update(self, dt, steering_angle, acceleration, noise=False, images=False):
        self.steering_angle = steering_angle
        self.velocity += acceleration * dt
        self.velocity = min(self.velocity,self.max_velocity)
        self.x += self.velocity * cos(self.theta) * dt
        self.y += self.velocity * sin(self.theta) * dt
        self.theta += (self.velocity / self.L) * tan(self.steering_angle) * dt
        if noise:
            self.x += np.random.normal(0,0.01)
            self.y += np.random.normal(0,0.01)
            self.theta += np.random.normal(0,0.01)
        self.theta = (self.theta + pi)  % (2 * pi) - pi
        if images:
            self.draw()

    def draw(self):

        image = Image.new("RGB", (self.img_w, self.img_h), (100, 100, 100))
        # Draw a rectangle on the image
        draw = ImageDraw.Draw(image)
        W = self.W * self.p2m
        L = self.L * self.p2m
        # define the angle at which to draw the rectangle (in degrees)
        x_c = 2*self.p2m  + self.x * self.p2m
        y_c = -2*self.p2m + self.img_h - self.y * self.p2m # flip because images have flipped y
        c_th = - self.theta # flip because images have flipped y
        x0 = x_c - L/2 * cos(c_th) - W/2 * sin(c_th)
        y0 = y_c - L/2 * sin(c_th) + W/2 * cos(c_th)
        x1 = x_c - L/2 * cos(c_th) + W/2 * sin(c_th)
        y1 = y_c - L/2 * sin(c_th) - W/2 * cos(c_th)
        x2 = x_c + L/2 * cos(c_th) + W/2 * sin(c_th)
        y2 = y_c + L/2 * sin(c_th) - W/2 * cos(c_th)
        x3 = x_c + L/2 * cos(c_th) - W/2 * sin(c_th)
        y3 = y_c + L/2 * sin(c_th) + W/2 * cos(c_th)

        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))
        draw.polygon([(6*self.p2m + 2*self.p2m, self.img_h - 6*self.p2m - 2*self.p2m), (6*self.p2m + 2*self.p2m, self.img_h - 19*self.p2m - 2*self.p2m), (19*self.p2m + 2*self.p2m, self.img_h -19*self.p2m - 2*self.p2m), (19*self.p2m + 2*self.p2m,self.img_h - 6*self.p2m - 2*self.p2m)], fill=(255, 50, 50))
        self.images.append(image)

    def save_gif(self, name=""):
        self.images[0].save('car'+ name +'.gif',
        save_all=True, append_images=self.images[1:], loop=0)
        


class LatController:
    def __init__(self):
        self.kh = 0.8
        self.kp = 0.8
        self.lookahead_distance = 1.0  # Lookahead distance

    def control(self, trajectory, pose):
        lookahead = np.zeros(2)
        lookahead[0] = pose[0] + self.lookahead_distance * cos(pose[2])
        lookahead[1] = pose[1] + self.lookahead_distance * sin(pose[2])
        # Find the point on the trajectory that is closest to the lookahead distance
        distances = np.linalg.norm(trajectory[:,:2] - lookahead, axis=1)
        idx = np.argmin(np.abs(distances))
        path_error = trajectory[idx,:2] - lookahead
        val = (float(lookahead[1] - pose[1]) * (trajectory[idx,0] - lookahead[0])) - \
           (float(lookahead[0] - pose[0]) * (trajectory[idx,1] - lookahead[1]))
        sign = 0
        if val > 0: sign = -1
        elif val < 0: sign = 1
        # Compute the heading error
        heading_error = trajectory[idx,2] - pose[2]
        # Compute the steering angle
        steering_angle = self.kh * heading_error + self.kp * np.linalg.norm(path_error) *sign
        # print(steering_angle)
        return steering_angle

class LongController:
    def __init__(self):
        self.cruise_speed = 5.0
        self.k = 0.1
    
    def control(self, vehicle_speed, dt):
        speed_error = self.cruise_speed - vehicle_speed
        acc = speed_error / dt
        return self.k * acc

if __name__ == '__main__':
    dt = 0.05
    lat = LatController()
    long = LongController()
    car = SingleTrackVehicleModel()
    planner = RRTStar(0.3)
    path_orig = planner.planning(5000)
    path_orig = np.flip(path_orig, axis=0)
    path = planner.interpolate_polyline(path_orig, 1000)
    car_points = []
    n = 0
    while sqrt((car.x - path[-1,0])**2 + (car.y - path[-1,1])**2) > 1 and n < 600:

        car_points.append(car.get_pos_as_vec())
        acc = lat.control(path,car.get_pos_as_vec())
        steer = long.control(car.velocity, dt)
        if abs(steer) > car.max_steer:
            steer = car.max_steer * steer/abs(steer)
        car.update(dt, acc, steer, False, True)
        n+=1

    plt.plot(path_orig[:,0], path_orig[:,1], label="Planned Path")
    plt.plot(path[:,0], path[:,1], label="Interpolated Path")
    car_points = np.array(car_points)
    print(car_points)
    car.save_gif("hey")
    plt.plot(car_points[:,0], car_points[:,1], label="Actual Trajectory")
    obstacle = np.array([[5.5,5.5], [5.5,19.5],[19.5,19.5],[19.5,5.5], [5.5,5.5]])
    plt.plot(obstacle[:,0], obstacle[:,1], label="Obstacle")
    plt.legend()
    plt.show()