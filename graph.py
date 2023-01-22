import numpy as np
from math import *

class Node:
    def __init__(self, pose):
        self.pose = pose
        self.connections = []
        self.parent = None
        self.cost = 0

class QuadTree:
    def __init__(self, xmin, ymin, xmax, ymax, depth=0):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.depth = depth
        self.points = []
        self.children = []
        self.MAX_DEPTH = 10

    def add_node(self, point):
        if len(self.children) == 0:
            self.points.append(point)
            if len(self.points) > 1 and self.depth < self.MAX_DEPTH:
                self.subdivide()
        else:
            self.add_node_to_children(point)

    def add_node_to_children(self, point):
        x = point.pose[0]
        y = point.pose[1]
        for child in self.children:
            if child.xmin <= x < child.xmax and child.ymin <= y < child.ymax:
                child.add_node(point)
                break

    def subdivide(self):
        xmin, ymin, xmax, ymax = self.xmin, self.ymin, self.xmax, self.ymax
        xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2
        self.children.append(QuadTree(xmin, ymin, xmid, ymid, self.depth + 1))
        self.children.append(QuadTree(xmid, ymin, xmax, ymid, self.depth + 1))
        self.children.append(QuadTree(xmin, ymid, xmid, ymax, self.depth + 1))
        self.children.append(QuadTree(xmid, ymid, xmax, ymax, self.depth + 1))
        for point in self.points:
            self.add_node_to_children(point)
        self.points = []

    def get_nearest(self, point, radius=0.3):
        r = radius
        neighbors = []
        self.find_neighbors_r(point, r, neighbors)
        while len(neighbors) == 0:
            self.find_neighbors_r(point, r, neighbors)
            r *= 1.5
        
        return neighbors

    def get_single_near(self, point):
        min_dist = np.inf
        nearest = None
        x = point[0]
        y = point[1]
        for p in self.points:
            dist = sqrt((x-p.pose[0])**2 + (y-p.pose[1])**2)
            if dist < min_dist:
                nearest = p
                min_dist = dist

        for child in self.children:
            search = False
            if nearest is None:
                search  = True
            else: 
                if child.xmin <= nearest.pose[0] + min_dist and nearest.pose[0] - min_dist <= child.xmax and child.ymin <= nearest.pose[1] + min_dist and nearest.pose[1] - min_dist <= child.ymax:
                    search = True
            if search:
                n, dist = child.get_single_near(point)
                if dist < min_dist:
                    nearest = n
                    min_dist = dist
        return nearest, min_dist


    def find_neighbors_r(self, point, radius, neighbors):
        x = point[0]
        y = point[1]
        if len(self.children) == 0:
            for p in self.points:
                if (x-p.pose[0])**2 + (y-p.pose[1])**2 <= radius**2:
                    neighbors.append(p)
        else:
            for child in self.children:
                if child.xmin <= x + radius and x - radius <= child.xmax and child.ymin <= y + radius and y - radius <= child.ymax:
                    child.find_neighbors_r(point, radius, neighbors)


class Graph:
    
    def __init__(self):
        self.all_nodes = QuadTree(-1,-1,30,30)
        print("graph init")

    def add_vertex(self, pose):
        n = Node(pose)
        self.all_nodes.add_node(n)
        return n

    def add_edge(self,p1, p2, cost):
        p2.parent = p1
        p2.cost = p1.cost + cost
        
    def get_near(self, point, radius = 0.1):
        neighbours = self.all_nodes.get_nearest(point, radius)
        return neighbours

    def get_single_near(self, point):
        return self.all_nodes.get_single_near(point)


    def dist(self, p1, p2):
        d = p2 - p1
        l = np.linalg.norm(d)
        return l

    def get_path(self, p):
        points = []
        if p.parent is not None:
            points.append(p.parent.pose)
            points.extend(self.get_path(p.parent))
        return points