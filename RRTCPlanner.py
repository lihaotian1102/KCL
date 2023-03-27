"""
King's College London
6CCE3ROS & 7CCEMROB Robotic Systems
Coursework 2 - Motion Planning
Path planning Code with RRT-Connect (Modified from Week 28 Tutorial - RRT, by Haotian Li)
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

# required libs
import operator
import copy
import time
from scipy.spatial import distance as dist

show_animation = True


class RRTConnect:
    """
    Class for RRT-Connect planning
    """

    class Node:
        """
        RRT-Connect Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500,
                 play_area=None,
                 robot_radius=0.0,
                 ):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius: robot body modeled as circle with given radius
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list_1 = []
        # RRT-Connect needs two lists to store start and end respectively (RRT only needs one)
        self.node_list_2 = []
        self.robot_radius = robot_radius

    # modified function (updated)
    def planning(self, animation=True):
        """
        rrt-connect path planning
        animation: flag for animation on or off
        """

        self.node_list_1 = [self.start]
        self.node_list_2 = [self.end]

        for i in range(self.max_iter):
            rnd_node = self.sample_free()

            nearest_ind_1 = self.get_nearest_node_index(self.node_list_1, rnd_node)
            nearest_node_1 = self.node_list_1[nearest_ind_1]
            new_node_1 = self.steer(nearest_node_1, rnd_node, self.expand_dis)

            # new function that determine the node whether inside, if inside, check obs free.
            if self.is_inside_play_area(new_node_1, self.play_area) and self.obstacle_free(new_node_1,
                                                                                           self.obstacle_list,
                                                                                           self.robot_radius):
                self.node_list_1.append(new_node_1)

                # steer
                nearest_ind_2 = self.get_nearest_node_index(self.node_list_2, new_node_1)
                nearest_node_2 = self.node_list_2[nearest_ind_2]
                new_node_2 = self.steer(nearest_node_2, new_node_1, self.expand_dis)

                # new function that determine the node whether inside, if inside, check obs free.
                if self.is_inside_play_area(new_node_2, self.play_area) and self.obstacle_free(new_node_2,
                                                                                               self.obstacle_list,
                                                                                               self.robot_radius):
                    self.node_list_2.append(new_node_2)
                    # steer
                    while True:
                        new_node_2_ = self.steer(new_node_2, new_node_1, self.expand_dis)

                        if self.obstacle_free(new_node_2_, self.obstacle_list, self.robot_radius):
                            self.node_list_2.append(new_node_2_)
                            new_node_2 = new_node_2_

                        else:
                            break

                        if operator.eq([new_node_2.x, new_node_2.y], [new_node_1.x, new_node_1.y]):
                            return self.merge_final_path()

            # kind of AVL tree thought: extend the smaller node list, keep balancing.
            if len(self.node_list_1) > len(self.node_list_2):
                list_tmp = copy.deepcopy(self.node_list_1)
                self.node_list_1 = copy.deepcopy(self.node_list_2)
                self.node_list_2 = list_tmp

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node, new_node_1)

        return None  # cannot find path

    # modified function (updated, conditional resolution)
    def steer(self, from_node, to_node, extend_length=float("inf")):

        d, theta = self.calc_distance_and_angle(from_node, to_node)

        if extend_length >= d:
            new_x = to_node.x
            new_y = to_node.y
        # if smaller than d, become new node.
        else:
            new_x = from_node.x + math.cos(theta)
            new_y = from_node.y + math.sin(theta)
        new_node = self.Node(new_x, new_y)
        new_node.path_x = [from_node.x]
        new_node.path_y = [from_node.y]
        new_node.path_x.append(new_x)
        new_node.path_y.append(new_y)

        new_node.parent = from_node

        return new_node

    # modified function (updated)
    def merge_final_path(self):

        path_1 = []
        node = self.node_list_1[-1]
        while node.parent is not None:
            path_1.append([node.x, node.y])
            node = node.parent
        path_1.append([node.x, node.y])

        path_2 = []
        node = self.node_list_2[-1]
        while node.parent is not None:
            path_2.append([node.x, node.y])
            node = node.parent
        path_2.append([node.x, node.y])

        # merge two paths (bottom-up & top-down)
        path = []
        for i in range(len(path_1) - 1, -1, -1):
            path.append(path_1[i])
        for i in range(len(path_2)):
            path.append(path_2[i])

        return path

    # modified function (updated)
    def calc_dist(self, x1, y1, x2, y2):

        dx = x1 - x2
        dy = y1 - y2
        return math.hypot(dx, dy)

    def sample_free(self):

        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    # modified function (updated)
    def draw_graph(self, rnd=None, rnd_2=None):

        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        # double rnd
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')
        if rnd_2 is not None:
            plt.plot(rnd_2.x, rnd_2.y, "^r")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd_2.x, rnd_2.y, self.robot_radius, '-b')
        # double node list
        for node in self.node_list_1:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
        for node in self.node_list_2:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
        # same obs
        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        # plt.pause(0.01)  # Mac only
        # pic storage (updated)
        # try:
        #     with open('count_data') as count_input:
        #         i = int(count_input.read())
        # except IOError:
        #     i = 1
        # plt.savefig("./output/pic/cpics1/" + str(i) + ".png", format='png')
        # print("Plot and store the step " + str(i) + " figure...")
        # i += 1
        # with open('count_data', 'w') as count_output:
        #     count_output.write(str(i))

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list_1, rnd_node):
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list_1]
        minind = dlist.index(min(dlist))

        return minind

    # add new function (updated)
    @staticmethod
    def is_inside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
                node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def obstacle_free(node, obstacleList, robot_radius):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size + robot_radius) ** 2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


# list storage func (updated)
def list_txt(path, list=None):
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist


def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # ====Search Path with RRT-Connect====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    rrtc = RRTConnect(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-2, 15],
        obstacle_list=obstacleList,
        # play_area=[0, 10, 0, 14]
        robot_radius=0.8
    )
    path = rrtc.planning(animation=show_animation)

    if path is None:
        print("Cannot find path!!")
    else:
        print("Found path!!")
        # length (updated)
        length = 0
        for j in range(0, len(path) - 1):
            length += dist.euclidean(path[j], path[j + 1])
        print("The total path length: %s." % (str(length)))
        # cal avg length part 1 (updated)
        with open(r'./output/performance/rrtc_length100.txt', 'r') as f:
            past_dl = f.readline()
            # print(past_dl)
            if len(past_dl) != 0:
                past_dl_num = float(past_dl)
                new_dl = past_dl_num + length
                with open(r'./output/performance/rrtc_length100.txt', 'w') as fw:
                    fw.write(str(new_dl))
            else:
                new_dl = length
                with open(r'./output/performance/rrtc_length100.txt', 'w') as fwe:
                    fwe.write(str(new_dl))
        # path storage (updated)
        # try:
        #     with open('count2_data') as count2_input:
        #         ii = int(count2_input.read())
        # except IOError:
        #     ii = 1
        # print("The path is:" + str(list(reversed(path))))
        # list_txt(path='./output/path/cpath' + str(ii) + '.txt', list=list(reversed(path)))

        # Draw final path
        if show_animation:
            rrtc.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            # plt.pause(0.01)  # Mac only
            # path-pic storage (updated)
            # plt.savefig("./output/path/cpath" + str(ii) + "_pic.png", format='png')
            # plt.show()
            # ii += 1
            # with open('count2_data', 'w') as count2_output:
            #     count2_output.write(str(ii))


if __name__ == '__main__':
    start = time.time()
    # run 100 times (updated)
    for k in range(100):
        main()
    # timer & instruction (updated)
    print("The total time cost: %ds." % (time.time() - start))
    # cal avg time (updated)
    print("The average time cost: %ds." % int((time.time() - start) / 100))
    # cal avg length part 2 (updated)
    with open(r'./output/performance/rrtc_length100.txt') as fc:
        total_len = fc.readline()
    print("The average path length: %s." % (str(float(total_len) / 100)))
    # clear count (updated)
    with open('count_data', 'w') as count_output:
        count_output.write(str(1))
    # with open('count2_data', 'w') as count_output:
    #     count_output.write(str(1))
