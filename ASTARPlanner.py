"""
King's College London
6CCE3ROS & 7CCEMROB Robotic Systems
Coursework 2 - Motion Planning
Path planning Code with ASTAR (Modified from Week 28 Tutorial - RRT and PythonRobotics, by Nanjun Pan, Yajing Zhang,
and Haotian Li)
PythonRobotics: https://github.com/AtsushiSakai/PythonRobotics
"""

import math

import numpy as np
import matplotlib.pyplot as plt

import time
from scipy.spatial import distance as dist

show_animation = True


class AStar:
    """
    Class for ASTAR planning
    """

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m], pixels of the map
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        """Define the search area node class, each Node contains coordinates x and y, movement cost cost and parent node index.
        """
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search
        Enter the coordinates (sx,sy) and (gx,gy) of the starting point and the target point,
        The final output is the coordinate set rx and ry of the points contained in the path.
        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            # Show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xg")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001) # Mac only

            # Dynamically demonstrate pathfinding by tracking current position current.x and current.y
            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        """Computational heuristic

        Args:
            n1 (_type_): _description_
            n2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        # the flexible region larger than obstacles since the map is not maze.
        if px < self.min_x-2:
            return False
        elif py < self.min_y-2:
            return False
        elif px >= self.max_x+2:
            return False
        elif py >= self.max_y+2:
            return False

        # collision check
        if self.obstacle_map[int(node.x)][int(node.y)]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        # the flexible region larger than obstacles since the map is not maze.
        self.min_x = round(min(ox)-2)
        self.min_y = round(min(oy)-2)
        self.max_x = round(max(ox)+2)
        self.max_y = round(max(oy)+2)
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(int(self.y_width))]
                             for _ in range(int(self.x_width))]
        for ix in range(int(self.x_width)):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(int(self.y_width)):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

# decimal range (updated)
def decimal_range(start, stop, increment):
    while start < stop: 
        yield start
        start += increment

def main():
    print(__file__ + " start!!")
    # start and goal position
    sx = 0.0  # [m]
    sy = 0.0  # [m]
    gx = 6.0  # [m]
    gy = 10.0  # [m]
    grid_size = 0.2  # [m]
    robot_radius = 0.8  # [m]

    # set obstacle positions
    ox, oy = [], []
    for i in decimal_range(-1.0, 1.0, 0.05):
        for j in decimal_range(0.0, math.sqrt(1.0-i*i), 0.05):
            ox.append(5.0+i)
            oy.append(5.0+j)
            ox.append(5.0+i)
            oy.append(5.0-j)
    for i in decimal_range(-2.0, 2.0, 0.05):
        for j in decimal_range(0.0, math.sqrt(4.0-i*i), 0.05):
            ox.append(3.0+i)
            oy.append(6.0+j)
            ox.append(3.0+i)
            oy.append(6.0-j)
    for i in decimal_range(-2.0, 2.0, 0.05):
        for j in decimal_range(0.0, math.sqrt(4.0-i*i), 0.05):
            ox.append(3.0+i)
            oy.append(8.0+j)
            ox.append(3.0+i)
            oy.append(8.0-j)
    for i in decimal_range(-2.0, 2.0, 0.05):
        for j in decimal_range(0.0, math.sqrt(4.0-i*i), 0.05):
            ox.append(3.0+i)
            oy.append(10.0+j)
            ox.append(3.0+i)
            oy.append(10.0-j)
    for i in decimal_range(-2.0, 2.0, 0.05):
        for j in decimal_range(0.0, math.sqrt(4.0-i*i), 0.05):
            ox.append(7.0+i)
            oy.append(5.0+j)
            ox.append(7.0+i)
            oy.append(5.0-j)
    for i in decimal_range(-2.0, 2.0, 0.05):
        for j in decimal_range(0.0, math.sqrt(4.0-i*i), 0.05):
            ox.append(9.0+i)
            oy.append(5.0+j)
            ox.append(9.0+i)
            oy.append(5.0-j)
    for i in decimal_range(-1.0, 1.0, 0.05):
        for j in decimal_range(0.0, math.sqrt(1.0-i*i), 0.05):
            ox.append(8.0+i)
            oy.append(10.0+j)
            ox.append(8.0+i)
            oy.append(10.0-j)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".w")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")

    r = [1, 2, 2, 2, 2, 2, 1]
    a = [5, 3, 3, 3, 7, 9, 8]
    b = [5, 6, 8, 10, 5, 5, 10]

    # formatting
    for i in range(7):
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = a[i] + r[i] * np.cos(theta)
        y = b[i] + r[i] * np.sin(theta)
        plt.plot(x, y, linewidth=2.0, color="b")
        plt.axis('equal')

    astar = AStar(ox, oy, grid_size, robot_radius)
    rx, ry = astar.planning(sx, sy, gx, gy)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001) # Mac only
        length = 0
        # zip rx, ry to path list (updated, new pth calculation method)
        path = list(zip(rx, ry))
        # print(reversed(path))
        for j in range(0, len(path) - 1):
            length += dist.euclidean(path[j], path[j + 1])
        print("The total path length: %s." % (str(length)))
        # cal avg length part 1 (updated)
        with open(r'./output/performance/astar_length10.txt', 'r') as f:
            past_dl = f.readline()
            # print(past_dl)
            if len(past_dl) != 0:
                past_dl_num = float(past_dl)
                new_dl = past_dl_num + length
                with open(r'./output/performance/astar_length10.txt', 'w') as fw:
                    fw.write(str(new_dl))
            else:
                new_dl = length
                with open(r'./output/performance/astar_length10.txt', 'w') as fwe:
                    fwe.write(str(new_dl))
        # plt.savefig("astar_result.png")
        plt.show()


if __name__ == '__main__':
    main()
