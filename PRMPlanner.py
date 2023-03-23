"""
King's College London
6CCE3ROS & 7CCEMROB Robotic Systems
Coursework 2 - Motion Planning
Path planning Code with PRM (Modified from Week 28 Tutorial - RRT and PythonRobotics, by Yu Su, Xintian Liu,
and Haotian Li)
PythonRobotics: https://github.com/AtsushiSakai/PythonRobotics
"""

import math

import numpy as np
# import matplotlib as mlp
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from celluloid import Camera

import time
from scipy.spatial import distance as dist

# parameter
N_SAMPLE = 500  # the size of the random point set V
N_KNN = 10  # Number of field points for a sample point
MAX_EDGE_LEN = 30.0  # [m] Maximum edge length

show_animation = True

"""
kd-tree for fast nearest-neighbor search

query(self, x[, k, eps, p, distance_upper_bound]): query for neighbours in the vicinity of the kd-tree


"""


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost  # Weight per edge
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + \
               str(self.cost) + "," + str(self.parent_index)


def planning(start_x, start_y, goal_x, goal_y, obstacle_x_list, obstacle_y_list, robot_radius, *, came=None, rng=None):
    """
    Run probabilistic road map planning

    :param start_x: start x position
    :param start_y: start y position
    :param goal_x: goal x position
    :param goal_y: goal y position
    :param obstacle_x_list: obstacle x positions
    :param obstacle_y_list: obstacle y positions
    :param robot_radius: robot radius
    :param rng: Random number constructors
    :return:
    """
    obstacle_kd_tree = KDTree(np.vstack((obstacle_x_list, obstacle_y_list)).T)
    # Sample point set generation
    sample_x, sample_y = sample_points(start_x, start_y, goal_x, goal_y,
                                            robot_radius,
                                            obstacle_x_list, obstacle_y_list,
                                            obstacle_kd_tree, rng)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")

    # Generate probabilistic road maps
    road_map = generate_road_map(sample_x, sample_y, robot_radius, obstacle_kd_tree)
    # Planning paths with Dijkstra
    rx, ry = dijkstra_planning(
        start_x, start_y, goal_x, goal_y, road_map, sample_x, sample_y, came)

    return rx, ry


def is_collision(sx, sy, gx, gy, rr, obstacle_kd_tree):
    """Determine if a collision has occurred, true collision, false no collision
        rr: Robot radius
    """
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    n_step = round(d / D)

    for i in range(n_step):
        dist, _ = obstacle_kd_tree.query([x, y])  # Find neighbours near kd-tree
        if dist <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    dist, _ = obstacle_kd_tree.query([gx, gy])
    if dist <= rr:
        return True  # collision

    return False  # OK


def generate_road_map(sample_x, sample_y, rr, obstacle_kd_tree):
    """
    Probabilistic roadmap generation

    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    robot_radius: Robot Radius[m]
    obstacle_kd_tree: KDTree object of obstacles
    """

    road_map = []
    n_sample = len(sample_x)
    sample_kd_tree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):
        # For each point q in V, choose k neighbourhood points
        dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        edge_id = []

        for ii in range(1, len(indexes)):
            nx = sample_x[indexes[ii]]
            ny = sample_y[indexes[ii]]
            # Each domain point is judged and if it has not yet formed a path together with its neighbour,
            # they are joined to form a path and collision detection is performed,
            # and if there is no collision, the path is retained.
            if not is_collision(ix, iy, nx, ny, rr, obstacle_kd_tree):
                edge_id.append(indexes[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    #  plot_road_map(road_map, sample_x, sample_y)

    return road_map


def dijkstra_planning(sx, sy, gx, gy, road_map, sample_x, sample_y, came):
    """
    s_x: start x position [m]
    s_y: start y position [m]
    goal_x: goal x position [m]
    goal_y: goal y position [m]
    obstacle_x_list: x position list of Obstacles [m]
    obstacle_y_list: y position list of Obstacles [m]
    robot_radius: robot radius [m]
    road_map: Well-constructed road map [m]
    sample_x: Set of sampling points x [m]
    sample_y: The set of sampling points y [m]

    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """

    start_node = Node(sx, sy, 0.0, -1)
    goal_node = Node(gx, gy, 0.0, -1)
    # Constructing open and closed collections using the dictionary approach:
    # The openList table consists of the nodes to be examined
    # and the closeList table consists of the nodes that have already been examined.
    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = start_node

    path_found = True
    # The steps are the same as for the A-star algorithm
    while True:
        # If open_set is empty
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        # show graph
        if show_animation and len(closed_set.keys()) % 2 == 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001) # Mac only
            if came != None:
                came.snap()

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            goal_node.parent_index = current.parent_index
            goal_node.cost = current.cost
            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id],
                             current.cost + d, c_id)

            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if path_found is False:
        return [], []

    # generate final course
    rx, ry = [goal_node.x], [goal_node.y]
    parent_index = goal_node.parent_index
    while parent_index != -1:
        n = closed_set[parent_index]
        rx.append(n.x)
        ry.append(n.y)
        parent_index = n.parent_index

    return rx, ry


def sample_points(sx, sy, gx, gy, rr, ox, oy, obstacle_kd_tree, rng):
    """Sample point set generation
    """
    # the flexible region larger than obstacles since the map is not maze.
    max_x = max(ox) + 2
    max_y = max(oy) + 2
    min_x = min(ox) - 3
    min_y = min(oy) - 4

    sample_x, sample_y = [], []

    if rng is None:
        rng = np.random.default_rng()

    while len(sample_x) <= N_SAMPLE:
        tx = (rng.random() * (max_x - min_x)) + min_x
        ty = (rng.random() * (max_y - min_y)) + min_y

        # Look up the distance of the nearest point in the obstacle to [tx, ty]
        dist, index = obstacle_kd_tree.query([tx, ty])

        # A distance greater than the radius of the robot means that there is no collision,
        # and this collision-free point is added to V and repeated n times.
        if dist >= rr:
            sample_x.append(tx)
            sample_y.append(ty)
    #  the starting and target points
    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)

    return sample_x, sample_y


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")

# decimal range (updated)
def decimal_range(start, stop, increment):
    while start < stop:
        yield start
        start += increment

def main(rng=None):
    print( " start!!")
    fig = plt.figure(1)

    # came = Camera(fig)  # Use when saving motion pictures
    came = None
    # start and goal position
    sx = 0.0  # [m]
    sy = 0.0  # [m]
    gx = 6.0  # [m]
    gy = 10.0  # [m]
    robot_size = 0.8 # [m]

    ox = []
    oy = []

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

    if show_animation:
        plt.plot(ox, oy, ".w")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")
        if came != None:
            came.snap()

        # formatting
        r = [1, 2, 2, 2, 2, 2, 1]
        a = [5, 3, 3, 3, 7, 9, 8]
        b = [5, 6, 8, 10, 5, 5, 10]

        for i in range(7):
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = a[i] + r[i] * np.cos(theta)
            y = b[i] + r[i] * np.sin(theta)

            plt.plot(x, y, linewidth=2.0, color="b")
            plt.axis('equal')

    rx, ry = planning(sx, sy, gx, gy, ox, oy, robot_size, came=came, rng=rng)
    length = 0
    # zip rx, ry to path list (updated, new pth calculation method)
    path = list(zip(rx, ry))
    # print(reversed(path))
    for j in range(0, len(path) - 1):
        length += dist.euclidean(path[j], path[j + 1])
    print("The total path length: %s." % (str(length)))
    # cal avg length part 1 (updated)
    with open(r'./output/performance/prm_length10.txt', 'r') as f:
        past_dl = f.readline()
        # print(past_dl)
        if len(past_dl) != 0:
            past_dl_num = float(past_dl)
            new_dl = past_dl_num + length
            with open(r'./output/performance/prm_length10.txt', 'w') as fw:
                fw.write(str(new_dl))
        else:
            new_dl = length
            with open(r'./output/performance/prm_length10.txt', 'w') as fwe:
                fwe.write(str(new_dl))

    assert rx, 'Cannot found path'

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.pause(0.001) # Mac only
        if came!=None:
            came.snap()
            animation = came.animate()
            animation.save('trajectory.gif')
        # plt.savefig("prm_result.png")
        plt.show()


if __name__ == '__main__':
    start = time.time()
    # run 10 times (updated)
    for k in range(10):
        main()
    # timer & instruction (updated)
    print("The total time cost: %ds." % (time.time() - start))
    # cal avg time (updated)
    print("The average time cost: %ds." % int((time.time() - start) / 10))
    # cal avg length part 2 (updated)
    with open(r'./output/performance/prm_length10.txt') as fc:
        total_len = fc.readline()
    print("The average path length: %s." % (str(float(total_len) / 10)))
    # clear count (updated)
    with open('count_data', 'w') as count_output:
        count_output.write(str(1))
    # with open('count2_data', 'w') as count_output:
    #     count_output.write(str(1))
