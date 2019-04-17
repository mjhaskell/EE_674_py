import numpy as np
from message_types.msg_waypoints import msg_waypoints
from chap11.dubins_params import dubins_params
from IPython.core.debugger import Pdb

class planRRT():
    def __init__(self):
        self.segmentLength = 300 # standard length of path segments
        self.clearance = 10
        self.dubins_path = dubins_params()

    def planPath(self, wpp_start, wpp_end, R_min, map):
        self.segmentLength = 2.5*R_min

        # desired down position is down position of end node
        pd = wpp_end.item(2)

        # specify start and end nodes from wpp_start and wpp_end
        # format: N, E, D, chi, cost, parentIndex, connectsToGoalFlag,
        start_node = np.array([wpp_start.item(0), wpp_start.item(1), pd, 0, 0, -1, 0])
        end_node = np.array([wpp_end.item(0), wpp_end.item(1), pd, 0, 0, 0, 0])

        # establish tree starting with the start node
        tree = start_node.reshape(1,len(start_node))

        # check to see if start_node connects directly to end_node
        if ((np.linalg.norm(start_node[0:3] - end_node[0:3]) < self.segmentLength ) and not self.collision(start_node, end_node, map)):
            waypoints = msg_waypoints()
            waypoints.ned[:,0] = start_node[:3]
            waypoints.ned[:,1] = end_node[:3]
            waypoints.num_waypoints = 2
            waypoints.cost[0][0] = 0
            waypoints.parent_idx[0][0] = 0
            waypoints.flag_connect_to_goal[0][0] = 1
            return waypoints

        numPaths = 0
        while numPaths < 3:
            tree, flag = self.extendTree(tree, end_node, map, pd, R_min)
            numPaths = numPaths + flag

        # find path with minimum cost to end_node
        path = self.findMinimumPath(tree, end_node)
        return self.smoothPath(path, map, R_min)

    def generateRandomPoint(self, map, pd):
        north_east = np.random.uniform(0,map.city_width,2)
        pt = np.hstack([north_east, pd])
        return pt

    def collision(self, start_node, end_node, map, radius):
        self.dubins_path.update(start_node[:3], start_node.item(3), \
                end_node[:3], end_node.item(3), radius)
        pts = self.pointsAlongPath(start_node, end_node, 5)

        for i in range(len(map.building_north)):
            norths = np.abs(pts[:,0] - map.building_north[i])
            in_north = (norths < map.building_width/2 + self.clearance)
            if np.any(in_north):
                in_north_pts = pts[in_north]
                for j in range(len(map.building_east)):
                    easts = np.abs(in_north_pts[:,1] - map.building_east[j])
                    in_east = (easts < map.building_width/2+self.clearance)
                    if np.any(in_east):
                        in_east_pts = in_north_pts[in_east]
                        pd = end_node[2]
                        if (-pd < map.building_height[j,i]+self.clearance):
                            return True
        return False

    def pointsAlongPath(self, start_node, end_node, Del):
        q = end_node[:3] - start_node[:3]
        L = np.linalg.norm(q)
        num_pts = int(np.ceil(L / Del))
        q /= L
        ned = np.zeros((num_pts,3))

        for i in range(num_pts):
            ned[i,:] = start_node[:3]+q*Del*(i+1)

        return ned

    def extendTree(self, tree, end_node, map, pd, radius):
        success = False
        while not success:
            pt = self.generateRandomPoint(map, pd)
            distances = (pt.item(0)-tree[:,0])**2 + (pt.item(1)-tree[:,1])**2
            idx = np.argmin(distances)

            D = min(np.sqrt(distances[idx]), self.segmentLength)
            q = pt - tree[idx,:3]
            q /= np.linalg.norm(q)
            new_pt = tree[idx,:3] + q*D
            if not self.collision(tree[idx,:3], new_pt, map, radius):
                success = True
                dist_to_goal = np.linalg.norm(new_pt - end_node[:3])
                chi = np.arctan2((pt.item(1)-tree[idx,1],pt.item(0)-tree[idx,0]))
                if dist_to_goal < self.segmentLength and not self.collision(new_pt,end_node[:3],map):
                    flag = 1
                    new_node = np.hstack([new_pt, chi, D+tree[idx,3], idx, 0])
                    tree = np.vstack([tree,new_node])
                    end_D = np.linalg.norm(end_node[:3] - new_pt)
                    chi = np.arctan2(end_node[1]-new_pt[1],end_node[0]-new_pt[0])
                    cur_end_node = np.hstack([end_node[:3],chi,end_D+new_node[3],idx+1,flag])
                    tree = np.vstack([tree,cur_end_node])
                else:
                    flag = 0
                    new_node = np.hstack([new_pt, chi, D+tree[idx,3], idx, flag])
                    tree = np.vstack([tree,new_node])

                return tree, flag

    def findMinimumPath(self, tree, end_node):
        idxs = np.nonzero(tree[:,-1])
        idxs = idxs[0]
        idx = np.argmin(tree[idxs, 3])
        idx = idxs[idx]

        waypoints = tree[idx]
        while tree[idx,4] != -1:
            idx = int(tree[idx,4])
            waypoints = np.vstack([waypoints, tree[idx]])
        return waypoints[::-1]

    def smoothPath(self, path, map, radius):
        smoothed_path = []
        i = 0
        j = 1

        smoothed_path.append(path[i])
        while j < len(path)-1:
            ws = path[i]
            wp = path[j+1]
            if self.collision(ws, wp, map, radius):
                last_node = path[j]
                smoothed_path.append(last_node)
                i = j
            j += 1
                
        smoothed_path.append(path[-1])
        nodes = np.array(smoothed_path)
        waypoints = msg_waypoints()

        k = 0
        for node in nodes:
            waypoints.ned[:,k] = node[:3]
            waypoints.airspeed[0][k] = 25.
            waypoints.cost[0][k] = node[3]
            waypoints.parent_idx[0][k] = node[4]
            waypoints.flag_connect_to_goal[0][k] = node[5]
            k += 1

        waypoints.flag_wrap_waypoints = False
        waypoints.num_waypoints = k
        return waypoints
