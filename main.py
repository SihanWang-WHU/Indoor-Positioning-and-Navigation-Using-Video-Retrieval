import sys
import random
import numpy as np
import os
import time
from PIL import Image
import resnet as visual

# -------------------------------------------------------------------------------------------------------
# 存储楼间通道的路径权值和路网的list
# 从七楼到八楼走电梯的路径权值以及路网
dis7_8_0 = 100
traj7_8_0 = [6, 12]
# 从八楼到七楼走电梯的路径权值以及路网
dis8_7_0 = 100
traj8_7_0 = [12, 6]
# 从七楼到八楼走扶梯的路径权值以及路网
dis7_8_1 = 200
traj7_8_1 = [2, 19]
# 从八楼到七楼走扶梯的路径权值以及路网
dis8_7_1 = 200
traj8_7_1 = [19, 2]
# 从七楼到八楼走楼梯的路径权值以及路网
dis7_8_2 = 500
traj7_8_2 = [7, 12]
# 从八楼到七楼走楼梯的路径权值以及路网
dis8_7_2 = 500
traj8_7_2 = [12, 7]

# -------------------------------------------------------------------------------------------------------
# POI类的定义
class POI():
    tag = -1             # 点在路网的矩阵和归纳点的list中的索引， 默认为-1
    name = ''            # 点名，用char存储
    floor = -1           # 点所在的楼层，默认为-1
    classid = -1         # 点在神经网络中类的类id， 默认为-1
    x_coor = -1          # 点的x坐标，默认为-1
    y_coor = -1          # 点的y坐标，默认为-1
    is_elevator = -1     # 是否是电梯 是则为1，不是则是0，默认为-1
    is_escalator = -1    # 是否是扶梯 是则为1，不是则是0，默认为-1
    is_stair = -1        # 是否是楼梯 是则为1，不是则为0，默认为-1

    # -------------------------------------------------------------------------------------------------------
    # Function Name： init
    # Function Usage：POI类的构造函数
    # paras [in]：    self对象，需要初始化的tag, name, floor, x_coor, y_coor,
    #                 is_elevator, is_escalator, is_stair。 具体定义见类中的表达
    # paras [out]：   无
    # -------------------------------------------------------------------------------------------------------
    def __init__(self, tag, name, floor, classid, x_coor,
                 y_coor, is_elevator, is_escalator, is_stair):
        self.tag = tag
        self.name = name
        self.floor = floor
        self.classid = classid
        self.x_coor = x_coor
        self.y_coor = y_coor
        self.is_elevator = is_elevator
        self.is_escalator = is_escalator
        self.is_stair = is_stair
        if self.floor == 7:
            self.list = [np.inf for i in range(0, 10)]  # 构建每个节点的邻接list
        else:
            self.list = [np.inf for i in range(0, 12)]  # 构建每个节点的邻接list

# -------------------------------------------------------------------------------------------------------
# Function Name：  init_pts
# Function Usage： 封装的调用POI的构造函数来初始化点对象，用来构建点的数据库
# paras [in]：     无
# paras [out]：    pts：       所有点的list
#                  pts_seven: 七楼点的list
#                  pts_eight: 八楼点的list
#                  map_seven: 七楼的邻接矩阵
#                  map_eight: 八楼的邻接矩阵
# -------------------------------------------------------------------------------------------------------
def init_pts():
    # 潮牛一号
    p7_0 = POI(tag=0, name='CNYH', floor=7, classid=0, x_coor=1319.98159999959,
               y_coor=-1422.32410000078, is_elevator=1, is_stair=1, is_escalator=1)
    p7_0.list[0] = 0.00000
    p7_0.list[1] = 175.634
    p7_0.list[8] = 220.643

    # 围炉
    p7_1 = POI(tag=1, name='WL', floor=7, classid=12, x_coor=1435.82780000008,
               y_coor=-1554.33479999937, is_elevator=1, is_escalator=1, is_stair=1)
    p7_1.list[0] = 175.634
    p7_1.list[1] = 0.00000
    p7_1.list[7] = 598.938

    # 7楼扶梯
    p7_2 = POI(tag=2, name='FUTISEV', floor=7, classid=5, x_coor=2028.52900000009,
               y_coor=-1468.12370000034, is_elevator=0, is_escalator=1, is_stair=0)
    p7_2.list[2] = 0.00000
    p7_2.list[9] = 414.493

    # 椰子鸡
    p7_3 = POI(tag=3, name='YZJ', floor=7, classid=16, x_coor=2324.87959999963,
               y_coor=-1096.3384000007, is_elevator=1, is_escalator=1, is_stair=1)
    p7_3.list[3] = 0.00000
    p7_3.list[4] = 401.664
    p7_3.list[8] = 1002.79
    p7_3.list[9] = 399.559

    # 遇见阿里
    p7_4 = POI(tag=4, name='YJAL', floor=7, classid=14, x_coor=2723.60589999984,
               y_coor=-1047.84459999948, is_elevator=1, is_escalator=1, is_stair=1)
    p7_4.list[3] = 401.664
    p7_4.list[4] = 0.00000
    p7_4.list[5] = 235.144

    # 翠小小
    p7_5 = POI(tag=5, name='CXX', floor=7, classid=1, x_coor=2947.21600000001,
               y_coor=-975.104000000283, is_elevator=1, is_escalator=1, is_stair=1)
    p7_5.list[4] = 235.144
    p7_5.list[5] = 0.00000
    p7_5.list[6] = 123.958

    # 7楼电梯
    p7_6 = POI(tag=6, name='DIANTISEV', floor=7, classid=3, x_coor=3071.14439999964,
               y_coor=-972.409900000318, is_elevator=1, is_escalator=0, is_stair=0)
    p7_6.list[5] = 123.958
    p7_6.list[6] = 0.00000
    p7_6.list[7] = 280.031

    # 7楼楼梯
    p7_7 = POI(tag=7, name='LOUTISEV', floor=7, classid=8, x_coor=3211.23739999998,
               y_coor=-1214.87859999947, is_elevator=0, is_escalator=0, is_stair=1)
    p7_7.list[1] = 598.938
    p7_7.list[6] = 280.031
    p7_7.list[7] = 0.00000
    p7_7.list[9] = 812.162

    # 7楼节点1
    p7_x1 = POI(tag=8, name='node7.1', floor=7, classid=-1, x_coor=1327.6514,
                y_coor=-1201.814, is_elevator=1, is_escalator=1, is_stair=1)
    p7_x1.list[0] = 220.643
    p7_x1.list[3] = 1002.79
    p7_x1.list[8] = 0.00000

    # 7楼节点2
    p7_x2 = POI(tag=9, name='node7.2', floor=7, classid=-1, x_coor=2442.9029,
                y_coor=-1478.0689, is_elevator=1, is_escalator=1, is_stair=1)
    p7_x2.list[2] = 414.493
    p7_x2.list[3] = 399.559
    p7_x2.list[7] = 812.162
    p7_x2.list[9] = 0.00000

    # 炉鱼
    p8_0 = POI(tag=10, name='LY', floor=8, classid=9, x_coor=1458.54260000027,
               y_coor=-1227.17860000021, is_elevator=1, is_escalator=1, is_stair=1)
    p8_0.list[0] = 0.00000
    p8_0.list[1] = 371.824
    p8_0.list[8] = 286.044

    # 娘惹裙厨
    p8_1 = POI(tag=11, name='NRQC', floor=8, classid=10, x_coor=1830.32799999974,
               y_coor=-1232.56680000014, is_elevator=1, is_escalator=1, is_stair=1)
    p8_1.list[0] = 371.824
    p8_1.list[1] = 0.00000
    p8_1.list[9] = 127.792
    p8_1.list[11] = 684.428

    # 8楼电梯+楼梯
    p8_2 = POI(tag=12, name='DIANLOUTIEIG', floor=8, classid=2, x_coor=2899.88430000003,
               y_coor=-1070.92100000008, is_elevator=1, is_escalator=0, is_stair=1)
    p8_2.list[2] = 0.000000
    p8_2.list[3] = 389.5276
    p8_2.list[11] = 399.421

    # 黄记煌
    p8_3 = POI(tag=13, name='HJH', floor=8, classid=6, x_coor=2934.90759999956,
               y_coor=-1458.87089999951, is_elevator=1, is_escalator=1, is_stair=1)
    p8_3.list[2] = 389.527
    p8_3.list[3] = 0.00000
    p8_3.list[4] = 59.0861

    # 鱼酷
    p8_4 = POI(tag=14, name='YK', floor=8, classid=15, x_coor=2910.66069999989,
               y_coor=-1512.75280000083, is_elevator=1, is_escalator=1, is_stair=1)
    p8_4.list[3] = 59.0861
    p8_4.list[4] = 0.00000
    p8_4.list[5] = 97.1371

    # 苏梅花园
    p8_5 = POI(tag=15, name='SMHY', floor=8, classid=11, x_coor=2819.0614,
               y_coor=-1545.0820000004, is_elevator=1, is_escalator=1, is_stair=1)
    p8_5.list[4] = 97.1371
    p8_5.list[5] = 0.00000
    p8_5.list[6] = 153.941

    # 小菜园
    p8_6 = POI(tag=16, name='XCY', floor=8, classid=13, x_coor=2665.49789999984,
               y_coor=-1555.85840000025, is_elevator=1, is_escalator=1, is_stair=1)
    p8_6.list[5] = 153.9412
    p8_6.list[6] = 0.000000
    p8_6.list[10] = 803.775
    p8_6.list[11] = 422.147

    # 绿茶
    p8_7 = POI(tag=17, name='LC', floor=8, classid=9, x_coor=1210.68580000009,
               y_coor=-1515.44690000079, is_elevator=1, is_escalator=1, is_stair=1)
    p8_7.list[7] = 0.000000
    p8_7.list[8] = 145.4812
    p8_7.list[10] = 651.598

    # 猪肚鸡
    p8_8 = POI(tag=18, name='ZDJ', floor=8, classid=17, x_coor=1210.68580000009,
               y_coor=-1369.96570000052, is_elevator=1, is_escalator=1, is_stair=1)
    p8_8.list[0] = 286.044
    p8_8.list[7] = 145.481
    p8_8.list[8] = 0.00000
    p8_8.list[9] = 660.252

    # 8楼扶梯
    p8_9 = POI(tag=19, name='FUTIEIG', floor=8, classid=4, x_coor=1870.73940000031,
                y_coor=-1353.80110000073, is_elevator=0, is_escalator=1, is_stair=0)
    p8_9.list[1] = 127.792
    p8_9.list[8] = 660.252
    p8_9.list[9] = 0.00000
    p8_9.list[10] = 179.95

    # 8楼节点1
    p8_x1 = POI(tag=20, name='node8.1', floor=8, classid=-1, x_coor=1862.0326,
                y_coor=-1533.5399, is_elevator=1, is_escalator=1, is_stair=1)
    p8_x1.list[6] = 803.775
    p8_x1.list[7] = 651.598
    p8_x1.list[9] = 179.950
    p8_x1.list[10] = 0.0000

    # 8楼节点2
    p8_x2 = POI(tag=21, name='node8.2', floor=8, classid=-1, x_coor=2511.2027,
                y_coor=-1162.9194, is_elevator=1, is_escalator=1, is_stair=1)
    p8_x2.list[1] = 684.428
    p8_x2.list[2] = 399.421
    p8_x2.list[6] = 422.147
    p8_x2.list[11] = 0.0000

    # pts是所有点的list
    pts = [p7_0, p7_1, p7_2, p7_3, p7_4, p7_5, p7_6, p7_7, p7_x1, p7_x2,
           p8_0, p8_1, p8_2, p8_3, p8_4, p8_5, p8_6, p8_7, p8_8, p8_9, p8_x1, p8_x2]
    # pts_seven是七楼点的list
    pts_seven = [p7_0, p7_1, p7_2, p7_3, p7_4, p7_5, p7_6, p7_7, p7_x1, p7_x2]
    # pts_seven是八楼点的list
    pts_eight = [p8_0, p8_1, p8_2, p8_3, p8_4, p8_5, p8_6, p8_7, p8_8, p8_9, p8_x1, p8_x2]
    # map_seven是七楼的map（list）
    map_seven = [p7_0.list, p7_1.list, p7_2.list, p7_3.list, p7_4.list,
                 p7_5.list, p7_6.list, p7_7.list, p7_x1.list, p7_x2.list]
    # map_eight是八楼的map（list）
    map_eight = [p8_0.list, p8_1.list, p8_2.list, p8_3.list, p8_4.list, p8_5.list,
                 p8_6.list, p8_7.list, p8_8.list, p8_9.list, p8_x1.list, p8_x2.list]

    return pts, pts_seven, pts_eight, map_seven, map_eight

# -------------------------------------------------------------------------------------------------------
# Function Name：  init_dijkstra
# Function Usage： dijkstra算法的前期准备工作
# paras [in]：     start_num    开始点的索引
#                  start_floor  开始点的楼层
#                  mapping_list 对应的邻接矩阵
# paras [out]：    node_cost    下一步点的权值以及相应的父节点
#                  close_list   已经计算过的节点
# -------------------------------------------------------------------------------------------------------
def init_dijkstra(start_num, start_floor, mapping_list):
    # 计算起始点与它相邻的节点，更新其node_cost和父节点,并且将A点放入close_list里面
    node0 = start_num
    close_list = []
    # arr这个数组用来表示每个节点的[节点名 node_cost 父节点]
    if start_floor == 7:
        node_cost = [[np.inf for i in range(0, 3)] for i in range(0, 10)]  # 构建全是无穷大的二维列表
        for i in range(0, 10):
            node_cost[i][0] = i
        for i in range(0, 10):
            if mapping_list[int(node0)][i] < node_cost[i][1]:
                node_cost[i][2] = node0
                node_cost[i][1] = mapping_list[int(node0)][i]
        close_list.append(int(node0))
    else:
        node_cost = [[np.inf for i in range(0, 3)] for i in range(0, 12)]  # 构建全是无穷大的二维列表
        for i in range(0, 12):
            node_cost[i][0] = i
        for i in range(0, 12):
            if mapping_list[int(node0)][i] < node_cost[i][1]:
                node_cost[i][2] = node0
                node_cost[i][1] = mapping_list[int(node0)][i]
        close_list.append(int(node0))
    return node_cost, close_list

# -------------------------------------------------------------------------------------------------------
# Function Name：  choose_min
# Function Usage： 用来选择node_cost最小的节点
# paras [in]：     node_cost    下一步点的权值以及相应的父节点
#                  close_list   已经计算过的节点
# paras [out]：    node0        下一步要计算你的节点的索引
# -------------------------------------------------------------------------------------------------------
def choose_min(node_cost, close_list):
    node_cost = np.array(node_cost)  # 将node_cost从list转换成array
    open_list = list(set(node_cost[:, 0].tolist()) - set(close_list))  # 建立一个open_list放入没有被遍历的点
    final_list = []
    for i in open_list:
        final_list.append(node_cost[int(i)].tolist())
    final_list = np.array(final_list)  # final_list转换成array，才可以利用np.where找最小值
    node0 = final_list[np.where(final_list[:, 1] == final_list[:, 1].min())][0][0]  # 将node_cost最小的点的节点名给node0
    return int(node0)

# -------------------------------------------------------------------------------------------------------
# Function Name：  count_cost
# Function Usage： 构建count_cost函数用来计算相邻节点的node_cost
#                  计算node0邻节点的node_cost，此时的node_cost值就是地图上的代价值加上父节点的代价值，如果已经比原来小则更新node_cost和父节点
#                  并将node0放入close_list里面
# paras [in]：     start_num    开始点的索引
#                  start_floor  开始点的楼层
#                  mapping_list 对应的邻接矩阵
#                  node_cost    下一步点的权值以及相应的父节点
#                  close_list   已经计算过的节点
# paras [out]：    更新后的node_cost和close_list
# -------------------------------------------------------------------------------------------------------
def count_cost(mapping_list, node_cost, close_list, start_num, start_floor):
    node0 = start_num
    if start_floor == 7:
        for i in range(0, 10):
            if mapping_list[node0][i] + node_cost[node0][1] < node_cost[i][1]:
                node_cost[i][2] = node0
                node_cost[i][1] = mapping_list[node0][i] + node_cost[node0][1]
        close_list.append(node0)
    else:
        for i in range(0, 12):
            if mapping_list[node0][i] + node_cost[node0][1] < node_cost[i][1]:
                node_cost[i][2] = node0
                node_cost[i][1] = mapping_list[node0][i] + node_cost[node0][1]
        close_list.append(node0)
    return [node_cost, close_list]

# -------------------------------------------------------------------------------------------------------
# Function Name：  dijkstra
# Function Usage： 单层dijkstra算法迭代的过程
# paras [in]：     start_num    开始点的索引
#                  end_num      结束点的索引
#                  start_floor  开始点的楼层
#                  mapping_list 对应的邻接矩阵
# paras [out]：    dis          计算出来的最短路径的权值
#                  destination  计算出来的路径
# -------------------------------------------------------------------------------------------------------
def dijkstra(start_num, end_num, floor, mapping_list):
    if floor ==7:
        node_cost, close_list = init_dijkstra(start_num, floor, mapping_list)
        # 终点不在被选取的list里面的时候继续循环下去
        while end_num not in close_list:
            node0 = choose_min(node_cost, close_list)  # 找node_cost最小的节点
            [node_cost, close_list] = count_cost(mapping_list, node_cost, close_list, node0, floor)  # 计算邻节点
        destination_list = [end_num]
        destination = [end_num]
        dis = node_cost[end_num][1]
        # print("最短的路径代价为:", node_cost[end_num][1])
        while start_num not in destination_list:
            end_num = node_cost[end_num][2]
            destination_list.append(end_num)
            destination.append(end_num)
        # print("最短路径为：", destination_list)
    else:
        # 八楼的tag是从10开始索引的，但是八楼的地图数组还是从[0][0]开始
        # 在这里把开始点和目的地的tag都减去10， 否则数组会越界
        start_num = start_num - 10
        end_num = end_num - 10
        node_cost, close_list = init_dijkstra(start_num, floor, mapping_list)
        # 终点不在被选取的list里面的时候继续循环下去
        while end_num not in close_list:
            node0 = choose_min(node_cost, close_list)  # 找node_cost最小的节点
            [node_cost, close_list] = count_cost(mapping_list, node_cost, close_list, node0, floor)  # 计算邻节点
        destination_list = [end_num]
        destination = [end_num + 10]
        # print("最短的路径代价为:", node_cost[end_num][1])
        dis = node_cost[end_num][1]
        while start_num not in destination_list:
            end_num = node_cost[end_num][2]
            destination_list.append(end_num)
            destination.append(end_num + 10)
        # print("最短路径为：", destination)
    return dis, destination

# -------------------------------------------------------------------------------------------------------
# Function Name：   classid2tag
# Function Usage：  深度学习的结果是classid， 通过这个函数映射到tag上面去（即pts的索引）
# paras [in]：      classid
# paras [out]：     相应兴趣点的索引（tag）
# -------------------------------------------------------------------------------------------------------
def classid2tag(classid):
    # 这里的tags是以classid为索引的tag
    tags = [0, 5, 12, 6, 19, 2, 13, 17, 7, 10, 11, 15, 1, 16, 4, 14, 3, 18]
    return tags[classid]

# -------------------------------------------------------------------------------------------------------
# Function Name：    navigation
# Function Usage：   路径规划封装的函数
# paras [in]：       img       通过用户手机采集的图像，格式是Image.open的返回值
#                    end_num  目的地的索引
#                    strategy  路径规划的策略 3为路径最短 0为只走电梯 1为只走扶梯 2为只走楼梯
# paras [out]：      dis       计算出来的最小长度
#                    traj      计算出来的最短路径
# -------------------------------------------------------------------------------------------------------
def navigation(img, end_num, strategy):
    start_time = time.process_time()
    # 模型的初始化
    model_name = 'model.pth'
    model, transform_test = visual.model_init(model_name)
    # 进行深度学习的位置计算
    predictclass, classid, duration = visual.resnet_predict(transform_test, model, img)
    start_num = classid2tag(classid)
    # print('time_elapsed:{}'.format(duration))

    # 路径规划的地图初始化
    points, points_seven, points_eight, map_seven, map_eight = init_pts()
    map_seven = np.array(map_seven)
    map_eight = np.array(map_eight)

    start_floor = points[start_num].floor
    end_floor = points[end_num].floor

    print('当前楼层：{}'.format(start_floor))
    print('目的地：{}'.format(points[end_num].name))
    print('目的地楼层：{}'.format(end_floor))

    # 如果当前定位和目的地在同一层楼
    if start_floor == end_floor:
        if start_floor == 7:
            dis, traj = dijkstra(start_num, end_num, 7, map_seven)
        elif start_floor == 8:
            dis, traj = dijkstra(start_num, end_num, 8, map_eight)

    # 如果当前定位在七楼，目的地在八楼
    if start_floor == 7 and end_floor == 8:
        # 所有路都走
        if strategy == 3:
            # 走电梯
            dis7_0, traj7_0 = dijkstra(start_num, 6, 7, map_seven)
            dis8_0, traj8_0 = dijkstra(12, end_num, 8, map_eight)
            dis_0 = dis7_0 + dis8_0 + dis7_8_0
            traj8_0.extend(traj7_0)
            traj_0 = traj8_0
            # 走扶梯
            dis7_1, traj7_1 = dijkstra(start_num, 2, 7, map_seven)
            dis8_1, traj8_1 = dijkstra(19, end_num, 8, map_eight)
            dis_1 = dis7_1 + dis8_1 + dis7_8_1
            traj8_1.extend(traj7_1)
            traj_1 = traj8_1
            # 走楼梯
            dis7_2, traj7_2 = dijkstra(start_num, 7, 7, map_seven)
            dis8_2, traj8_2 = dijkstra(12, end_num, 8, map_eight)
            dis_2 = dis7_2 + dis8_2 + dis7_8_2
            traj8_2.extend(traj7_2)
            traj_2 = traj8_2

            compare = np.array([dis_0, dis_1, dis_2])
            traj_all = [traj_0, traj_1, traj_2]
            index = compare.argmin()
            dis = compare[index]
            traj = traj_all[index]

        # 只走电梯
        elif strategy == 0:
            dis7_0, traj7_0 = dijkstra(start_num, 6, 7, map_seven)
            dis8_0, traj8_0 = dijkstra(12, end_num, 8, map_eight)
            dis = dis7_0 + dis8_0 + dis7_8_0
            traj8_0.extend(traj7_0)
            traj = traj8_0

            # 只走扶梯
        elif strategy == 1:
            dis7_1, traj7_1 = dijkstra(start_num, 2, 7, map_seven)
            dis8_1, traj8_1 = dijkstra(19, end_num, 8, map_eight)
            dis = dis7_1 + dis8_1 + dis7_8_1
            traj8_1.extend(traj7_1)
            traj = traj8_1

            # 只走楼梯
        elif strategy == 2:
            dis7_2, traj7_2 = dijkstra(start_num, 7, 7, map_seven)
            dis8_2, traj8_2 = dijkstra(12, end_num, 8, map_eight)
            dis = dis7_2 + dis8_2 + dis7_8_2
            traj8_2.extend(traj7_2)
            traj = traj8_2

        else:
            print("请选择正确的路径规划策略！")
            sys.exit()

    # 如果当前定位在八楼，目的地在七楼
    elif start_floor == 8 and end_floor == 7:
        # 所有路都走
        if strategy == 3:
            # 走电梯
            dis8_0, traj8_0 = dijkstra(start_num, 12, 8, map_eight)
            dis7_0, traj7_0 = dijkstra(6, end_num, 7, map_seven)
            dis_0 = dis7_0 + dis8_0 + dis8_7_0
            traj7_0.extend(traj8_0)
            traj_0 = traj7_0

            # 走扶梯
            dis8_1, traj8_1 = dijkstra(start_num, 19, 8, map_eight)
            dis7_1, traj7_1 = dijkstra(2, end_num, 7, map_seven)
            dis_1 = dis7_1 + dis8_1 + dis8_7_1
            traj7_1.extend(traj8_1)
            traj_1 = traj7_1

            # 走楼梯
            dis8_2, traj8_2 = dijkstra(start_num, 12, 8, map_eight)
            dis7_2, traj7_2 = dijkstra(7, end_num, 7, map_seven)
            dis_2 = dis7_2 + dis8_2 + dis8_7_2
            traj7_2.extend(traj8_2)
            traj_2 = traj7_2

            compare = np.array([dis_0, dis_1, dis_2])
            traj_all = [traj_0, traj_1, traj_2]
            index = compare.argmin()
            dis = compare[index]
            traj = traj_all[index]

        # 只走电梯
        elif strategy == 0:
            dis8_0, traj8_0 = dijkstra(start_num, 12, 8, map_eight)
            dis7_0, traj7_0 = dijkstra(6, end_num, 7, map_seven)
            dis = dis7_0 + dis8_0 + dis8_7_0
            traj7_0.extend(traj8_0)
            traj = traj7_0

            # 只走扶梯
        elif strategy == 1:
            dis8_1, traj8_1 = dijkstra(start_num, 19, 8, map_eight)
            dis7_1, traj7_1 = dijkstra(2, end_num, 7, map_seven)
            dis = dis7_1 + dis8_1 + dis8_7_1
            traj7_1.extend(traj8_1)
            traj = traj7_1

            # 只走楼梯
        elif strategy == 2:
            dis8_2, traj8_2 = dijkstra(start_num, 12, 8, map_eight)
            dis7_2, traj7_2 = dijkstra(7, end_num, 7, map_seven)
            dis = dis7_2 + dis8_2 + dis8_7_2
            traj7_2.extend(traj8_2)
            traj = traj7_2

        else:
            print("请选择正确的路径规划策略！")
            sys.exit()

    names = []
    coors = []
    traj.reverse()
    traj_lenth = len(traj)
    for i in range(0, traj_lenth):
        names.append(points[traj[i]].name)
        coors.append([points[traj[i]].x_coor, points[traj[i]].y_coor])
    end_time = time.process_time()
    duration = end_time - start_time

    if strategy == 0:
        print("规划策略为只走电梯")
    elif strategy == 1:
        print("规划策略为只走扶梯")
    elif strategy == 2:
        print("规划策略为只走楼梯")
    elif strategy == 3:
        print("规划策略为路径最短")

    print("最短路线长度为：", dis)
    print("最短路径为：", traj)
    print('依次经历的店铺为：{}'.format(names))
    print('依次经历的坐标为:{}'.format(coors))
    print('计算时间：{}s'.format(duration))
    print("------------------------------------")
    print(" ")

    return dis, traj


if __name__ == "__main__":
    path = 'test/'
    testList = os.listdir(path)
    for file in testList:
        img = Image.open(path + file)
        end_num = random.randint(0, 21)
        strategy = random.randint(0, 3)
        dis, traj = navigation(img, end_num, strategy)
