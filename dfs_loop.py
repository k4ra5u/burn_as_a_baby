import time
from pynput import mouse
from Quartz import CGEventCreateMouseEvent, CGEventPost, kCGHIDEventTap
from Quartz import kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp, kCGMouseButtonLeft
from PIL import ImageGrab, Image
import numpy as np
import cv2
import io
from pwn import *
from alibabacloud_ocr_api20210707.client import Client as ocr_api20210707Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_darabonba_stream.client import Client as StreamClient
from alibabacloud_ocr_api20210707 import models as ocr_api_20210707_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
from collections import defaultdict, deque
import copy
import time

rows = 16
cols = 10
use_api = True 
accepted_target = 140
next_solve_min_times = 4
next_solve_max_times = 8

# 全局变量存储鼠标拖动的起点和终点
stop_flag = 0
start_position = None
end_position = None
is_selecting = False
max_score = 0
start_time = 0
max_sum = 0
final_order = []
final_order_pre = []

def mouse_down(x, y):
    """按下鼠标左键"""
    event = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event)

def mouse_up(x, y):
    """释放鼠标左键"""
    event = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event)

def mouse_move(x, y):
    """移动鼠标到指定位置"""
    event = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event)

def mouse_click(x, y):
    """
    使用 CGEventCreateMouseEvent 模拟鼠标点击
    :param x: 点击的 X 坐标
    :param y: 点击的 Y 坐标
    """
    # 创建鼠标按下事件
    event_down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
    # 创建鼠标松开事件
    event_up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
    
    # 发送事件
    CGEventPost(kCGHIDEventTap, event_down)
    CGEventPost(kCGHIDEventTap, event_up)

def drag_rectangle(start_x, start_y, end_x, end_y, step=10):
    """
    模拟鼠标按住并拖动
    :param start_x: 起点 X 坐标
    :param start_y: 起点 Y 坐标
    :param end_x: 终点 X 坐标
    :param end_y: 终点 Y 坐标
    :param step: 鼠标移动的步长
    """
    mouse_down(start_x, start_y)
    time.sleep(0.1)

    x, y = start_x, start_y
    while x <= end_x and y <= end_y:
        mouse_move(x, y)
        time.sleep(0.01)  # 控制平滑性
        x += step
        y += step

    mouse_up(end_x, end_y)
    print(f"拖动完成：从 ({start_x}, {start_y}) 到 ({end_x}, {end_y})")

def on_click(x, y, button, pressed):
    """鼠标点击事件监听"""
    global start_position, end_position, is_selecting
    if button == mouse.Button.left:
        if pressed:  # 按下左键
            start_position = (x, y)
            is_selecting = True
            print(f"起点：{start_position}")
        else:  # 松开左键
            end_position = (x, y)
            is_selecting = False
            print(f"终点：{end_position}")
            return False  # 停止监听

def on_move(x, y):
    """鼠标移动事件监听"""
    if is_selecting:
        print(f"鼠标移动到 ({x}, {y})")

def preprocess_image(image):
    # 转为灰度图
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    
    # 二值化（高对比度）
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # 去噪（可选，影响识别的精度）
    denoised = cv2.fastNlMeansDenoising(binary, None, 3, 7, 21)
    
    # 转回 PIL 图像
    # pil_image = Image.fromarray(denoised)
    
    # return pil_image
    return denoised

def capture_and_recognize_matrix(start_pos, end_pos):
    """
    截取矩形区域，预处理图像并识别矩阵中的数字
    :param start_pos: 起点坐标 (x, y)
    :param end_pos: 终点坐标 (x, y)
    """
    # 确定矩形的边界并转换为整数
    left = int(min(start_pos[0], end_pos[0]))
    top = int(min(start_pos[1], end_pos[1]))
    right = int(max(start_pos[0], end_pos[0]))
    bottom = int(max(start_pos[1], end_pos[1]))

    # 截图
    print(f"正在截图区域：({left}, {top}, {right}, {bottom})")
    for i in range(10):
        try:
            screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
            # 将截屏转换为二进制数据
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')  # 可以选择格式，比如 PNG 或 JPEG
            binary_data = buffer.getvalue()

            # 图像预处理
            # rsub_images = split_image_into_grid(screenshot, rows=8, cols=8)
            # ocr_results = []

            # OCR 识别（保留布局）
            print("正在识别矩阵中的数字...")


            matrix = ocr_from_alibaba(binary_data)

                

            print("识别结果：")
            print(matrix)
            break
        except Exception as e:
            continue
    return matrix


    # # OCR 识别（保留布局）
    # print("正在识别矩阵中的数字...")
    # ocr_result = pytesseract.image_to_string(
    #     processed_image, config="--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789"
    # )
    # # ocr_result = recognize_with_easyocr(screenshot)

    # # 输出识别结果
    # print("识别结果：")
    # print(ocr_result)

def create_client():
    """
    使用AK&SK初始化账号Client

    @return: Client
    @throws Exception
    """
    # 工程代码泄露可能会导致 AccessKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考。
    # 建议使用更安全的 STS 方式，更多鉴权访问方式请参见：https://help.aliyun.com/document_detail/378659.html。
    config = open_api_models.Config(
        access_key_id=os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'],
        access_key_secret=os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
    )
    # Endpoint 请参考 https://api.aliyun.com/product/ocr-api
    config.endpoint = 'ocr-api.cn-hangzhou.aliyuncs.com'
    return ocr_api20210707Client(config)

def ocr_from_alibaba(image_stream):
        if use_api:
            client = create_client()
            # 需要安装额外的依赖库，直接点击下载完整工程即可看到所有依赖。
            # body_stream = StreamClient.read_from_file_path('/Users/xiongjing/Desktop/wecaht-game/maze.jpg')
            recognize_all_text_request = ocr_api_20210707_models.RecognizeAllTextRequest(
                body=image_stream,
                type='Advanced'
            )
            runtime = util_models.RuntimeOptions()
            response = client.recognize_all_text_with_options(recognize_all_text_request, runtime)
            # 解析 Content
            content_string = response.body.data.content
            print(content_string)
        else:
            content_string = b'3 2 6 2 4 6 5 5 9 3 6 5 7 9 2 7 5 8 6 6 4 1 4 1 2 7 8 4 7 9 2 4 1 7 5 2 3 4 6 6 5 6 4 6 1 2 9 7 4 8 9 4 4 6 3 9 2 3 5 5 4 5 8 6 4 9 7 3 9 3 3 3 7 3 1 5 9 1 3 8 3 6 8 7 2 3 2 8 9 4 1 9 4 2 9 7 7 1 1 5 3 5 5 2 1 1 2 2 4 9 8 3 1 8 9 3 7 2 2 8 3 2 5 9 2 8 6 4 2 6 9 5 6 9 4 8 1 7 9 4 1 8 8 9 4 9 5 6 1 4 5 7 7 2 3 4 8 1 8 1 '

        # 提取数字并转换为列表
        content_list = list(map(int, content_string.split()))

        # 转换为二维数组
        matrix = np.array(content_list).reshape(rows, cols)

        # 输出结果
        return matrix

def drag_rectangle_with_maze_id(r1,c1,r2,c2):
    start_x, start_y = start_position
    end_x, end_y = end_position
    x_layout = (end_x - start_x) // cols
    y_layout = (end_y - start_y) // rows
    x1 = start_x + c1 * x_layout + x_layout // 2
    y1 = start_y + r1 * y_layout + y_layout // 2
    x2 = start_x + c2 * x_layout + x_layout // 2 + 1
    y2 = start_y + r2 * y_layout + y_layout // 2 + 1
    drag_rectangle(x1, y1, x2, y2)
    pass

# check if sm2 is in sm1
def check_full_overlap(sm1,sm2):
    x11,y11,x12,y12,sum2,_ = sm2
    x21,y21,x22,y22,sum1,_ = sm1
    if x11 >= x21 and x11 <=x22 and x12 >=x21 and x12 <=x22:
        if y11 >=y21 and y11 <=y22 and y12 >=y21 and y12 <=y22:
            if sum2 < sum1:
                return True
    return False

def check_overlap(sm1,sm2):
    i1, j1, k1, l1,_,_ = sm1 # 矩阵 A 的左上角 (i1, j1) 和右下角 (k1, l1)
    i2, j2, k2, l2,_,_ = sm2  # 矩阵 B 的左上角 (i2, j2) 和右下角 (k2, l2)
    
    # 判断是否有重叠
    if i1 <= k2 and k1 >= i2 and j1 <= l2 and l1 >= j2:
        return True  # 有重叠
    return False  # 没有重叠

# 在当前matrix下每一轮最多的解法
def dfs(matrix,dfs1_stop,orders,lap,submatrices,max_orders,hit_max_times,vis,graph,indegree,start,step,order,cur_area):
    global accepted_target
    global stop_flag
    global final_order
    global final_order_pre
    global max_score

    sub_matrix_lens = len(submatrices)
    if dfs1_stop == 1 or stop_flag == 1:
        return max_orders,hit_max_times
    if step == 0 :
        for subm in order:
            print(subm)
        dfs1_stop = 1
        orders = list(order)
        return max_orders,hit_max_times
    visited_ids = []
    # for j in range(sub_matrix_lens):
    #     print(submatrices[j],indegree[j],"dfsa")
    for i in range(start,sub_matrix_lens):
        _,_,_,_,val,_ = submatrices[i]
        if indegree[i] == 0 and vis[i] ==0 and val <= 30:
            order.append(submatrices[i])
            # print("appand",submatrices[i])
            # if val != 10:
            #     print(submatrices[i],"wrong")
                # for j in range(sub_matrix_lens):
                #     print(submatrices[j],indegree[j],"dfsb")
            visited_ids.append(i)
            vis[i] = 1
            for neighbor in graph[i]:
                indegree[neighbor] -= 1
            for j in range(sub_matrix_lens):
                if lap[i][j] == 1 and vis[j] == 0:
                    visited_ids.append(j)
                    vis[j] = 1
            # print(i)
            a,b,c,d,_,_ = submatrices[i]
            this_area = (c+1-a)* (d+1-b)

            # print("pre ",len(order),max_orders,step -1,cur_area + this_area)
            max_orders,hit_max_times = dfs(matrix,dfs1_stop,orders,lap,submatrices,max_orders,hit_max_times,vis,graph,indegree,0,step - 1,order,cur_area + this_area)
            # 代表前面已经没有路了 统计当前是否是最优值
            if dfs1_stop == 1 or stop_flag == 1:
                return max_orders,hit_max_times
            # print("aft ",len(order),max_orders,step -1,cur_area + this_area)
            if len(order) > max_orders:
                orders = []
                for subm in order:
                    orders.append(subm)
                    # print(subm)
                max_orders = len(order)
                hit_max_times = 0

            if len(order) == max_orders:
                if hit_max_times >= next_solve_min_times:
                    copy_order = list(order)
                    final_order_pre.append(copy_order)
                    new_matrix = np.copy(matrix)
                    for subm in copy_order:
                        a,b,c,d,_,_ = subm
                        for i in range(a,c+1):
                            for j in range(b,d+1):
                                new_matrix[i,j] = 0
                    this_submatrices = list(submatrices)
                    this_indegree = copy.deepcopy(indegree)
                    # print("before:")
                    # for j in range(sub_matrix_lens):
                    #     print(submatrices[j],indegree[j])
                    solve(new_matrix)
                    submatrices = this_submatrices
                    indegree = this_indegree
                    # print("after:")
                    # for j in range(sub_matrix_lens):
                    #     print(submatrices[j],indegree[j])
                    final_order_pre.pop()

                hit_max_times += 1
                if hit_max_times >= next_solve_max_times:
                    dfs1_stop = 1        
            s = order.pop()
            # print("pop",s)
            for neighbor in graph[i]:
                indegree[neighbor] += 1
            for j in visited_ids:
                vis[j] = 0
            # print("back")
    return max_orders,hit_max_times
# 在当前matrix状态下最多的得分
def solve(matrix):
    global stop_flag
    global accepted_target


    # pause()
    global final_order
    global final_order_pre
    global max_score
    # print(matrix)
    if stop_flag == 1:
        return 
    submatrices = []
    r, c = rows,cols 
    id = 0
    max_orders = 0
    hit_max_times = 0
    all_sum = matrix.sum() // 10
    
    id = 0
    
    # Step 1: Find all valid submatrices
    for i in range(r):
        for j in range(c):
            for k in range(i, r):
                for l in range(j, c):
                    # Check if the sum is a multiple of 10
                    sub_sum = matrix[i:k+1, j:l+1].sum()
                    # print(i,j,k,l,sub_sum)
                    if sub_sum > 0 and sub_sum % 10 == 0:
                        submatrix = [i,j,k,l,sub_sum,id]
                        submatrices.append(submatrix)
                        id +=1 

    
    # Step 2: Build dependency graph
    graph = defaultdict(list)
    indegree = defaultdict(int)
    sub_matrix_lens = len(submatrices)
    lap = np.zeros((sub_matrix_lens, sub_matrix_lens), dtype=int)
    for i, sm1 in enumerate(submatrices):
        for j, sm2 in enumerate(submatrices):
            if i != j and check_full_overlap(sm1,sm2):  # Check full overlap
                # print(i,sm1,j,sm2)
                graph[j].append(i)
                indegree[i] += 1
            elif i!=j and check_overlap(sm1,sm2): # check overlap
                lap[i,j] = lap[j,i] = 1

    
    # Step 3: Topological sorting
    for i in range(len(submatrices)):
        # print(submatrices[i])
        if indegree[i] == 0 : 
            _,_,_,_,sub_sum,_ = submatrices[i]
            if sub_sum !=10:
                indegree[i] = 1000000
                for nabor in graph[i]:
                    indegree[nabor] = 100000
        # print(submatrices[i],indegree[i])
    zero_cnt = 0
    for i in range(len(submatrices)):
        if indegree[i] == 0 :
            zero_cnt = 1
            break
    if zero_cnt == 1:
        vis = np.zeros((sub_matrix_lens), dtype=int)
        order = []
        # print(all_sum)
        dfs1_stop = 0
        orders = []
        dfs(matrix,dfs1_stop,orders,lap,submatrices,max_orders,hit_max_times,vis,graph,indegree,0,all_sum,order,0)

    # dfs返回说明当前matrix的遍历已经结束
    if stop_flag == 1:
        return
    this_score = 0
    for i in range(r):
        for j in range(c):
            if matrix[i,j] == 0:
                this_score += 1

    print(this_score,max_score)

    if this_score > max_score:
        max_score = this_score
        final_order = list(final_order_pre)
        print(final_order)
        print(this_score,max_score)
        print("solve: dfs fin")
        print(matrix)
        if max_score > accepted_target:
            print(final_order)
            print(this_score,max_score)
            print("solve: dfs fin")
            print(matrix)
            # pause()
            stop_flag = 1
    cur_time = time.time()
    if cur_time - start_time > 102:
        stop_flag = 1



    
    # if orders != []:
    #     pause()
    #     drag_rectangle_with_maze_id(0,0,0,0)
    #     for subm in orders: 
    #         a,b,c,d,_,_ = subm
    #         print(subm)
    #         sleep(0.3)
    #         drag_rectangle_with_maze_id(a,b,c,d)

    # for subm in orders:
    #     a,b,c,d,_,_ = subm
    #     for i in range(a,c+1):
    #         for j in range(b,d+1):
    #             matrix[i,j] = 0
    # print(type(matrix))
    # print(matrix)
    # pause()
    # solve()

def entry():
    global final_order
    global max_score
    global max_sum
    global stop_flag
    global cur_num
    stop_flag = 0
    cur_num = 0 
    max_sum = matrix.sum()
    max_score = 0
    final_order = []
    solve(matrix)
    print("max score: ",max_score)

    # pause()
    for order in final_order:
        drag_rectangle_with_maze_id(0,0,0,0)
        for subm in order: 
            a,b,c,d,_,_ = subm
            print(subm)
            sleep(0.2)
            drag_rectangle_with_maze_id(a,b,c,d)

    pass


if __name__ == "__main__":
    # print("请在屏幕上按住鼠标左键，从左上角框选一个矩形，然后释放左键...")
    # with mouse.Listener(on_click=on_click, on_move=on_move) as listener:
    #     listener.join()
    while True:
        retry_position = (1339.5, 743.29296875)
        start_position = (1141.17578125, 250.79296875)
        end_position = (1573.58203125, 940.92578125)
        retry_x,retry_y = retry_position
        start_time = time.time()
        mouse_click(retry_x,retry_y)
        mouse_click(retry_x,retry_y)
        mouse_click(retry_x,retry_y)
        # pause()
        sleep(4)


        if start_position and end_position:
            print(f"捕获的矩形坐标：起点 {start_position}，终点 {end_position}")
            start_x, start_y = start_position
            end_x, end_y = end_position
            # (1141, 250, 1573, 940)

            matrix = capture_and_recognize_matrix(start_position, end_position)
            ori_matrix = np.copy(matrix)
            entry()
            this_second = time.time()
            if this_second - start_time >= 130:
                continue
            else:
                sleep(130 - this_second + start_time)


            # print("开始模拟拖动...")
            # drag_rectangle(start_x, start_y, end_x, end_y)
        else:
            print("未捕获到有效的矩形坐标")
