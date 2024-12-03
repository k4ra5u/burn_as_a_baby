import time
from pynput import mouse
from Quartz import CGEventCreateMouseEvent, CGEventPost, kCGHIDEventTap
from Quartz import kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp, kCGMouseButtonLeft
from PIL import ImageGrab, Image
import pytesseract
import numpy as np
import cv2
import easyocr
import io
from pwn import *
from alibabacloud_ocr_api20210707.client import Client as ocr_api20210707Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_darabonba_stream.client import Client as StreamClient
from alibabacloud_ocr_api20210707 import models as ocr_api_20210707_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
from collections import defaultdict, deque


rows = 16
cols = 10
matrix = None
use_api = True 
hit_max_times =0
dfs1_stop = 0
dfs2_stop = 0
# 全局变量存储鼠标拖动的起点和终点
start_position = None
end_position = None
is_selecting = False
lap = None
submatrices = []
sub_matrix_lens = 0
max_orders = 0
orders = []
config = '--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789'

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
            content_string = b'7 3 8 6 4 7 1 1 2 7 2 1 1 8 1 4 3 1 3 1 1 9 1 4 4 1 6 9 1 2 4 4 2 2 4 2 1 2 2 1 1 4 1 4 1 1 1 4 7 7 5 2 4 5 1 2 2 1 4 4 1 7 7 1'

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


def dfs(vis,graph,indegree,start,step,order):
    global max_orders
    global hit_max_times
    global dfs1_stop
    global orders
    if dfs1_stop == 1:
        return
    if step == 0 :
        for subm in order:
            print(subm)
        exit(0)
    visited_ids = []
    for i in range(start,sub_matrix_lens):
        if indegree[i] == 0 and vis[i] ==0 :
            order.append(submatrices[i])
            visited_ids.append(i)
            vis[i] = 1
            for neighbor in graph[i]:
                indegree[neighbor] -= 1
            for j in range(sub_matrix_lens):
                if lap[i][j] == 1 and vis[j] == 0:
                    visited_ids.append(j)
                    vis[j] = 1
            # print(i)
            dfs(vis,graph,indegree,0,step - 1,order)
            if dfs1_stop == 1:
                return
            print(len(order),max_orders,step -1)
            if len(order) > max_orders:
                orders = []
                for subm in order:
                    orders.append(subm)
                    print(subm)

                    
                max_orders = len(order)
                hit_max_times = 0
            if len(order) == max_orders:
                hit_max_times += 1
                if hit_max_times ==4550:
                    dfs1_stop = 1
                    pause()
                    drag_rectangle_with_maze_id(0,0,0,0)
                    for subm in order: 
                        a,b,c,d,_,_ = subm
                        print(subm)
                        sleep(0.3)
                        drag_rectangle_with_maze_id(a,b,c,d)
                    # exit(0)
            
            order.pop()
            for neighbor in graph[i]:
                indegree[neighbor] += 1
            for j in visited_ids:
                vis[j] = 0
            # print("back")
    pass

def solve():
    global sub_matrix_lens
    global lap
    global hit_max_times
    global matrix
    global submatrices
    global max_orders
    global dfs1_stop
    global all_sum
    global orders
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
                        # print(submatrix)

    
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
        print(submatrices[i])
        if indegree[i] == 0 : 
            _,_,_,_,sub_sum,_ = submatrices[i]
            if sub_sum !=10:
                indegree[i] = 1000  
    vis = np.zeros((sub_matrix_lens), dtype=int)
    order = []
    print(all_sum)
    dfs1_stop = 0
    orders = []
    dfs(vis,graph,indegree,0,all_sum,order)
    if dfs1_stop == 0 and orders != []:
        pause()
        drag_rectangle_with_maze_id(0,0,0,0)
        for subm in orders: 
            a,b,c,d,_,_ = subm
            print(subm)
            sleep(0.3)
            drag_rectangle_with_maze_id(a,b,c,d)

    for subm in order:
        a,b,c,d,_,_ = subm
        for i in range(a,c+1):
            for j in range(b,d+1):
                matrix[i,j] = 0
    pause()
    solve()

    # queue = deque([i for i in range(len(submatrices)) if indegree[i] == 0])
    # order = []
    # while queue:
    #     node = queue.popleft()
    #     order.append(node)
    #     for neighbor in graph[node]:
    #         indegree[neighbor] -= 1
    #         if indegree[neighbor] == 0:
    #             queue.append(neighbor)
    
    # # Step 4: Check if valid and return order
    # if len(order) == len(submatrices):
    #     for i in order:
    #         print(submatrices[i])
    #         a,b,c,d,sum,id = submatrices[i]
    #         drag_rectangle_with_maze_id(a,b,c,d)
    #         sleep(0.5)
    # else:
    #     print( None)  # No valid order found
    #     for i in order:
    #         print(submatrices[i])
    #         a,b,c,d,sum,id = submatrices[i]
    #         drag_rectangle_with_maze_id(a,b,c,d)
    #         sleep(0.2)

if __name__ == "__main__":
    print("请在屏幕上按住鼠标左键，从左上角框选一个矩形，然后释放左键...")
    with mouse.Listener(on_click=on_click, on_move=on_move) as listener:
        listener.join()

    if start_position and end_position:
        print(f"捕获的矩形坐标：起点 {start_position}，终点 {end_position}")
        start_x, start_y = start_position
        end_x, end_y = end_position

        matrix = capture_and_recognize_matrix(start_position, end_position)
        solve()

        # print("开始模拟拖动...")
        # drag_rectangle(start_x, start_y, end_x, end_y)
    else:
        print("未捕获到有效的矩形坐标")
