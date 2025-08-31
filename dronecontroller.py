import cv2
import numpy as np
import sim
import math
import time


class DroneController:
    def __init__(self, ip='127.0.0.1', port=19997):
        """初始化无人机控制器"""
        self.clientID = None
        self.target_handle = None
        self.current_pos = None
        self.current_ori = None
        self.ip = ip
        self.port = port

        # 为两个摄像头分别创建句柄变量
        self.downward_vision_sensor_handle = None  # 俯视摄像头 (原Vision_sensor)
        self.forward_vision_sensor_handle = None  # 前视摄像头 (新Vision_sensor0)

    def connect(self):
        """连接到CoppeliaSim"""
        self.clientID = sim.simxStart(self.ip, self.port, True, True, 5000, 5)
        if self.clientID == -1:
            raise Exception("Failed to connect to CoppeliaSim")
        print("Successfully connected to CoppeliaSim")
        return True

    def start_simulation(self):
        """启动仿真"""
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)
        print("Simulation started")

    def stop_simulation(self):
        """停止仿真"""
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
        print("Simulation stopped")

    def initialize_objects(self):
        """初始化无人机和传感器对象"""
        # 获取无人机目标对象
        res, self.target_handle = sim.simxGetObjectHandle(
            self.clientID, 'Quadricopter_target', sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            raise Exception("Failed to get Quadricopter_target handle")

        # 获取【俯视】摄像头 'Vision_sensor'
        res, self.downward_vision_sensor_handle = sim.simxGetObjectHandle(
            self.clientID, 'Vision_sensor', sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            raise Exception("Failed to get downward Vision_sensor handle")

        # 获取【前视】摄像头 'Vision_sensor0'
        res, self.forward_vision_sensor_handle = sim.simxGetObjectHandle(
            self.clientID, 'Vision_sensor0', sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            raise Exception("Failed to get forward Vision_sensor0 handle")

        # 初始化两个摄像头的图像流
        sim.simxGetVisionSensorImage(
            self.clientID, self.downward_vision_sensor_handle, 0, sim.simx_opmode_streaming)
        sim.simxGetVisionSensorImage(
            self.clientID, self.forward_vision_sensor_handle, 0, sim.simx_opmode_streaming)

        print("Objects (including both cameras) initialized successfully")

    def get_current_pose(self):
        """获取当前位置和姿态"""
        res, pos = sim.simxGetObjectPosition(
            self.clientID, self.target_handle, -1, sim.simx_opmode_blocking)
        res, ori = sim.simxGetObjectOrientation(
            self.clientID, self.target_handle, -1, sim.simx_opmode_blocking)
        if res != sim.simx_return_ok:
            raise Exception("Failed to get current pose")

        self.current_pos = pos
        self.current_ori = ori
        return pos, ori

    def move_to_pose(self, target_pos, target_ori=None, duration=5.0, steps_per_meter=100):
        """[改进方法] 平滑地移动到指定位置和姿态"""
        current_pos, current_ori = self.get_current_pose()
        if target_ori is None:
            target_ori = current_ori
        distance = math.sqrt(sum([(target_pos[i] - current_pos[i]) ** 2 for i in range(3)]))
        total_steps = max(50, int(distance * steps_per_meter))
        time_step = duration / total_steps
        need_orientation_interpolation = not np.allclose(current_ori, target_ori, atol=1e-3)
        for i in range(total_steps):
            eased_alpha = self._ease_in_out_sine((i + 1) / total_steps)
            interpolated_pos = [current_pos[j] + (target_pos[j] - current_pos[j]) * eased_alpha for j in range(3)]
            if need_orientation_interpolation:
                interpolated_ori = [current_ori[j] + (target_ori[j] - current_ori[j]) * eased_alpha for j in range(3)]
            else:
                interpolated_ori = current_ori
            sim.simxSetObjectPosition(self.clientID, self.target_handle, -1, interpolated_pos, sim.simx_opmode_oneshot)
            sim.simxSetObjectOrientation(self.clientID, self.target_handle, -1, interpolated_ori,
                                         sim.simx_opmode_oneshot)
            time.sleep(time_step)
        sim.simxSetObjectPosition(self.clientID, self.target_handle, -1, target_pos, sim.simx_opmode_oneshot)
        if need_orientation_interpolation:
            sim.simxSetObjectOrientation(self.clientID, self.target_handle, -1, target_ori, sim.simx_opmode_oneshot)
        # print(f"Smoothly moved to position: {target_pos}") # 在AI循环中会产生过多信息，可注释掉
        return True

    def _ease_in_out_sine(self, x):
        """一个简单的缓入缓出函数"""
        return -(math.cos(math.pi * x) - 1) / 2

    def move_relative(self, dx=0, dy=0, dz=0, duration=3.0):
        """相对当前位置和朝向移动"""
        current_pos, current_ori = self.get_current_pose()
        yaw = current_ori[2]
        world_dx = dx * math.cos(yaw) - dy * math.sin(yaw)
        world_dy = dx * math.sin(yaw) + dy * math.cos(yaw)
        target_pos = [current_pos[0] + world_dx, current_pos[1] + world_dy, current_pos[2] + dz]
        return self.move_to_pose(target_pos, duration=duration)

    def rotate(self, yaw_degrees=0, pitch_degrees=0, roll_degrees=0, duration=3.0):
        """旋转无人机（以度为单位）"""
        current_pos, current_ori = self.get_current_pose()
        target_ori = [
            current_ori[0] + math.radians(roll_degrees),
            current_ori[1] + math.radians(pitch_degrees),
            current_ori[2] + math.radians(yaw_degrees)
        ]
        return self.move_to_pose(current_pos, target_ori, duration=duration)

    # 内部通用的图像处理函数
    def _process_image(self, errCode, resolution, image, filename):
        if errCode == sim.simx_return_ok and image:
            sensor_frame = (np.array(image, dtype=np.int16) + 128).astype(np.uint8)
            # sensor_frame = np.array(image, dtype=np.uint8)
            sensor_frame.resize([resolution[1], resolution[0], 3])  # 注意CoppeliaSim的解析度顺序
            sensor_frame = cv2.cvtColor(sensor_frame, cv2.COLOR_RGB2BGR)
            sensor_frame = cv2.flip(sensor_frame, 0)
            if filename:
                cv2.imwrite(filename, sensor_frame)
                print(f"Photo saved as {filename}")
            return sensor_frame
        return None

    def take_photo_downward(self, filename=None):
        """使用【俯视】摄像头拍照"""
        errCode, resolution, image = sim.simxGetVisionSensorImage(
            self.clientID, self.downward_vision_sensor_handle, 0, sim.simx_opmode_buffer)
        if errCode != sim.simx_return_ok:
            print(f"Failed to capture downward image, error code: {errCode}")
            return None
        return self._process_image(errCode, resolution, image, filename)

    def take_photo_forward(self, filename=None):
        """使用【前视】摄像头拍照"""
        errCode, resolution, image = sim.simxGetVisionSensorImage(
            self.clientID, self.forward_vision_sensor_handle, 0, sim.simx_opmode_buffer)
        if errCode != sim.simx_return_ok:
            print(f"Failed to capture forward image, error code: {errCode}")
            return None
        return self._process_image(errCode, resolution, image, filename)

    def get_status(self):
        """获取无人机状态"""
        pos, ori = self.get_current_pose()
        return {
            "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
            "orientation": {
                "roll": math.degrees(ori[0]),
                "pitch": math.degrees(ori[1]),
                "yaw": math.degrees(ori[2])
            }
        }

    def disconnect(self):
        """断开连接"""
        if self.clientID is not None:
            sim.simxFinish(self.clientID)
            print("Disconnected from CoppeliaSim")



if __name__ == "__main__":
    # 创建控制器实例
    drone = DroneController(ip='127.0.0.1', port=19997)

    try:
        # 1. 连接到CoppeliaSim
        drone.connect()

        # 2. 启动仿真
        drone.start_simulation()

        # 3. 初始化对象（无人机+两个摄像头）
        drone.initialize_objects()

        # 4. 获取初始状态
        status = drone.get_status()
        print("Initial status:", status)

        # 5. 平移到某个新位置（向前移动 1 米）
        print("Moving forward 1 meter...")
        drone.move_relative(dx=1.0, dy=0.0, dz=0.0, duration=3.0)

        # 6. 拍摄俯视照片
        print("Taking downward photo...")
        downward_img = drone.take_photo_downward(filename="downward.jpg")

        # 7. 拍摄前视照片
        print("Taking forward photo...")
        forward_img = drone.take_photo_forward(filename="forward.jpg")

        # 8. 再旋转 90 度
        print("Rotating yaw +90°...")
        drone.rotate(yaw_degrees=90, duration=5.0)

        # 9. 获取旋转后的状态
        new_status = drone.get_status()
        print("After movement status:", new_status)

        # 等待 2 秒
        time.sleep(2)

        # 10. 停止仿真
        drone.stop_simulation()

    except Exception as e:
        print("Error:", e)

    finally:
        # 确保断开连接
        drone.disconnect()