import json
import os
import base64
import sys
import tempfile
import subprocess
from openai import OpenAI
from dronecontroller import DroneController


def safe_execute_code(code_string: str, timeout_seconds: int = 15) -> str:
    """
    在一个新的子进程中安全地执行Python代码字符串，并返回其输出。
    使用临时文件方式，避免命令行参数长度限制和编码问题。
    """
    try:
        # 方案1: 使用临时文件 (推荐)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(code_string)
            temp_file_path = temp_file.name

        try:
            # 执行临时文件
            result = subprocess.run(
                [sys.executable, temp_file_path],  # 使用当前Python解释器
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                encoding='utf-8',  # 明确指定编码
                errors='replace'  # 处理编码错误
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                if not output:
                    return "代码成功执行，没有输出。"
                return f"代码成功执行。输出:\n{output}"
            else:
                error_msg = result.stderr.strip()
                return f"代码执行出错:\n{error_msg}"

        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass

    except subprocess.TimeoutExpired:
        return f"错误：代码执行超时（超过 {timeout_seconds} 秒）。"
    except UnicodeError as e:
        return f"编码错误: {str(e)}"
    except Exception as e:
        return f"执行代码时发生未知错误: {str(e)}"


class DroneAI:
    def __init__(self, config_file='fe8_config.json', use_model="gpt-4o-mini", drone_controller=None):
        """初始化AI控制系统"""
        with open(config_file) as f:
            config = json.load(f)
            os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
            os.environ["OPENAI_BASE_URL"] = config["OPENAI_BASE_URL"]
        self.use_model = use_model
        self.client = OpenAI()
        if drone_controller is None:
            self.drone = DroneController()
        else:
            self.drone = drone_controller
        self.messages = [
            {"role": "system", "content": """你是一个专业的无人机飞行员AI助手。
            你可以控制无人机执行各种飞行任务，包括：
            - 移动到指定位置
            - 相对移动
            - 旋转
            - 拍照
            - 分析拍摄的图片内容
            - 获取状态信息
            - "代码编写与执行"能力：你可以使用 `execute_python_code` 工具来动态编写并运行Python代码。
            这个能力对于执行复杂数学计算、处理文本、或实现其他工具未直接提供的自定义逻辑非常有用。
            代码将在一个隔离的安全环境中执行。你必须使用 `print()` 函数来输出结果，这样系统才能捕获并使用它。

            重要执行流程：
            1. 当收到用户任务时，首先使用 create_task_plan 工具创建详细的执行计划
            2. 任务计划应该包含所有需要执行的步骤，包括动作、描述和参数
            3. 创建计划后，按照计划依次执行每个步骤
            4. 每个步骤执行完成后继续下一步，直到所有任务完成
            5. 不要在中途停止等待用户输入

            任务分解原则：
            - 将复杂任务拆分为具体的操作步骤
            - 每个步骤应该是一个明确的动作，如果做不到则不去执行
            - 考虑步骤之间的依赖关系和时序
            - 为复杂动作添加必要的状态检查和等待时间

            坐标系说明：X轴向前，Y轴向左，Z轴向上。
            区域中心(0,0,0)
            角度单位为度，距离单位为米。
            步进一步最少为0.1米。

            图片分析能力：
            - 你可以分析拍摄的图片，识别其中的物体、场景、人物等
            - 可以描述图片的内容、颜色、构图等视觉信息
            - 可以回答关于图片内容的具体问题
            - 分析结果可以用于后续的飞行决策

            示例任务分解：
            任务："飞到(5,0,10)然后拍照并分析图片内容"
            步骤1: 移动到位置(5,0,10) 
            步骤2: 拍摄照片 
            步骤3: 分析拍摄的图片内容

            任务："飞到(5,0,10)然后正方形区域监视"
            步骤1: 移动到初始监视位置(5,0,10) 
            步骤2: 拍摄照片 
            步骤3: 分析拍摄的图片内容
            步骤4: 向前飞一定距离
            步骤5: 拍摄照片 
            步骤6：分析拍摄的图片内容
            步骤7: 旋转90度
            步骤8：重复步骤4，5，6，7直到回到始监视位置


            任务："寻找红色物体并拍照记录"
            步骤1: 移动到搜索起始位置
            步骤2: 拍摄照片
            步骤3: 分析图片，检查是否有红色物体
            步骤4: 如果发现目标，记录位置信息；如果未发现，继续移动搜索，每次移动0.5米方向任意

            任务："侦察一个以当前位置为中心，半径为10米的圆形区域"
            步骤1: 调用
            `get_drone_status`
            获取当前位置作为圆心。
            步骤2: 调用
            `execute_python_code`，生成一段Python代码来计算该圆形路径上的8个坐标点，并用print()
            打印成一个JSON格式的列表。
            步骤3: 系统接收到包含坐标点列表的字符串。
            步骤4: 依次调用
            `move_to_position`
            飞到每一个坐标点。
            """}

        ]

    def setup_drone(self):
        """设置无人机"""
        self.drone.connect()
        self.drone.start_simulation()
        self.drone.initialize_objects()
        print("Drone setup completed")

    def encode_image_to_base64(self, image_path):
        """将图片编码为base64格式"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    def get_tools(self):
        """定义可用的工具"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_task_plan",
                    "description": "创建详细的任务执行计划，将复杂任务分解为具体步骤",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_description": {"type": "string", "description": "原始任务描述"},
                            "steps": {
                                "type": "array",
                                "description": "任务步骤列表",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step_number": {"type": "integer", "description": "步骤编号"},
                                        "action": {"type": "string", "description": "要执行的动作"},
                                        "description": {"type": "string", "description": "步骤详细描述"},
                                        "parameters": {"type": "object", "description": "执行参数"}
                                    }
                                }
                            }
                        },
                        "required": ["task_description", "steps"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_to_position",
                    "description": "移动无人机到指定的绝对位置",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number", "description": "X坐标位置"},
                            "y": {"type": "number", "description": "Y坐标位置"},
                            "z": {"type": "number", "description": "Z坐标位置（高度）"},
                            "duration": {"type": "number", "description": "移动持续时间（秒）", "default": 5.0}
                        },
                        "required": ["x", "y", "z"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "move_relative",
                    "description": "相对当前位置移动无人机",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dx": {"type": "number", "description": "X方向移动距离", "default": 0},
                            "dy": {"type": "number", "description": "Y方向移动距离", "default": 0},
                            "dz": {"type": "number", "description": "Z方向移动距离", "default": 0},
                            "duration": {"type": "number", "description": "移动持续时间（秒）", "default": 3.0}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "rotate_drone",
                    "description": "旋转无人机",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "yaw": {"type": "number", "description": "偏航角度（度）", "default": 0},
                            "pitch": {"type": "number", "description": "俯仰角度（度）", "default": 0},
                            "roll": {"type": "number", "description": "翻滚角度（度）", "default": 0},
                            "duration": {"type": "number", "description": "旋转持续时间（秒）", "default": 3.0}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "take_photo_downward",
                    "description": "使用【俯视】摄像头拍摄照片",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "保存的文件名",
                                         "default": "drone_photo_downward.jpg"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "take_photo_forward",
                    "description": "使用【前视】摄像头拍摄照片",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "保存的文件名",
                                         "default": "drone_photo_forward.jpg"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_image",
                    "description": "分析指定图片的内容，识别物体、场景等信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "string", "description": "要分析的图片文件路径"},
                            "analysis_prompt": {
                                "type": "string",
                                "description": "分析提示词，指定分析重点",
                                "default": "请详细描述这张图片的内容，包括看到的物体、场景、颜色、构图等信息。"
                            },
                            "specific_question": {
                                "type": "string",
                                "description": "对图片的具体问题（可选）",
                                "default": ""
                            }
                        },
                        "required": ["image_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_drone_status",
                    "description": "获取无人机当前状态信息",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "wait",
                    "description": "等待指定时间",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "seconds": {"type": "number", "description": "等待的秒数"}
                        },
                        "required": ["seconds"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": "在一个安全的环境中编写并执行一段Python代码。当需要进行复杂计算、实现自定义逻辑、或处理其他工具无法完成的任务时使用此功能。代码必须使用 print() 函数来返回结果。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "需要执行的Python代码字符串。"}
                        },
                        "required": ["code"]
                    }
                }
            }
        ]

    def print_task_plan(self, plan_data):
        """打印任务计划"""
        print(" 任务分解计划:")
        print("=" * 80)
        print(f" 原始任务: {plan_data['task_description']}")
        print(f" 总步骤数: {len(plan_data['steps'])}")
        print("-" * 80)

        for step in plan_data['steps']:
            print(f"步骤 {step['step_number']}: {step['action']}")
            print(f"    描述: {step['description']}")
            if 'parameters' in step and step['parameters']:
                print(f"    参数: {step['parameters']}")
            print()

        print("=" * 80)
        print()

    def analyze_image_with_ai(self, image_path, analysis_prompt, specific_question=""):
        """多模态大模型分析图片"""
        try:
            # 编码图片
            base64_image = self.encode_image_to_base64(image_path)
            if not base64_image:
                return "无法读取图片文件"

            # 构建分析消息
            content = [
                {"type": "text", "text": analysis_prompt}
            ]

            if specific_question:
                content.append({
                    "type": "text",
                    "text": f"\n\n具体问题: {specific_question}"
                })

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

            # 调用视觉模型分析
            response = self.client.chat.completions.create(
                # model="gpt-4o-mini",
                # model="Qwen2___5-7B-Instruct",
                model=self.use_model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=1000,
                temperature=0.5
            )

            analysis_result = response.choices[0].message.content
            return analysis_result

        except Exception as e:
            return f"图片分析出错: {str(e)}"

    def execute_tool(self, tool_call):
        """执行工具调用"""
        function_name = tool_call.function.name

        try:
            if isinstance(tool_call.function.arguments, str):
                args = json.loads(tool_call.function.arguments)
            else:
                args = tool_call.function.arguments
        except json.JSONDecodeError as e:
            return f"参数解析错误: {str(e)}"

        try:
            if function_name == "create_task_plan":
                self.print_task_plan(args)
                return f"Task plan created with {len(args['steps'])} steps"


            elif function_name == "move_to_position":
                x = float(args["x"]) if "x" in args else 0.0
                y = float(args["y"]) if "y" in args else 0.0
                z = float(args["z"]) if "z" in args else 0.0
                duration = float(args.get("duration", 5.0))
                result = self.drone.move_to_pose(
                    [x, y, z],
                    duration=duration
                )
                return f"Successfully moved to position ({x}, {y}, {z})"

            elif function_name == "move_relative":
                dx = float(args.get("dx", 0))
                dy = float(args.get("dy", 0))
                dz = float(args.get("dz", 0))
                duration = float(args.get("duration", 3.0))

                result = self.drone.move_relative(dx, dy, dz, duration=duration)
                return f"Successfully moved relative ({dx}, {dy}, {dz})"

            elif function_name == "rotate_drone":
                yaw = float(args.get("yaw", 0))
                pitch = float(args.get("pitch", 0))
                roll = float(args.get("roll", 0))
                duration = float(args.get("duration", 3.0))

                result = self.drone.rotate(yaw, pitch, roll, duration=duration)
                return f"Successfully rotated by yaw:{yaw}°, pitch:{pitch}°, roll:{roll}°"

            elif function_name == "take_photo_downward":
                filename = args.get("filename", "drone_photo.jpg")
                image = self.drone.take_photo_downward(filename)
                if image is not None:
                    return f"Photo taken and saved as {filename}"
                else:
                    return "Failed to take photo"

            elif function_name == "take_photo_forward":
                filename = args.get("filename", "drone_photo_forward.jpg")
                image = self.drone.take_photo_forward(filename)
                if image is not None:
                    return f"Forward photo taken and saved as {filename}"
                else:
                    return "Failed to take forward photo"

            elif function_name == "analyze_image":
                image_path = args["image_path"]
                analysis_prompt = args.get("analysis_prompt",
                                           "请详细描述这张图片的内容，包括看到的物体、场景、颜色、构图等信息。")
                specific_question = args.get("specific_question", "")

                print(f"     正在分析图片: {image_path}")
                if specific_question:
                    print(f"     具体问题: {specific_question}")

                analysis_result = self.analyze_image_with_ai(image_path, analysis_prompt, specific_question)

                print(f"     分析结果:")
                print(f"    " + "─" * 60)
                for line in analysis_result.split('\n'):
                    print(f"    {line}")
                print(f"    " + "─" * 60)

                return analysis_result

            elif function_name == "get_drone_status":
                status = self.drone.get_status()
                return json.dumps(status, indent=2)

            elif function_name == "wait":
                seconds = args.get("seconds", 1)
                import time
                time.sleep(seconds)
                return f"Waited for {seconds} seconds"

            elif function_name == "execute_python_code":
                code_to_run = args["code"]
                print(f"     正在执行AI生成的代码...")
                # 调用安全执行函数
                execution_result = safe_execute_code(code_to_run)
                print(f"     代码执行结果: {execution_result}")
                return execution_result

        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"

    def chat(self, user_input):
        """与AI对话并执行命令"""
        self.messages.append({"role": "user", "content": user_input})

        print(f" AI正在分析任务: {user_input}")

        max_iterations = 20  # 防止无限循环
        iteration_count = 0

        while iteration_count < max_iterations:
            iteration_count += 1

            response = self.client.chat.completions.create(
                model=self.use_model,
                # model="Qwen2___5-7B-Instruct",
                # model="gpt-4o-mini",
                messages=self.messages,
                temperature=0.7,
                tools=self.get_tools()
            )

            assistant_message = response.choices[0].message
            self.messages.append(assistant_message)

            # 如果AI决定调用工具
            if assistant_message.tool_calls:
                print(f"执行步骤 {iteration_count}:")

                # 执行所有工具调用
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name

                    try:
                        if isinstance(tool_call.function.arguments, str):
                            args_display = json.loads(tool_call.function.arguments)
                        else:
                            args_display = tool_call.function.arguments
                    except:
                        args_display = "参数解析失败"

                    if function_name == "create_task_plan":
                        print(f"    创建任务计划...")
                    elif function_name == "analyze_image":
                        print(f"    执行: {function_name}")
                    else:
                        print(f"    执行: {function_name} - 参数: {args_display}")

                    result = self.execute_tool(tool_call)

                    if function_name != "create_task_plan" and function_name != "analyze_image":
                        print(f"    结果: {result}")

                    # 工具执行结果     ->    对话历史
                    self.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": result
                    })
                # 继续循环，让AI决定是否需要执行更多步骤
                continue
            else:
                # 不再调用工具，说明任务完成或需要用户输入
                print(" 任务执行完成!")
                return assistant_message.content

        print(" 达到最大执行步骤限制，任务可能未完全完成")
        return "任务执行达到步骤限制，请检查执行状态"

    def cleanup(self):
        """清理资源"""
        self.drone.stop_simulation()
        self.drone.disconnect()


# 使用示例
if __name__ == "__main__":
    # 初始化无人机AI系统
    drone_ai = DroneAI()
    drone_ai.setup_drone()

    try:
        # 示例任务1: 拍照并分析
        print("\n" + "=" * 60)
        print("示例任务1: 拍照并分析图片内容")
        print("=" * 60)
        response = drone_ai.chat("飞到坐标(3,0,2)位置，拍一张照片，然后分析照片中的内容")
        print(f"\n最终回复: {response}")

        # 示例任务2: 寻找特定物体
        print("\n" + "=" * 60)
        print("示例任务2: 寻找红色物体")
        print("=" * 60)
        response = drone_ai.chat("在当前区域搜索红色物体，如果发现就拍照记录，并分析是什么物体")
        print(f"\n最终回复: {response}")

        # 示例任务3: 分析已有图片
        print("\n" + "=" * 60)
        print("示例任务3: 分析指定图片")
        print("=" * 60)
        response = drone_ai.chat("分析图片文件 'drone_photo.jpg'，重点看看有没有人或车辆")
        print(f"\n最终回复: {response}")

    except KeyboardInterrupt:
        print("\n用户中断执行")
    finally:
        drone_ai.cleanup()
        print("程序结束")
