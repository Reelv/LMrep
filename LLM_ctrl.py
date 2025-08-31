from dronellm import DroneAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

if __name__ == "__main__":
    drone_ai = DroneAI(config_file="modelscope.json", use_model='Qwen/Qwen3-235B-A22B-Instruct-2507')
    try:
        drone_ai.setup_drone()

        # 交互式控制
        print(" 无人机控制系统启动完成！")
        print(" 系统支持复杂长序列任务自动分解和执行")


        while True:
            user_input = input(" 用户指令: ")
            if user_input.lower() == 'quit':
                break

            print("=" * 60)
            response = drone_ai.chat(user_input)
            print(f" AI最终回复: {response}")
            print("=" * 60 + "\n")

    except Exception as e:
        print(f" Error: {e}")
    finally:
        drone_ai.cleanup()
