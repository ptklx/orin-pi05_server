import dataclasses
import logging
import socket
# import time
from pprint import pformat
import draccus
import torch
import asyncio
import websockets
import numpy as np
import json
import os
import signal
import msgpack_numpy_pt as msgpack_numpy

# 禁用 Hugging Face 自动下载
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# 配置日志以减少不必要的调试信息
import logging
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)


from step3_infer_onnx import pi05_create_infer

# 配置日志
logger = logging.getLogger(__name__)

# 初始值为默认的 480 和 640
img_height: int = 224
img_width: int = 224
current_dir = os.path.dirname(os.path.abspath(__file__))


# --- PolicyServer 的配置参数 ---

@dataclasses.dataclass
class PolicyServerConfig:
    """PolicyServer 的配置参数."""

    policy_type: str ='pi05'
    default_prompt: str | None = None
    port: int = 16673
    record: bool = False
    # inference_latency: float = 0.033 
    # fps: float = 30

def create_policy(cfg: PolicyServerConfig):
    try:
        policy=pi05_create_infer()
    except Exception as e:
        logger.error(f"模型加载前检查失败: {e}")
        raise
    return policy

class PolicyServer:
    def __init__(self, config: PolicyServerConfig):
        self.config = config
        logger.info(f"加载策略: 类型='{config.policy_type}'")
        self.policy = create_policy(config) # 加载策略模型
        # 显式设置设备
        self.device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_metadata = dataclasses.asdict(self.policy.config) # 获取策略配置作为元数据
        self._packer = msgpack_numpy.Packer()

        def infer(obs):
            # 推理并转换动作 (ONNX 需要 numpy 输入)
            noise = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(1, 50, 32),
            ).astype(np.float32)
            with torch.no_grad():
                action = self.policy.select_action(obs,noise)
                
            action_np = action #to_numpy(action)
            
            # 确保返回的动作字典包含 'action' 字段
            if not isinstance(action_np, dict):
                action_np = {"action": action_np}
            elif 'action' not in action_np and 'actions' not in action_np:
                action_np = {"action": list(action_np.values())[0]}
            
            # 记录模式
            if self.config.record:
                with open("policy_records.txt", "a", encoding='utf-8') as f:
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            if isinstance(obj, bytes):
                                try:
                                    return obj.decode('utf-8')
                                except UnicodeDecodeError:
                                    try:
                                        return obj.decode('latin-1')
                                    except UnicodeDecodeError:
                                        import base64
                                        return base64.b64encode(obj).decode('ascii')
                            return json.JSONEncoder.default(self, obj)
                    f.write(json.dumps({"obs": obs, "action": action_np}, cls=NumpyEncoder, ensure_ascii=False) + "\n")
            
            return action_np        
        
        self.policy.infer = infer

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info("创建 WebSocket 服务器 (host: %s, ip: %s)", hostname, local_ip)
        
        self._websocket_server = None # Will be set by serve function

    async def handle_websocket_connection(self, websocket):
        """处理单个 WebSocket 客户端连接."""
        client_id = websocket.remote_address
        logger.info(f"客户端 {client_id} 已连接.")

        # 发送策略元数据给客户端
        metadata_message = self._packer.pack(self.policy_metadata)
        await websocket.send(metadata_message)
        logger.info(f"已向客户端 {client_id} 发送策略元数据 (大小: {len(metadata_message)} 字节).")

        try:
            while True:
                # 接收观测数据
                try:
                    message = await websocket.recv()
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"客户端 {client_id} 连接已关闭.")
                    break
                
                # Deserialization (使用 msgpack_numpy 反序列化观测数据)
                # nosec - this is a known risk, but implied by the client's design
                obs_data = msgpack_numpy.unpackb(message)

                # Call policy inference
                action_data = self.policy.infer(obs_data)

                # 尝试将动作数据转换为 numpy 数组
                def convert_to_numpy(data):
                    # 如果是字典，尝试转换其中的 NumPy 数组
                    if isinstance(data, dict):
                        converted_dict = {}
                        for k, v in data.items():
                            if isinstance(v, torch.Tensor):
                                converted_dict[k] = v.cpu().numpy()
                            elif isinstance(v, np.ndarray):
                                converted_dict[k] = v
                            else:
                                converted_dict[k] = v
                        return converted_dict
                    
                    # 处理 Tensor
                    if isinstance(data, torch.Tensor):
                        return data.cpu().numpy()
                    
                    # 处理列表
                    if isinstance(data, list):
                        return np.array(data)
                    
                    return data

                try:
                    converted_action_data = convert_to_numpy(action_data)
                except Exception as e:
                    logger.error(f"动作数据转换失败: {e}")
                    converted_action_data = action_data

                # Serialization (使用 msgpack_numpy 序列化动作数据)
                response_message = self._packer.pack(converted_action_data) # default=str 确保所有不可序列化的类型都能被处理

                # Send action data back to client
                try:
                    await websocket.send(response_message)
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"客户端 {client_id} 连接在发送响应时关闭.")
                    break
                
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"客户端 {client_id} 已正常断开连接.")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"客户端 {client_id} 连接异常断开: {e}")
        except Exception as e:
            logger.error(f"处理客户端 {client_id} 时发生错误: {e}", exc_info=True)
        finally:
            try:
                await websocket.close(code=1000, reason="Server closing connection")
            except Exception:
                pass
            logger.info(f"客户端 {client_id} 连接处理结束.")

    async def start_websocket_server(self):
        """启动 WebSocket 服务器."""
        self._websocket_server = await websockets.serve(
            self.handle_websocket_connection,
            "0.0.0.0",
            self.config.port,
            max_size=100 * 1024 * 1024,
            compression=None  # 禁用压缩以减少开销
        )
        logger.info(f"WebSocket PolicyServer 已在 ws://0.0.0.0:{self.config.port} 启动")
        await self._websocket_server.wait_closed()

    def serve_forever(self) -> None:
        """启动 WebSocket 服务器并一直运行 (同步入口点)."""
        asyncio.run(self.start_websocket_server())

    def stop(self):
        """停止服务器."""
        logger.info("服务器正在停止...")
        if self._websocket_server:
            self._websocket_server.close()
            # asyncio.run(self._websocket_server.wait_closed()) # This line might block if not called correctly
        logger.info("服务器已停止.")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """使用给定配置启动 PolicyServer."""
    logger.info(pformat(dataclasses.asdict(cfg)))

    policy_server = PolicyServer(cfg)
    try:
        policy_server.serve_forever()
    except KeyboardInterrupt:
        logger.info("关闭服务器...")
    finally:
        policy_server.stop()
        logger.info("服务器已终止")

def signal_handler(sig, frame):
    print(f"收到信号 {sig}，正在安全退出...")
    raise KeyboardInterrupt

if __name__ == "__main__":
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal_handler)
    logging.basicConfig(level=logging.INFO, force=True) # force=True 确保basicConfig重新配置
    serve()

# python ./policy_server_v2.py --port=13668 --policy_type=act --record=True
#
#.   sudo docker exec -it noetic_VR /bin/bash

'''
python lerobot_infer_v6.py --port 16668 \
    --horizon_steps=30 --num_steps=1000 \
    --prompt "把两个面包放到一个蓝色篮子里面"


'''
#
