"""WebSocket 测试客户端 — 持续发送 test_openpi_example.pkl 数据到 policy_server_v2_onnx.py

用法:
    python policy_client_test.py [--host localhost] [--port 16678] [--interval 0.1] [--count 10]
"""

import argparse
import asyncio
import os
import pickle
import time

import numpy as np

# 复用 server 端相同的 msgpack_numpy 序列化
import msgpack_numpy_pt as msgpack_numpy


async def run_client(host: str, port: int, interval: float, count: int):
    import websockets

    # 加载测试数据
    path_file = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(path_file, "data_param/test_openpi_example.pkl")
    with open(pkl_path, "rb") as f:
        example = pickle.load(f)

    # 加入 prompt/task 字段（如果 pkl 里没有）
    if "task" not in example and "prompt" not in example:
        example["task"] = "pick up the object"

    packer = msgpack_numpy.Packer()
    uri = f"ws://{host}:{port}"
    print(f"连接到 {uri} ...")

    async with websockets.connect(uri, max_size=100 * 1024 * 1024) as ws:
        # 先接收策略元数据
        metadata_raw = await ws.recv()
        metadata = msgpack_numpy.unpackb(metadata_raw)
        print(f"收到策略元数据: {metadata}")

        for i in range(count):
            # 序列化观测数据
            packed = packer.pack(example)
            t0 = time.perf_counter()
            await ws.send(packed)

            # 接收动作
            response = await ws.recv()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            action_data = msgpack_numpy.unpackb(response)

            # 打印摘要
            if isinstance(action_data, dict):
                for k, v in action_data.items():
                    shape = v.shape if isinstance(v, np.ndarray) else "scalar"
                    print(f"[{i+1}/{count}] 延迟={elapsed_ms:.1f}ms  key={k}  shape={shape}")
            else:
                print(f"[{i+1}/{count}] 延迟={elapsed_ms:.1f}ms  type={type(action_data)}")

            if interval > 0 and i < count - 1:
                await asyncio.sleep(interval)

    print("测试完成.")


def main():
    parser = argparse.ArgumentParser(description="Policy Server WebSocket 测试客户端")
    # parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--host", type=str, default="192.168.110.52")
    parser.add_argument("--port", type=int, default=16678)
    parser.add_argument("--interval", type=float, default=0.5, help="发送间隔(秒)")
    parser.add_argument("--count", type=int, default=5, help="发送次数")
    args = parser.parse_args()

    asyncio.run(run_client(args.host, args.port, args.interval, args.count))


if __name__ == "__main__":
    main()
