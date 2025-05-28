import json
from multiprocessing import shared_memory
import os
import socket
import subprocess
import time
import re
import threading
from pandia.agent.action import Action
from pandia.agent.env_config import ENV_CONFIG
from pandia.agent.utils import sample
from pandia.constants import K, M, WEBRTC_RECEIVER_CONTROLLER_PORT, WEBRTC_SENDER_SB3_PORT

def parse_rangable(value):
    if type(value) is str and '-' in value:
        return [v for v in value.split('-')]
    else:
        return value


class ClientProtocol():
    def __init__(self) -> None:
        super().__init__()
        self.shm = shared_memory.SharedMemory(name="pandia", create=True, 
                                              size=Action.shm_size())
        self.process = None
        self.bw = parse_rangable(os.getenv('BANDWIDTH', 10 * M))
        self.delay = parse_rangable(os.getenv('DELAY', 0))
        self.loss = parse_rangable(os.getenv('LOSS', 0))
        self.height = os.getenv('WIDTH', 720)
        self.fps = int(os.getenv('FPS', 25))
        self.obs_socket_path = os.getenv('OBS_SOCKET_PATH', '')
        self.ctrl_socket_path = os.getenv('CTRL_SOCKET_PATH', '')
        self.logging_path = os.getenv('LOGGING_PATH', '')
        self.sb3_logging_path = os.getenv('SB3_LOGGING_PATH', '')
        if os.path.exists(self.logging_path):
            os.remove(self.logging_path)
        if os.path.exists(self.sb3_logging_path):
            os.remove(self.sb3_logging_path)
        print(f'bw: {self.bw}, delay: {self.delay}, loss: {self.loss}, '
              f'obs_socket_path: {self.obs_socket_path}, '
              f'ctrl_socket_path: {self.ctrl_socket_path}', flush=True)


    # def start_simulator(self, bw = "02651.json", delay = 30, loss = 0, jitter = 0):
    #     # bw = int(bw / K)
    #     bw_file = "/app/traffic_shell/trace_data/" + bw
    #     # bw_file = bw
    #     print(f'Start sender with bandwidth: {bw_file} file, delay {delay}ms, loss {loss}, jitter {jitter}ms', flush=True)
    #     os.system("chmod +x /app/traffic_shell/tc.sh")
    #     with open("/app/traffic_shell/tc_log.log", "w") as log_file_net:
    #         subprocess.Popen([f"/app/traffic_shell/tc.sh", str(bw_file), str(delay), str(loss), str(jitter)], stdout=log_file_net, stderr=log_file_net)
    #     obs_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    #     obs_sock.connect(self.obs_socket_path)
    #     get_tbf_rate(obs_sock)
    #     self.sb3_logging_path = f'/app/media/sb3.log'
    #     if self.sb3_logging_path:
    #         log_file = open(self.sb3_logging_path, 'w')
    #         print(f'Logging to {self.sb3_logging_path}', flush=True)
    #     else:
    #         log_file = subprocess.DEVNULL
    #     self.process = \
    #         subprocess.Popen(['/app/simulation_video_save', # simulation_ztw_12.29
    #                           '--obs_socket', self.obs_socket_path,
    #                           '--resolution', str(self.height), '--fps', str(self.fps),
    #                           '--logging_path', self.logging_path,
    #                           '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/WebRTC-FrameDropper/Disabled', 
    #                           '--path', '/app/media',
    #                           '--dump_path', '/app/media/res_video'],
    #                           stdout=log_file, stderr=log_file, shell=False)

    def start_simulator(self, bw="02651.json", delay=30, loss=0, jitter=0):
        # ==== 2. 重启 tc.sh 网络限制 ====
        bw_file = "/app/traffic_shell/trace_data/" + bw
        print(f'Start sender with bandwidth: {bw_file} file, delay {delay}ms, loss {loss}, jitter {jitter}ms', flush=True)

        os.system("chmod +x /app/traffic_shell/tc.sh")
        tc_log_file = open("/app/traffic_shell/tc_log.log", "w")
        
        self.tc_process = subprocess.Popen(
            ["/app/traffic_shell/tc.sh", str(bw_file), str(delay), str(loss), str(jitter)],
            stdout=tc_log_file,
            stderr=tc_log_file
        )

        # ==== 3. 与观测器通信 ====
        obs_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        obs_sock.connect(self.obs_socket_path)
        get_tbf_rate(obs_sock)

        # ==== 4. 启动视频模拟子进程 ====
        self.sb3_logging_path = f'/app/media/sb3.log'
        log_file = open(self.sb3_logging_path, 'w') if self.sb3_logging_path else subprocess.DEVNULL

        self.process = subprocess.Popen(
            ['/app/simulation_video_save',
            '--obs_socket', self.obs_socket_path,
            '--resolution', str(self.height), '--fps', str(self.fps),
            '--logging_path', self.logging_path,
            '--force_fieldtrials=WebRTC-FlexFEC-03-Advertised/Enabled/WebRTC-FlexFEC-03/Enabled/WebRTC-FrameDropper/Disabled',
            '--path', '/app/media',
            '--dump_path', '/app/media/res_video'],
            stdout=log_file, stderr=log_file, shell=False
        )

    
    def datagram_received(self, data: bytes, addr) -> None:
        # Kill the simulation 
        if data[0] == 0:
            print(f'[{time.time()}] Received kill command', flush=True)
            # ==== 1. 清理旧的 tc.sh 和 video simulation 进程 ====
            if hasattr(self, 'tc_process') and self.tc_process is not None:
                try:
                    self.tc_process.terminate()
                    self.tc_process.wait(timeout=3)
                    print("[start_simulator] Previous tc.sh process terminated.", flush=True)
                except Exception as e:
                    print(f"[start_simulator] Failed to terminate tc.sh process: {e}", flush=True)
                    self.tc_process.kill()
                self.tc_process = None

            if hasattr(self, 'process') and self.process is not None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=3)
                    print("[start_simulator] Previous simulation_video_save process terminated.", flush=True)
                except Exception as e:
                    print(f"[start_simulator] Failed to terminate simulation_video_save process: {e}", flush=True)
                    self.process.kill()
                self.process = None
        # Send the action
        elif data[0] == 1:
            data = data[1:]
            data_str = ''.join('{:02x}'.format(x) for x in data[:16])
            print(f'[{time.time()}] Received action: {data_str}', flush=True)
            assert len(data) == len(self.shm.buf), f'Invalid action size: {len(data)} != {len(self.shm.buf)}'
            self.shm.buf[:] = data[:]
        elif data[0] == 2:
            config = json.loads(data[1:].decode()) if len(data) > 1 else {}
            print(f'[{time.time()}] Received start command: {config}', flush=True)
            self.start_simulator(config['bw'], config['delay'], config['loss'], config['jitter'])
        else:
            print(f'Unknown command: {data[0]}', flush=True)

def get_tbf_rate(sock):
    # 执行 tc 命令获取 TBF 速率
    cmd = ["tc", "-s", "qdisc", "show", "dev", "lo"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    output = result.stdout

    # 正则匹配速率（示例输出：rate 1Mbit）
    match = re.search(r"qdisc tbf .*? rate (\d+)(\w+)", output)
    rate = {"rate": int(match.group(1))} if match else {"rate": 0}
    buf = bytearray(1)
    buf[0] = 4
    buf += json.dumps(rate).encode()
    sock.send(buf)
    print(f'[{time.time()}] Send rate: {rate}', flush=True)
    threading.Timer(0.06, get_tbf_rate, [sock]).start()


def main():
    client = ClientProtocol()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    ctrl_sock_path = client.ctrl_socket_path
    print(f'Connecting to {ctrl_sock_path}...', flush=True)
    sock.bind(ctrl_sock_path)
    os.chmod(ctrl_sock_path, 0o777)
    while True:
        data, addr = sock.recvfrom(1024)
        client.datagram_received(data, addr)



if __name__ == '__main__':
    main()
