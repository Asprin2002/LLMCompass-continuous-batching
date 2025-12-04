import sys
import os
import csv
import json
import pandas as pd

# 添加当前路径
sys.path.append(os.getcwd())

# 导入必要模块
from serving.scheduler import Scheduler
from software_model.selective_batching import TransformerBlockSelectiveBatchingTP
from software_model.utils import DataType, data_type_dict

# 导入硬件模型类
from hardware_model.system import System
from hardware_model.device import Device
from hardware_model.compute_module import ComputeModule, Core, VectorUnit, SystolicArray, Overhead
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule
from hardware_model.interconnect import InterConnectModule, LinkModule, TopologyType

# ==========================================
# 1. Hardware Loader (JSON -> Python Objects)
# ==========================================
def load_system_from_json(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # --- A. 解析 Interconnect ---
    ic_config = config['interconnect']
    link_config = ic_config['link']
    
    link_module = LinkModule(
        bandwidth_per_direction=link_config['bandwidth_per_direction_byte'],
        bandwidth_both_direction=link_config['bandwidth_both_directions_byte'],
        latency=link_config['latency_second'],
        flit_size=link_config['flit_size_byte'],
        max_payload_size=link_config['max_payload_size_byte'],
        header_size=link_config['header_size_byte']
    )
    
    # 拓扑类型映射
    topo_str = ic_config['topology']
    topology = TopologyType.FC if topo_str == "FC" else TopologyType.RING
    
    interconnect = InterConnectModule(
        device_count=config['device_count'],
        topology=topology,
        link_module=link_module,
        link_count_per_device=ic_config['link_count_per_device']
    )

    # --- B. 解析 Device ---
    dev_config = config['device']
    chiplet = dev_config['compute_chiplet']
    core_cfg = chiplet['core']
    sys_arr_cfg = core_cfg['systolic_array']
    vec_unit_cfg = core_cfg['vector_unit']
    
    # 1. 构建 Core 组件
    # Vector Unit
    vector_unit = VectorUnit(
        total_vector_flops_per_cycle=vec_unit_cfg['vector_width'] * vec_unit_cfg['flop_per_cycle'], # 估算
        word_size=2 if vec_unit_cfg['data_type'] == 'fp16' else 4,
        flops_per_exp=35, # A100 default
        vector_width=vec_unit_cfg['vector_width'],
        vector_count=4, # A100 default from compute_module.py
        data_type=data_type_dict.get(vec_unit_cfg['data_type'], data_type_dict['fp16'])
    )
    
    # Systolic Array
    systolic_array = SystolicArray(
        array_height=sys_arr_cfg['array_height'],
        array_width=sys_arr_cfg['array_width'],
        mac_per_cycle=sys_arr_cfg['mac_per_cycle'],
        input_word_size=2, # fp16
        output_word_size=2
    )
    
    # Core
    core = Core(
        vector_unit=vector_unit,
        systolic_array=systolic_array,
        systolic_array_count=4, # A100 default (Tensor Cores per SM)
        SRAM_size=core_cfg['SRAM_KB'] * 1024
    )
    
    # 2. 构建 Compute Module
    # 计算 Memory Bandwidth (HBM)
    # bandwidth = active_channels * pins * bandwidth_per_pin / 8 (bits to bytes)
    io_cfg = dev_config['io']
    mem_bw = (io_cfg['memory_channel_active_count'] * 
              io_cfg['pin_count_per_channel'] * 
              io_cfg['bandwidth_per_pin_bit']) / 8
    
    # 计算 L2 bandwidth (Global Buffer)
    l2_bw = io_cfg['global_buffer_bandwidth_per_cycle_byte']
    
    compute_module = ComputeModule(
        core=core,
        core_count=chiplet['core_count'],
        clock_freq=dev_config['frequency_Hz'],
        l2_size=io_cfg['global_buffer_MB'] * 1024 * 1024,
        l2_bandwidth_per_cycle=l2_bw,
        overhead=Overhead(2.1e-5, 1.2e-5, 4.5e-5, 4.5e-5) # A100 Overhead
    )
    
    # 3. 构建 IO Module (HBM Bandwidth 放在这里)
    # LLMCompass 中 IOModule 通常指对外通信或内存带宽，这里我们填入 HBM 带宽
    # 注意：matmul.py 有时会读 io_module.bandwidth 作为内存带宽
    io_module = IOModule(
        bandwidth=mem_bw, 
        latency=1e-6
    )
    
    # 4. 构建 Memory Module (Capacity)
    memory_module = MemoryModule(
        memory_capacity=dev_config['memory']['total_capacity_GB'] * 1e9
    )
    
    device = Device(compute_module, io_module, memory_module)
    
    # --- C. 构建 System ---
    system = System(device, interconnect)
    return system

# ==========================================
# 2. Trace Generator
# ==========================================
def generate_dummy_trace(filepath):
    print(f"Generating dummy trace at {filepath}...")
    data = [
        [100, 10, 0],         
        [50,  20, 1000000],   
        [200, 5,  2000000],   
        [128, 128, 3000000],  
        [32,  32, 4000000]    
    ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['input_toks', 'output_toks', 'arrival_time_ns'])
        writer.writerows(data)

# ==========================================
# 3. Main Simulation
# ==========================================
def run_test():
    # 路径配置
    TRACE_FILE = "inputs/share-gpt-req100-rate10.tsv"
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" # 确保 config 文件存在
    HARDWARE_CONFIG = "configs/GA100.json" # 硬件配置文件路径
    
    # 1. 准备数据
    #generate_dummy_trace(TRACE_FILE)
    
    # 2. 加载硬件 (从 JSON)
    print(f"Loading hardware from {HARDWARE_CONFIG}...")
    try:
        system = load_system_from_json(HARDWARE_CONFIG)
    except FileNotFoundError:
        print(f"Error: Could not find hardware config at {HARDWARE_CONFIG}")
        return
    except Exception as e:
        print(f"Error loading hardware: {e}")
        return
    
    # 3. 初始化后端
    print("Initializing Backend...")
    backend = TransformerBlockSelectiveBatchingTP(
        d_model=4096,
        n_heads=32,
        device_count=1,
        data_type=DataType('fp16', 2) # FP16
    )
    
    # 4. 初始化调度器
    print("Initializing Scheduler...")
    try:
        scheduler = Scheduler(
            model=MODEL_NAME,
            max_batch=64,
            npu_num=1,
            npu_group=1,
            npu_mem=40,
            fp=16,
            block_size=16,
            req_num=10000,
            verbose=False
        )
        scheduler.generate(TRACE_FILE)
    except Exception as e:
        print(f"Scheduler Init Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Loaded {len(scheduler.request)} requests.")
    
    # 5. 模拟循环
    current_time = 0.0 
    total_steps = 0
    
    print("\n--- Starting Simulation Loop ---\n")
    
# 防止死循环的计数器
    stuck_counter = 0 
    total_iterations = 0
    
    while not scheduler.is_request_empty():
        batch = scheduler.schedule(current_time, sys=0)
        
        if batch is None:
            # --- 调度失败处理逻辑 ---
            if len(scheduler.request) > 0:
                next_req = scheduler.request[0]
                next_arrival = next_req.arrival
                
                if next_arrival > current_time:
                    # 情况 A: 没请求，跳到下一个请求到达时间
                    current_time = next_arrival
                    stuck_counter = 0 # 重置计数器
                else:
                    # 情况 B: 有请求已到达 (Ready)，但调度器不干活
                    # 极有可能是内存满了，或者调度器逻辑卡住
                    stuck_counter += 1
                    current_time += 100000 # 步长增加到 100us，防止太慢
                    
                    # 如果连续 100 次都在忙等待，说明出问题了，打印 Debug 信息
                    if stuck_counter % 100 == 0:
                        print(f"[Warning] Stuck at {current_time/1e6:.3f} ms. "
                              f"Next Req ID: {next_req.id}, Type: {'Prefill' if next_req.is_init else 'Decode'}, "
                              f"Input: {next_req.input}")
                        
                        # 强制打印一次内存状态 (需要 MemoryModel 支持，或者我们预估一下)
                        mem = scheduler.memory
                        print(f"   >>> Memory Used: {mem.used_mem/1e9:.2f} GB / {mem.npu_mem/1e9:.2f} GB")
                        
                        # 如果是 Decode 请求卡住，通常是因为 KV Cache 预估过大
                        if not next_req.is_init:
                            # 尝试手动查看 block 大小
                            kv_need = mem.get_block_kv([next_req], 1)
                            print(f"   >>> This req needs KV size: {kv_need/1e9:.4f} GB")

                    # 如果卡了太久 (例如空转了 10万次)，强制退出防止死机
                    if stuck_counter > 100000:
                        print("Error: Simulation Deadlock Detected! Scheduler refuses to schedule ready requests.")
                        break
            else:
                # 只有 inflight，没有 waiting request
                current_time += 100000
            continue
        
        # --- 调度成功，重置计数器 ---
        stuck_counter = 0
        
        # ... (后续计算延迟的代码保持不变) ...
        # 计算延迟
        try:
            layer_latency_sec = backend(batch.requests, system)
        except AttributeError as e:
            print(f"Backend Error: {e}")
            raise e
            
        iteration_latency_ns = layer_latency_sec * 32 * 1e9
        finish_time = current_time + iteration_latency_ns
        
        # 打印信息
        p_count = sum(1 for r in batch.requests if r.is_init)
        d_count = sum(1 for r in batch.requests if not r.is_init)
        batch_type_str = []
        if p_count > 0: batch_type_str.append(f"{p_count} P")
        if d_count > 0: batch_type_str.append(f"{d_count} D")
        type_summary = "+".join(batch_type_str)
        
        print(f"[Time {current_time/1e6:8.2f} ms] Batch {batch.batch_id:3} | Size: {len(batch.requests):2} | {type_summary:^10} | Latency: {iteration_latency_ns/1e6:6.3f} ms")
        
        scheduler.add_done(batch.batch_id, sys=0, finish=finish_time)
        current_time = finish_time
        total_iterations += 1

    print("\n--- Simulation Finished ---")
    scheduler.print_result()

if __name__ == "__main__":
    run_test()