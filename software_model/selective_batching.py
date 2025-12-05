from software_model.operators import Operator, Reshape, Transpose
from software_model.matmul import Matmul, BatchedMatmul
from software_model.softmax import Softmax
from software_model.layernorm import LayerNorm
from software_model.gelu import GeLU
from software_model.utils import Tensor, DataType
from software_model.communication_primitives import AllReduceMultiPCB
from hardware_model.system import System

class TransformerBlockSelectiveBatchingTP(Operator):
    def __init__(self, d_model, n_heads, device_count, data_type: DataType):
        super().__init__(0, 0, 0, 0, data_type)
        self.d_model = d_model
        self.n_heads = n_heads
        self.device_count = device_count
        self.d_h = d_model // n_heads
        
        # 参数张量定义
        d = d_model
        self.Wq = Tensor([d, d // device_count], data_type)
        self.Wk = Tensor([d, d // device_count], data_type)
        self.Wv = Tensor([d, d // device_count], data_type)
        self.W0 = Tensor([d // device_count, d], data_type)
        self.W1 = Tensor([d, 4 * d // device_count], data_type)
        self.W2 = Tensor([4 * d // device_count, d], data_type)
        
        # 实例化基础算子
        # Linear Layers (QKV, Out, FFN) - 这些将用于扁平化后的大 Tensor
        self.Q_proj = Matmul(data_type)
        self.K_proj = Matmul(data_type)
        self.V_proj = Matmul(data_type)
        self.H_matmul0 = Matmul(data_type) # Output proj
        self.H_matmul1 = Matmul(data_type) # FFN Up/Gate
        self.H_matmul2 = Matmul(data_type) # FFN Down
        
        # Activations & Norms
        self.layer_norm0 = LayerNorm(data_type)
        self.layer_norm1 = LayerNorm(data_type)
        self.H_gelu = GeLU(data_type)
        self.allreduce_mha = AllReduceMultiPCB(data_type)
        self.allreduce_ffn = AllReduceMultiPCB(data_type) # 用于 FFN 后的 AllReduce

        # Attention Specific Operators
        # 我们需要分别模拟 Prefill 和 Decode 的 Attention 计算
        self.Attn_Q_mul_K = BatchedMatmul(data_type)
        self.Attn_Softmax = Softmax(data_type)
        self.Attn_A_mul_V = BatchedMatmul(data_type)
        
        self.Reshape = Reshape(data_type)
        self.Transpose = Transpose(data_type)

    def update_workload(self, batch_requests):
        """
        核心逻辑：根据当前的混合 Batch 更新算子的工作负载状态。
        
        Args:
            batch_requests: list of Request objects (from scheduler.py)
        """
        total_tokens = 0
        self.request_meta = []

        # 1. 遍历 Batch，统计 Token 总数并分类请求
        for req in batch_requests:
            is_prefill = req.is_init
            
            # scheduler.py 中: 
            # prefill 阶段: req.input 是 prompt 长度
            # decode 阶段: req.input 是当前的 kv cache 长度 (包含历史)
            
            if is_prefill:
                num_tokens = req.input
                seq_len = req.input
                query_len = req.input
            else:
                num_tokens = 1
                seq_len = req.input # total history length
                query_len = 1
            
            total_tokens += num_tokens
            self.request_meta.append({
                'is_prefill': is_prefill,
                'seq_len': seq_len,   # Attention 的 K/V 长度
                'query_len': query_len # Attention 的 Q 长度
            })

        # 2. 模拟 Linear Layers (Shared Weights)
        # Orca 将所有 token 拼成一个 [Total_Tokens, Hidden] 的大矩阵
        # 我们通过执行一次虚拟的 __call__ 来更新算子内部记录的 Input Shape
        
        # 构造虚拟输入 [Total_Tokens, Hidden]
        # 只需要 Shape 正确即可触发 LLMCompass 的逻辑
        flattened_input_shape = [1, total_tokens, self.d_model] # Batch dim=1, Seq=Total
        dummy_input = Tensor(flattened_input_shape, self.data_type)

        # 这里的调用是为了让 operator 记住 input shape，以便后续调用 roofline_model
        # Q/K/V Projections
        self.Q_proj(dummy_input, self.Wq)
        self.K_proj(dummy_input, self.Wk)
        self.V_proj(dummy_input, self.Wv)
        
        # FFN Layers & Output Projection
        # 假设中间维度也是 flatten 的
        self.H_matmul0(Tensor([1, total_tokens, self.d_model // self.device_count], self.data_type), self.W0)
        self.H_matmul1(dummy_input, self.W1) # Up
        self.H_matmul2(Tensor([1, total_tokens, 4 * self.d_model // self.device_count], self.data_type), self.W2) # Down

        # Norms & GeLU
        self.layer_norm0(dummy_input)
        self.layer_norm1(dummy_input)
        self.H_gelu(Tensor([1, total_tokens, 4 * self.d_model // self.device_count], self.data_type))
        
        # AllReduce (对 flattened tokens 进行规约)
        # 如果存在多卡 (TP > 1)，我们需要告诉 AllReduce 算子这次要传输多少数据。
        # 在 Transformer 中，AllReduce 发生在 Output Projection 之后和 FFN Down Projection 之后。
        # 传输的数据形状通常是 [Batch, Sequence, Hidden_Size]。
        # 在我们的 Selective Batching 中，就是 [1, total_tokens, d_model]。
        if self.device_count > 1:
            # 构造一个全尺寸的虚拟 Tensor
            # AllReduce 是对所有卡的结果求和，所以输入输出 shape 都是 [total_tokens, d_model]
            # 模拟 Attention 输出后的 AllReduce 和 FFN 输出后的 AllReduce
            reduce_tensor = Tensor([1, total_tokens, self.d_model], self.data_type)
            self.allreduce_mha(reduce_tensor)
            self.allreduce_ffn(reduce_tensor)

    def roofline_model(self, system: System):
        """
        计算整个混合 Batch 的延迟
        """
        device = system.device
        interconnect = system.interconnect
        
        latency = 0.0

        # ==========================================
        # Part A: Flattened Linear Layers (Compute Bound)
        # ==========================================
        # 3x Proj + Out Proj + 2x FFN (Gate/Up + Down) = 6 Matmuls (LLaMA style usually has 7, but fit to existing code)
        # existing code: Wq, Wk, Wv, W0, W1, W2. 
        
        overhead = device.compute_module.overhead.matmul
        
        latency += 3 * (self.Q_proj.roofline_model(device) + overhead) # Q, K, V
        latency += (self.H_matmul0.roofline_model(device) + overhead) # Out Proj
        latency += (self.H_matmul1.roofline_model(device) + overhead) # FFN Up
        latency += (self.H_matmul2.roofline_model(device) + overhead) # FFN Down
        
        # Norms & Activations
        latency += 2 * (self.layer_norm0.roofline_model(device) + device.compute_module.overhead.layernorm)
        latency += (self.H_gelu.roofline_model(device) + device.compute_module.overhead.gelu)

        # ==========================================
        # Part B: Selective Attention (Memory/Compute Bound Mixed)
        # ==========================================
        # 必须针对每个请求分别计算 Attention 开销，因为 Shape 各不相同
        
        attn_latency = 0.0
        h_per_dev = self.n_heads // self.device_count
        
        for req in self.request_meta:
            q_len = req['query_len']
            k_len = req['seq_len'] # total context length
            
            # 1. Score Calculation: Q * K^T
            # Q shape: [b=1, heads, q_len, head_dim]
            # K shape: [b=1, heads, head_dim, k_len]
            # Result: [b=1, heads, q_len, k_len]
            
            # 我们通过构造虚拟 Tensor 并调用 BatchedMatmul 来获取该形状下的延迟
            q_tensor = Tensor([1, h_per_dev, q_len, self.d_h], self.data_type)
            k_tensor_T = Tensor([1, h_per_dev, self.d_h, k_len], self.data_type)
            
            # 更新 BatchedMatmul 内部状态
            self.Attn_Q_mul_K(q_tensor, k_tensor_T)
            attn_latency += self.Attn_Q_mul_K.roofline_model(device) + overhead
            
            # 2. Softmax
            # Input: [1, heads, q_len, k_len]
            attn_score = Tensor([1, h_per_dev, q_len, k_len], self.data_type)
            self.Attn_Softmax(attn_score)
            attn_latency += self.Attn_Softmax.roofline_model(device) + device.compute_module.overhead.softmax
            
            # 3. Aggregation: Score * V
            # Score: [1, heads, q_len, k_len]
            # V: [1, heads, k_len, head_dim]
            v_tensor = Tensor([1, h_per_dev, k_len, self.d_h], self.data_type)
            self.Attn_A_mul_V(attn_score, v_tensor)
            attn_latency += self.Attn_A_mul_V.roofline_model(device) + overhead

        latency += attn_latency

        # ==========================================
        # Part C: Communication (AllReduce)
        # ==========================================
        if self.device_count > 1:
            
            # AllReduce happens twice: after Attention output and after FFN output    
            comm_latency = self.allreduce_mha.simulate(interconnect) * 2 # 两次 (Attn + FFN)
            latency += comm_latency

        return latency
    
    def compile_and_simulate(self, system, compile_mode='heuristic'):
        """
        使用硬件模拟器 (Systolic Array Simulator) 计算延迟。
        """
        device = system.device
        interconnect = system.interconnect
        
        latency = 0.0
        # 获取 overhead 配置
        oh = device.compute_module.overhead 

        # 1. Linear Parts
        # Q, K, V
        latency += 3 * (self.Q_proj.compile_and_simulate(device, compile_mode) + oh.matmul)
        # Projections
        latency += (self.H_matmul0.compile_and_simulate(device, compile_mode) + oh.matmul)
        latency += (self.H_matmul1.compile_and_simulate(device, compile_mode) + oh.matmul)
        latency += (self.H_matmul2.compile_and_simulate(device, compile_mode) + oh.matmul)
        
        # Norms & Act
        latency += 2 * (self.layer_norm0.compile_and_simulate(device, compile_mode) + oh.layernorm)
        latency += (self.H_gelu.compile_and_simulate(device, compile_mode) + oh.gelu)

        # 2. Attention Parts
        # 必须遍历每个请求，更新 shape 后再模拟
        attn_latency = 0.0
        h_per_dev = self.n_heads // self.device_count
        
        for req in self.request_meta:
            q_len = req['query_len']
            k_len = req['seq_len']
            
            # 构造 Tensor
            q_tensor = Tensor([1, h_per_dev, q_len, self.d_h], self.data_type)
            k_tensor_T = Tensor([1, h_per_dev, self.d_h, k_len], self.data_type)
            attn_score = Tensor([1, h_per_dev, q_len, k_len], self.data_type)
            v_tensor = Tensor([1, h_per_dev, k_len, self.d_h], self.data_type)

            # Score Calculation (Q * K)
            self.Attn_Q_mul_K(q_tensor, k_tensor_T) # 更新 shape
            attn_latency += self.Attn_Q_mul_K.compile_and_simulate(device, compile_mode) + oh.matmul
            
            # Softmax
            self.Attn_Softmax(attn_score) # 更新 shape
            attn_latency += self.Attn_Softmax.compile_and_simulate(device, compile_mode) + oh.softmax
            
            # Value Aggregation (Score * V)
            self.Attn_A_mul_V(attn_score, v_tensor) # 更新 shape
            attn_latency += self.Attn_A_mul_V.compile_and_simulate(device, compile_mode) + oh.matmul

        latency += attn_latency

        # 3. Communication
        if self.device_count > 1 and interconnect is not None:
            # AllReduce 的 simulate 方法在基类中是一样的，不需要 compile_and_simulate
            comm_latency = self.allreduce_mha.simulate(interconnect)
            latency += comm_latency * 2

        return latency

    def run_on_gpu(self):
        """
        基于真实 GPU 运行或 Profiling 数据的延迟。
        """
        latency = 0.0
        
        # 1. Linear Parts
        latency += 3 * self.Q_proj.run_on_gpu()
        latency += self.H_matmul0.run_on_gpu()
        latency += self.H_matmul1.run_on_gpu()
        latency += self.H_matmul2.run_on_gpu()
        
        latency += 2 * self.layer_norm0.run_on_gpu()
        latency += self.H_gelu.run_on_gpu()

        # 2. Attention Parts
        attn_latency = 0.0
        h_per_dev = self.n_heads // self.device_count
        
        for req in self.request_meta:
            q_len = req['query_len']
            k_len = req['seq_len']
            
            q_tensor = Tensor([1, h_per_dev, q_len, self.d_h], self.data_type)
            k_tensor_T = Tensor([1, h_per_dev, self.d_h, k_len], self.data_type)
            attn_score = Tensor([1, h_per_dev, q_len, k_len], self.data_type)
            v_tensor = Tensor([1, h_per_dev, k_len, self.d_h], self.data_type)

            # 必须调用 __call__ 更新 shape，因为 run_on_gpu 内部可能依赖 input_shape 来查表
            self.Attn_Q_mul_K(q_tensor, k_tensor_T)
            attn_latency += self.Attn_Q_mul_K.run_on_gpu()
            
            self.Attn_Softmax(attn_score)
            attn_latency += self.Attn_Softmax.run_on_gpu()
            
            self.Attn_A_mul_V(attn_score, v_tensor)
            attn_latency += self.Attn_A_mul_V.run_on_gpu()

        latency += attn_latency

        # 3. Communication
        # 参照 transformer.py，run_on_gpu 通常不包含 AllReduce 延迟 (设为0)
        # 因为单卡 Profiling 无法测量多卡通信
        allreduce_total_latency = 0
        latency += allreduce_total_latency

        self.latency_on_gpu = latency
        return self.latency_on_gpu

    def __call__(self, batch_requests, system, mode='roofline', compile_mode='heuristic'):
        """
        统一入口。
        Args:
            batch_requests: 请求列表
            system: 硬件系统对象
            mode: 计算模式 ('roofline', 'simulate', 'gpu')
            compile_mode: 仅在 mode='simulate' 时有效，传递给底层模拟器
        """
        # 1. 更新算子负载 (Shape)
        self.update_workload(batch_requests)

        # 2. 根据模式分发
        if mode == 'roofline':
            # 默认模式：基于公式
            return self.roofline_model(system)
            
        elif mode == 'simulate':
            # 模拟模式：速度较慢，基于 Systolic Array 行为模拟
            return self.compile_and_simulate(system, compile_mode=compile_mode)
            
        elif mode == 'gpu':
            # 实机/查表模式：基于本地 GPU 或 Profiling 数据
            return self.run_on_gpu()
            
        else:
            raise ValueError(f"Unknown execution mode: {mode}. Supported: 'roofline', 'simulate', 'gpu'")