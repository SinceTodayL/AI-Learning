## Project: PosterMaker Optimization



> [!TIP]
>
> 本文件为代码修改日志文件
>
> 修改类型：
>
> * ADD: 增加代码段
> * DELETE: 删除代码段
> * CHANGE: 修改代码段
>
> 修改模块：
>
> * 1: 注释/中间变量/打印报错信息等
> * 2: 有关函数逻辑/参数的重要修改





==格式==：

>20250000 TYPE MODULE

**File: ** `.py`

**Person: ** LZ

**Where: ** `class `

```python
 
```

**Why: ** 

**Note: ** 







>20250530 ADD 1

**File: ** `app.py`

**Person: ** LZ

**Where: ** `class ImageGenerator`

```python
 # 显示内存使用情况
        if torch.cuda.is_available():
            self.print_memory_usage()
        
    def print_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"CUDA内存使用状态:")
            print(f"  已分配: {allocated:.2f} GB")
            print(f"  已预留: {reserved:.2f} GB") 
            print(f"  总容量: {total:.2f} GB")
            print(f"  可用: {total - reserved:.2f} GB\n")
```

**Why: ** 观察 CUDA 显存占用

**Note: ** 不影响模型运行过程



>20250530 ADD 1

**File: ** `app.py`

**Person: ** LZ

**Where: ** `function generate_image(main_image, mask_image, texts_str, prompt, seed)`

```python
 # 打印掩码类型、提示词长度、图片分辨率等
```

**Why: **  这是一个工具函数，调用之前的图像生成函数，加上了中间生成过程信息，便于后续调试

**Note: ** 不影响模型运行过程



>20250530 CHANGE 2

**File: ** `app.py`

**Person: ** LZ

**Where: ** `class ImageGenerator function initialize_models(self)`

```python
 # 将所有模型迁移至目标设备并设置精度
        vae = vae.to(device=device, dtype=weight_dtype)
        transformer = transformer.to(device=device, dtype=weight_dtype)
        text_encoder_one = text_encoder_one.to(device=device, dtype=weight_dtype)
        text_encoder_two = text_encoder_two.to(device=device, dtype=weight_dtype)
        text_encoder_three = text_encoder_three.to(device=device, dtype=weight_dtype)
        controlnet_inpaint = controlnet_inpaint.to(device=device, dtype=weight_dtype)
        controlnet_text = controlnet_text.to(device=device, dtype=weight_dtype)
        adapter = adapter.to(device=device, dtype=weight_dtype)
```

**Why: ** 将模块从默认 float32 转为 float16（或 bfloat16），大幅节省显存;  同时显式的迁移到 GPU

**Note: ** 会影响模型精度，目前如果调整为 float32，会爆显存



>20250530 ADD 2

**File: ** `app.py`

**Person: ** LZ

**Where: ** `class ImageGenerator function initialize_models(self)`

```python
  # 启用CUDA内存优化技术
        if torch.cuda.is_available():
            print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            optimizations_enabled = []
            # 尝试启用注意力切片
            try:
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing(1)
                    optimizations_enabled.append("注意力切片")
            except Exception as e:
                print(f"无法启用注意力切片: {e}")
            # 尝试启用VAE切片
            try:
                if hasattr(self.pipeline, 'enable_vae_slicing'):
                    self.pipeline.enable_vae_slicing()
                    optimizations_enabled.append("VAE切片")
            except Exception as e:
                print(f"无法启用VAE切片: {e}")
            # 尝试启用序列CPU卸载以实现最大内存节省
            try:
                if hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
                    self.pipeline.enable_sequential_cpu_offload()
                    optimizations_enabled.append("序列CPU卸载")
                elif hasattr(self.pipeline, 'enable_model_cpu_offload'):
                    self.pipeline.enable_model_cpu_offload()
                    optimizations_enabled.append("模型CPU卸载")
            except Exception as e:
                print(f"无法启用CPU卸载: {e}")
            # 手动VAE优化（适用于自定义管道）
            try:
                if hasattr(self.pipeline, 'vae') and hasattr(self.pipeline.vae, 'enable_slicing'):
                    self.pipeline.vae.enable_slicing()
                    optimizations_enabled.append("手动VAE切片")
            except Exception as e:
                print(f"无法启用手动VAE切片: {e}")
            # 手动注意力优化
            try:
                if hasattr(self.pipeline, 'transformer'):
                    # 启用梯度检查点（如果可用）
                    if hasattr(self.pipeline.transformer, 'enable_gradient_checkpointing'):
                        self.pipeline.transformer.enable_gradient_checkpointing()
                        optimizations_enabled.append("梯度检查点")    
                    # 启用前向分块以提高内存效率
                    if hasattr(self.pipeline.transformer, 'enable_forward_chunking'):
                        self.pipeline.transformer.enable_forward_chunking(chunk_size=1)
                        optimizations_enabled.append("前向分块")
            except Exception as e:
                print(f"无法启用梯度检查点: {e}")
            # 为ControlNet启用优化
            try:
                for controlnet_name in ['controlnet_inpaint', 'controlnet_text']:
                    if hasattr(self.pipeline, controlnet_name):
                        controlnet = getattr(self.pipeline, controlnet_name)
                        
                        # 为ControlNet启用梯度检查点
                        if hasattr(controlnet, 'enable_gradient_checkpointing'):
                            controlnet.enable_gradient_checkpointing()
                            optimizations_enabled.append(f"{controlnet_name}梯度检查点")
                        
                        # 为ControlNet启用前向分块
                        if hasattr(controlnet, 'enable_forward_chunking'):
                            controlnet.enable_forward_chunking(chunk_size=1)
                            optimizations_enabled.append(f"{controlnet_name}前向分块")
            except Exception as e:
                print(f"无法启用ControlNet优化: {e}")
            # 额外的内存节省技术
            try:
                # 设置内存高效注意力（如果可用）
                if hasattr(self.pipeline.transformer, 'set_use_memory_efficient_attention_xformers'):
                    self.pipeline.transformer.set_use_memory_efficient_attention_xformers(True)
                    optimizations_enabled.append("xformers注意力")
                    
                # 为VAE启用瓦片化模式（如果可用）
                if hasattr(self.pipeline.vae, 'enable_tiling'):
                    self.pipeline.vae.enable_tiling()
                    optimizations_enabled.append("VAE瓦片化")
```

**Why: ** 解决原版模型运行中存在的 **显存过高、易 OOM** 问题，提升在中低端显卡、AutoDL 等环境下的稳定性。

**Note: ** 这些改动不会改变模型本身的运行原理、结构或输出逻辑，它们改变的是“如何运行”而不是“模型做了什么”

==这段代码是解决 CUDA is out of memory 的功臣==



>20250530 ADD 1

**File: ** `app.py`

**Person: ** LZ

**Where: ** `function generate(self, main_image, mask_image, texts_str, prompt, seed_generator)`

```python
 # 多次
    if torch.cuda.is_available():
	torch.cuda.empty_cache()  # 缓存的显存，不会清理仍被张量占用的显存
	torch.cuda.ipc_collect()  # 主动清理已经不再使用的共享内存块
   
# 执行推理 - 减少推理步数以节省显存
num_steps = 5 if torch.cuda.is_available() else 1  # 进一步减少推理步数
            
# 使用与数据预处理一致的分辨率，避免张量尺寸不匹配
# data_processor默认使用(1024, 1024)，推理也应该使用相同分辨率
height, width = 1024, 1024  # 与data_processor的input_size保持一致
            
print(f"使用一致的处理分辨率: {width}x{height}")
print(f"开始管道推理，步数: {num_steps}, 分辨率: {width}x{height}")
```

**Why: **  释放 PyTorch 中 CUDA 显存资源（主要是避免内存泄漏），以及减少推理步数以节省显存 `num_steps`

**Note: ** 不影响模型运行过程，但是显存容易碎片化



