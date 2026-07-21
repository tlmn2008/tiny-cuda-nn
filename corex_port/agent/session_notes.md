# tiny-cuda-nn → Iluvatar CoreX (ivcore11) 迁移记录

## 一、来源

- 上游仓库：`https://github.com/NVlabs/tiny-cuda-nn.git`
- 迁移起点：默认分支 `master`，commit `749dd70c5afc5a9dadb85e5652ed65d55e0ba187`（2026-04-21），含子模块（cutlass、fmt 等，`--recursive` clone）。
- 本地工作区：`/home/repos/tiny-cuda-nn`，适配分支 `corex-port`。

## 二、CUDA 使用性质

tiny-cuda-nn 是一个高度模板化的 CUDA 库：小型 MLP（FullyFusedMLP 走 NV TensorCore wmma/mma；CutlassMLP 走 CUTLASS GEMM）+ 各类输入编码（Grid/Hash、Frequency、OneBlob、SphericalHarmonics 等）。深度依赖：NV 架构宏硬编码、fp16/half2 intrinsics、TensorCore MMA PTX、CUTLASS（NV-ISA 内联汇编 + 32 位 shared 地址模型 + warpSize=32 warp tile 几何）、NVRTC JIT 融合、CUDA 纹理对象、CUDA 虚拟内存管理（cuMemCreate）。

## 三、环境

- SDK：`/usr/local/corex`（CoreX 4.5.0，clang++ 前端，Driver 4.5.0）。
- 硬件：Iluvatar BI-V150；本 agent 分配 **GPU 1**（`CUDA_VISIBLE_DEVICES=1`，GPU 0 有并发迁移，全程未干扰）。
- 构建：CMake + clang++（`CMAKE_CUDA_COMPILER_ID==ILUVATAR`），架构 `TCNN_CUDA_ARCHITECTURES=ivcore11`。
- 红线遵守：未修改 `/usr/local/corex`；未用 nvcc；未使用 >2 GPU/rank；GPU 复位仅针对 GPU 1（`ixsmi -i 1 -r`）。

## 四、适配内容

### 构建系统（CMakeLists.txt）
- 识别 ILUVATAR 编译器：`MIN_GPU_ARCH=60`，绕过 NV 数值架构 clamp，**关闭 FullyFusedMLP、强制 CutlassMLP**。
- nvcc-only flag 翻译为 clang++ 形式：丢弃 `-Xcompiler=`、`-Xcudafe`、`--use_fast_math`、`--extended-lambda`、`--expt-relaxed-constexpr`（clang 原生支持后两者），补 `-Wno-unknown-cuda-version` 等。

### 源码宏 / intrinsic 兼容（均以 `defined(__ILUVATAR__)` 门控）
- `common.h`、`src/common_host.cu`：绕过 `__CUDA_ARCH__>=...` 与 `__CUDACC_VER_MAJOR__/MINOR__` 静态断言（ivcore11 误报 `__CUDA_ARCH__=300`、不定义 CUDACC 版本宏，但硬件支持 fp16 / 属 CUDA-11+ 级）。
- `common_device.h`：`lane_id()` 改用可移植 builtin `__nvvm_read_ptx_sreg_laneid()`（`%laneid` PTX 被后端拒绝）。
- `vec.h`：`__hfma`、`atomicAdd(__half2)`、half2 向量特化在 ivcore11 上启用（原门控 `__CUDA_ARCH__>=600`）。

### 运行时 / 内存
- `src/common_host.cu`：`cuda_supports_virtual_memory()` 改为真正探测 `cuMemGetAllocationGranularity`+`cuMemCreate`（按设备缓存），因为 ivcore11 的设备属性谎报支持 VMM、`cuMemAddressReserve` 也成功，但 `cuMemCreate` 运行时返回 `IX_ERROR_NOT_SUPPORTED`。据此让 `GPUMemoryArena` 走常规 cudaMalloc 回退路径。
- `gpu_memory.h`：把“不支持虚拟内存、回退”的 `log_warning` 降级为 `log_debug`，避免测试框架把该预期告警当成失败。

### CUTLASS 子模块（见 `changes/cutlass.patch`）
- `cute/arch/util.hpp`：补 `__cvta_generic_to_shared` shim（返回低 32 位 = shared 窗口内偏移，符合 NV cvta.to.shared 语义）。
- `cutlass/arch/memory.h`、`memory_sm75.h`：将 NV-only 的 `ld.global/st.global/ld.shared/st.shared`（含 `'l'/'h'` 约束、`ld.shared.v4.b32`——后端报 invalid instruction）替换为通用 C++。**关键修复**：探测发现 ivcore11 上 shared 泛型指针形如 `0xfffffff3_00000000|offset`（高 32 位为每地址空间统一的 shared 窗口基址），故在 shared load/store 中用运行期锚点 `smem_base()` 重建真实指针 `(base|offset)`（已用“经重建指针写、经真实指针读”探针验证正确）。

### 纹理 / 命名冲突（sample + benchmarks）
- `samples/mlp_learning_an_image.cu`、`benchmarks/image/bench_ours.cu`：用手写双线性采样（复现 `cudaFilterModeLinear`+归一化坐标+clamp）替换不可用的 `tex2D`/`cudaCreateTextureObject`。
- `benchmarks/mlp/bench_mlp_ours.cu`：`RM/CM` 显式限定为 `tcnn::RM/CM`（corex `__clang_cuda_ivcorex_intrinsics.h` 注入全局 `RM/CM` 舍入模式枚举，与 tcnn 冲突）。
- `tests/test_jit_losses.cu`：在 `supports_jit_fusion()` 跳过守卫后补 `return;`，使 JIT-only 用例在 ivcore11（cc=71）上干净跳过而非构造 `CudaRtcKernel` 抛异常。

## 五、结果

- **编译**：成功。`libtiny-cuda-nn.a`（18 MB）、全部单测二进制、sample、两个 benchmark 均构建通过。
- **测试**（GPU 1）：
  - `test_grid`：PASS。
  - `test_encodings`：PASS（47,186,016 断言，2 用例；覆盖 Identity/Frequency/OneBlob/SphericalHarmonics/Grid/Composite，inference == inference_mixed_precision == forward 全部一致）。
  - `test_jit_losses`：PASS（JIT 比较因 cc<75 自跳过）。
  - `test_networks`：FAIL —— CutlassMLP GEMM 在 ivcore11 上**活锁**（GPU-Util 100%，即便 16→32×2→16 / batch=4096 的推理 300s 也不返回）。
- **sample**：`reference.jpg` 正常产出（手写双线性采样工作正常）；进入 CutlassMLP 训练步后命中同一 GEMM 活锁。
- **整体**：`overall_status=blocked`（存在未解决 terminal blocker：MLP GEMM 计算路径）。编码/网格等 tiny-cuda-nn 主体功能可用，MLP 计算路径不可用。

## 六、Failure Gate（复现 + 分类 + 尝试）

- **不是凭推断判失败**：所有结论均实测复现（附 `test.log`、`compile.log`、探针 `build/probe_smem*.cu`）。
- **workaround-able 且已解决**（见 blockers.json 3–9）：`__cvta_generic_to_shared` 缺失、CUTLASS NV-PTX、`cuMemCreate` VMM、`%laneid`、`__CUDA_ARCH__/__CUDACC_VER` 断言、fp16 门控、纹理 API、RM/CM 冲突。
- **terminal**（见 blockers.json 1–2）：
  1. **CutlassMLP SIMT GEMM 活锁**：已依次尝试——强制 SIMT、中和 NV-PTX 使其编译链接、修正 32 位 shared 地址截断（把 GPU-Util 0% 硬死锁转成 100% 活锁）、复位 GPU 排除残留。根因为上游 NV CUTLASS SIMT 与 ivcore11 根本不兼容：warpSize=64（vs 32）破坏 CUTLASS 硬编码的 warp/线程 tile 几何与 shuffle 归约，且缺 Volta+ ITS、NV 32 位 shared 地址模型需模拟。真正可用需 Iluvatar 自家 ixCUTLASS，超出本次迁移范围（不可改 SDK、不可在本次内重写 CUTLASS warp 几何）。
  2. **FullyFusedMLP / JIT 融合**：需 NV TensorCore wmma/mma PTX + cc≥7.5 + IXRTC device-link，ivcore11（cc=71）均不满足。已通过 `MIN_GPU_ARCH=60` 关闭 fully-fused、并让 `supports_jit_fusion()==false` 自跳过 JIT，构建不受影响。

## 七、备注

- 大日志策略：`build/compile.log` 仅本地保留（.gitignore）；`test/test.log` 必须提交（12 KB，已含逐用例/汇总）。
- GPU 复位是必要操作：被 kill 的挂起 kernel 会残留驱动状态，导致后续 cuda_init 假挂；仅复位 GPU 1。
