
CC = nvcc
CFLAGS = 
fPIC_flag = -Xcompiler=-fPIC

obj_files = fused_elementwise_21.obj fused_elementwise_22.obj fused_elementwise_23.obj fused_elementwise_24.obj fused_elementwise_25.obj fused_elementwise_26.obj fused_elementwise_27.obj fused_elementwise_28.obj perm102_bmm_rrr_bias_53.obj perm102_bmm_rrr_bias_57.obj perm102_bmm_rrr_bias_61.obj perm102_bmm_rrr_bias_65.obj concatenate_39_constant_folding.obj permute021_41_constant_folding.obj concatenate_42_constant_folding.obj permute021_44_constant_folding.obj concatenate_45_constant_folding.obj permute021_47_constant_folding.obj concatenate_51_constant_folding.obj concatenate_55_constant_folding.obj constants.obj model_container_base.obj model_interface.obj debug_utility.obj utility.obj model_container.obj

%.obj : %.cu
	nvcc -t=0 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -w -gencode=arch=compute_80,code=[sm_80,compute_80] -O3 -std=c++17 --expt-relaxed-constexpr -DCUTLASS_DEBUG_TRACE_LEVEL=0 -DNDEBUG --use_fast_math -DAIT_USE_FAST_MATH=1 -Xcompiler=-fPIC -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -Xcompiler -fvisibility=hidden -I/home/wewe5215/miniconda3/envs/env_pytorch/lib/python3.12/site-packages/aitemplate/3rdparty/../static/include/kernels -I/home/wewe5215/miniconda3/envs/env_pytorch/lib/python3.12/site-packages/aitemplate/3rdparty/cutlass/include -I/home/wewe5215/miniconda3/envs/env_pytorch/lib/python3.12/site-packages/aitemplate/3rdparty/cutlass/tools/util/include -I/home/wewe5215/miniconda3/envs/env_pytorch/lib/python3.12/site-packages/aitemplate/3rdparty/cutlass/examples/35_gemm_softmax -I/home/wewe5215/miniconda3/envs/env_pytorch/lib/python3.12/site-packages/aitemplate/3rdparty/cutlass/examples/41_fused_multi_head_attention -I/home/wewe5215/miniconda3/envs/env_pytorch/lib/python3.12/site-packages/aitemplate/3rdparty/cutlass/examples/45_dual_gemm -I/home/wewe5215/miniconda3/envs/env_pytorch/lib/python3.12/site-packages/aitemplate/3rdparty/cutlass/../../backend/cuda/attention/src/./ -I/home/wewe5215/miniconda3/envs/env_pytorch/lib/python3.12/site-packages/aitemplate/3rdparty/cutlass/../../backend/cuda/attention/src/fmha -c -o $@ $<
%.obj : %.bin
	ld -r -b binary -o $@ $< && objcopy --rename-section .data=.lrodata,alloc,load,readonly,data,contents $@ $@

.PHONY: all clean clean_constants
all: test_0.so

test_0.so: $(obj_files)
	$(CC) -shared $(fPIC_flag) $(CFLAGS) -o $@ $(obj_files)



clean:
	rm -f *.obj test_0.so

clean_constants:
	rm -f constants.bin