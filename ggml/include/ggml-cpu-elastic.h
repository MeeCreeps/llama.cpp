// ggml-cpu-elastic.h
//
// 隔离的 CPU elastic 后端：复用 CPU compute 路径，套上 WBM 弹性内存管理。
// set_tensor 时把 mmap 源指针记到 WBM，graph_compute 前 ensure_resident，
// 周期 evict 用 madvise(MADV_DONTNEED) 回收 anonymous 页，再用 memcpy 从
// mmap 源 reload。
//
// 触发：调 ggml_backend_cpu_elastic_init() 显式启用。也通过 ggml backend
// registry 注册为 "CPU_Elastic" 设备。

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t            ggml_backend_cpu_elastic_init(void);
GGML_BACKEND_API bool                      ggml_backend_is_cpu_elastic(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cpu_elastic_buffer_type(void);
GGML_BACKEND_API ggml_backend_reg_t        ggml_backend_cpu_elastic_reg(void);

#ifdef  __cplusplus
}
#endif
