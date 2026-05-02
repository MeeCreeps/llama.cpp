// Fused LoRA delta kernel: out = base + scale * (B @ (A @ x))
//
// Replaces the GGML 4-node chain inserted by build_lora_mm in
// src/llama-graph.cpp:
//   tmp_a        = A @ x         (mul_mat A)
//   tmp_b        = B @ tmp_a     (mul_mat B)
//   tmp_b_scaled = scale * tmp_b (scale)
//   out          = base + tmp_b_scaled  (add)
//
// All four collapse into one launch per LoRA-target projection.
//
// GGML tensor layout convention: ne[0] is the contiguous dim, ne[1] is the
// outer dim, etc. row-major over ne[0].
//
//   A : (ne[0]=H_in, ne[1]=R)         -- LoRA A weight, f16, ~6 KB at H_in=3072
//   x : (ne[0]=H_in, ne[1]=seq)       -- layer activation, f32
//   B : (ne[0]=R, ne[1]=H_out)        -- LoRA B weight, f16
//   base : (ne[0]=H_out, ne[1]=seq)   -- result of base mul_mat (x @ W), f32
//   out : same shape as base, f32
//
// Workgroup organisation: ONE workgroup per token (s).
//   - Phase 1: WG_SIZE lanes cooperatively compute tmp_a[r] for r=0..R-1
//   - Phase 2: WG_SIZE lanes loop over h_out chunks of WG_SIZE
//
// No redundant work across workgroups — total MAC count matches the
// per-op chain (R * H_in for tmp_a + R * H_out for tmp_b per token).
// What's saved is the (3 * 196) per-forward op-launches and the DRAM
// round-trip for tmp_a / tmp_b / tmp_b_scaled intermediates.
//
// R is bounded by LORA_R_MAX (compile-time); v0 supports up to 64.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LORA_R_MAX 64
#define WG_SIZE    64

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void kernel_lora_delta_f16_f32(
        __global const half  * A,    const ulong offset_A,
        __global const float * x,    const ulong offset_x,
        __global const half  * B,    const ulong offset_B,
        __global const float * base, const ulong offset_base,
        __global       float * out,  const ulong offset_out,
        const float scale,
        const int H_in,
        const int R,
        const int H_out)
{
    A    = (__global const half  *)((__global const char *)A    + offset_A);
    x    = (__global const float *)((__global const char *)x    + offset_x);
    B    = (__global const half  *)((__global const char *)B    + offset_B);
    base = (__global const float *)((__global const char *)base + offset_base);
    out  = (__global       float *)((__global       char *)out  + offset_out);

    const int s   = get_group_id(1);
    const int lid = get_local_id(0);

    __local float tmp_a[LORA_R_MAX];
    __local float partial[WG_SIZE];

    // ---- Phase 1: cooperative computation of tmp_a[r] = sum_h A[h, r] * x[h, s] ----
    // for each r, each lane sums a stride of H_in and we reduce across the workgroup.
    for (int r = 0; r < R; ++r) {
        // A[h, r] is at element offset (r * H_in + h)
        // x[h, s] is at element offset (s * H_in + h)
        const __global const half  * a_row = A + (size_t)r * (size_t)H_in;
        const __global const float * x_row = x + (size_t)s * (size_t)H_in;

        float acc = 0.0f;
        for (int h = lid; h < H_in; h += WG_SIZE) {
            acc += vload_half(0, a_row + h) * x_row[h];
        }
        partial[lid] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduce 'partial' to a single value at index 0 within the workgroup
        for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                partial[lid] += partial[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            tmp_a[r] = partial[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ---- Phase 2: out[h_out, s] = base[h_out, s] + scale * sum_r B[r, h_out] * tmp_a[r] ----
    // Each lane handles its own h_out indices; the workgroup loops over chunks
    // of WG_SIZE.
    __global const float * base_row = base + (size_t)s * (size_t)H_out;
    __global       float * out_row  = out  + (size_t)s * (size_t)H_out;
    for (int h_out_base = 0; h_out_base < H_out; h_out_base += WG_SIZE) {
        const int h_out = h_out_base + lid;
        if (h_out >= H_out) {
            break;
        }
        // B[r, h_out] is at element offset (h_out * R + r)
        const __global const half * b_col = B + (size_t)h_out * (size_t)R;

        float delta = 0.0f;
        for (int r = 0; r < R; ++r) {
            delta += vload_half(0, b_col + r) * tmp_a[r];
        }

        out_row[h_out] = base_row[h_out] + scale * delta;
    }
}
