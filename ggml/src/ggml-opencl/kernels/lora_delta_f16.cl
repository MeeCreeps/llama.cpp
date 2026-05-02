// Fused LoRA delta kernels: out = base + scale * (B @ (A @ x))
//
// Iter 3 splits the fusion into two launches per LoRA-target projection:
//   kernel_lora_a_f16_f32          : tmp_a[r, s] = sum_h A[h, r] * x[h, s]
//   kernel_lora_b_scale_add_f16_f32: out[h_out, s] =
//                                       base[h_out, s] + scale * sum_r B[r, h_out] * tmp_a[r, s]
//
// vs the un-fused 4-op chain (mul_mat A -> mul_mat B -> scale -> add) we
// save 2 op-launches per projection (the scale and add). vs iter 2's
// single-kernel "redundant tmp_a per workgroup" we avoid recomputing
// tmp_a 48× per token.
//
// The tmp_a scratch is the GGML output buffer of the original mul_mat A
// node (which is bypassed but its allocation is already in place — see
// ggml_opencl_op_lora_delta_fused). Shape (R, seq), f32.
//
// Tensor layout (GGML row-major over ne[0]):
//   A     : (ne[0]=H_in,  ne[1]=R)        f16
//   x     : (ne[0]=H_in,  ne[1]=seq)      f32
//   B     : (ne[0]=R,     ne[1]=H_out)    f16
//   tmp_a : (ne[0]=R,     ne[1]=seq)      f32  — produced by kernel A,
//                                                consumed by kernel B
//   base  : (ne[0]=H_out, ne[1]=seq)      f32
//   out   : same shape as base            f32

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

#define WG_SIZE 64

// ----- Kernel A: tmp_a[r, s] = sum_h A[h, r] * x[h, s] -----
// global = (R * WG_SIZE, seq, 1); local = (WG_SIZE, 1, 1)
// One WG per (r, s); subgroup reduction of WG_SIZE=64 lanes over H_in.
__kernel
REQD_SUBGROUP_SIZE_64
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void kernel_lora_a_f16_f32(
        __global const half  * A,     const ulong offset_A,
        __global const float * x,     const ulong offset_x,
        __global       float * tmp_a, const ulong offset_tmp_a,
        const int H_in,
        const int R)
{
    A     = (__global const half  *)((__global const char *)A     + offset_A);
    x     = (__global const float *)((__global const char *)x     + offset_x);
    tmp_a = (__global       float *)((__global       char *)tmp_a + offset_tmp_a);

    const int s   = get_group_id(1);
    const int r   = get_group_id(0);
    const int lid = get_local_id(0);

    const __global const half4  * a_v = (const __global half4  *)(A + (size_t)r * (size_t)H_in);
    const __global const float4 * x_v = (const __global float4 *)(x + (size_t)s * (size_t)H_in);
    const int H_in_4 = H_in >> 2;

    float4 acc = (float4)(0.0f);
    for (int h4 = lid; h4 < H_in_4; h4 += WG_SIZE) {
        acc = mad(convert_float4(a_v[h4]), x_v[h4], acc);
    }
    float lane_sum = acc.s0 + acc.s1 + acc.s2 + acc.s3;
    float r_sum    = sub_group_reduce_add(lane_sum);

    if (lid == 0) {
        tmp_a[(size_t)s * (size_t)R + (size_t)r] = r_sum;
    }
}

// ----- Kernel B: out[h_out, s] = base[h_out, s] + scale * sum_r B[r, h_out] * tmp_a[r, s] -----
// global = (h_out_groups * WG_SIZE, seq, 1); local = (WG_SIZE, 1, 1)
// One WG per (h_out_tile, s); each lane computes one h_out output.
//
// tmp_a[r, s] for r=0..R-1 is contiguous in memory (R is ne[0]). We pull
// it into a small private register array per lane (R <= 64) once at the
// start, then dot with B[*, h_out].
__kernel
REQD_SUBGROUP_SIZE_64
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void kernel_lora_b_scale_add_f16_f32(
        __global const half  * B,     const ulong offset_B,
        __global const float * tmp_a, const ulong offset_tmp_a,
        __global const float * base,  const ulong offset_base,
        __global       float * out,   const ulong offset_out,
        const float scale,
        const int R,
        const int H_out)
{
    B     = (__global const half  *)((__global const char *)B     + offset_B);
    tmp_a = (__global const float *)((__global const char *)tmp_a + offset_tmp_a);
    base  = (__global const float *)((__global const char *)base  + offset_base);
    out   = (__global       float *)((__global       char *)out   + offset_out);

    const int s            = get_group_id(1);
    const int h_out_block  = get_group_id(0);
    const int lid          = get_local_id(0);
    const int h_out        = h_out_block * WG_SIZE + lid;
    if (h_out >= H_out) {
        return;
    }

    // Pull tmp_a row for this token into private registers (R <= 64).
    const __global const float * tmp_a_row = tmp_a + (size_t)s * (size_t)R;

    // B[r, h_out] for r=0..R-1 lies contiguously at element offset
    // h_out * R + r (since ne[0]=R is the contiguous dim).
    const __global const half * b_col = B + (size_t)h_out * (size_t)R;

    float delta = 0.0f;
    for (int r = 0; r < R; ++r) {
        delta = mad((float)vload_half(0, b_col + r), tmp_a_row[r], delta);
    }

    const size_t off = (size_t)s * (size_t)H_out + (size_t)h_out;
    out[off] = base[off] + scale * delta;
}
