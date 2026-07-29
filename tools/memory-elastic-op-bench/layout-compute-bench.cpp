#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include "ggml.h"

#include <CL/cl.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define CL_CHECK(expr) do { \
    const cl_int err_ = (expr); \
    if (err_ != CL_SUCCESS) { \
        throw std::runtime_error(std::string("OpenCL error ") + std::to_string(err_) + \
                                 " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
} while (0)

static const char * kernel_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline float fp16_bits_to_float(ushort bits) {
    return convert_float(as_half(bits));
}

// Physical index for a Q4 block. layout=0 keeps rows contiguous; layout=1
// stores equal-K blocks from all output rows contiguously.
inline int physical_block(int row, int kb, int m, int nb, int layout) {
    return layout == 0 ? row*nb + kb : kb*m + row;
}

inline int tile16_block(int row, int kb, int m, int nb) {
    const int ntk = (nb + 15)/16;
    return ((row/16)*ntk + kb/16)*256 + (kb%16)*16 + row%16;
}

__kernel void aos_to_split(
        __global const uchar * aos,
        __global ushort * scales,
        __global uchar * qplanes,
        int m, int nb, int layout) {
    const int gid = get_global_id(0);
    const int nblocks = m*nb;
    if (gid >= nblocks) return;
    const int row = gid/nb;
    const int kb = gid - row*nb;
    const int dst = physical_block(row, kb, m, nb, layout);
    __global const uchar * src = aos + gid*18;
    scales[dst] = (ushort) src[0] | ((ushort) src[1] << 8);
    for (int p = 0; p < 16; ++p) {
        qplanes[p*nblocks + dst] = src[2+p];
    }
}

// A 16x16 local-memory transpose makes both the row-major AOS reads and the
// block-major split writes contiguous. It produces exactly the same layout as
// aos_to_split(layout=1), but uses a layout-aware materialization algorithm.
__kernel void aos_to_block_tiled(
        __global const uchar * aos,
        __global ushort * scales,
        __global uchar * qplanes,
        int m, int nb) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int kb = get_group_id(0)*16 + lx;
    const int row = get_group_id(1)*16 + ly;
    __local uchar tile[16*16*18];
    __local uchar * mine = tile + (ly*16 + lx)*18;
    if (row < m && kb < nb) {
        __global const uchar * src = aos + (row*nb + kb)*18;
        for (int p = 0; p < 18; ++p) mine[p] = src[p];
    } else {
        for (int p = 0; p < 18; ++p) mine[p] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int out_row = get_group_id(1)*16 + lx;
    const int out_kb = get_group_id(0)*16 + ly;
    if (out_row < m && out_kb < nb) {
        __local uchar * src = tile + (lx*16 + ly)*18;
        const int dst = out_kb*m + out_row;
        const int nblocks = m*nb;
        scales[dst] = (ushort) src[0] | ((ushort) src[1] << 8);
        for (int p = 0; p < 16; ++p) qplanes[p*nblocks + dst] = src[2+p];
    }
}

// Materialize only a row range of the block-major layout. This models an
// incremental promotion whose work can be spread across token boundaries.
// row_start and row_count are multiples of the 16-row tile except at the tail.
__kernel void aos_to_block_tiled_chunk(
        __global const uchar * aos,
        __global ushort * scales,
        __global uchar * qplanes,
        int m, int nb, int row_start, int row_count) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int kb = get_group_id(0)*16 + lx;
    const int row = row_start + get_group_id(1)*16 + ly;
    const int row_end = min(m, row_start + row_count);
    __local uchar tile[16*16*18];
    __local uchar * mine = tile + (ly*16 + lx)*18;
    if (row < row_end && kb < nb) {
        __global const uchar * src = aos + (row*nb + kb)*18;
        for (int p = 0; p < 18; ++p) mine[p] = src[p];
    } else {
        for (int p = 0; p < 18; ++p) mine[p] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int out_row = row_start + get_group_id(1)*16 + lx;
    const int out_kb = get_group_id(0)*16 + ly;
    if (out_row < row_end && out_kb < nb) {
        __local uchar * src = tile + (lx*16 + ly)*18;
        const int dst = out_kb*m + out_row;
        const int nblocks = m*nb;
        scales[dst] = (ushort) src[0] | ((ushort) src[1] << 8);
        for (int p = 0; p < 16; ++p) qplanes[p*nblocks + dst] = src[2+p];
    }
}

// Conventional K-major (column-major in Q4-block space) AOS. Quantized
// values and their scale remain interleaved in the original 18-byte block.
__kernel void aos_to_kmajor_aos(
        __global const uchar * aos,
        __global uchar * output,
        int m, int nb) {
    const int kb = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= m || kb >= nb) return;
    __global const uchar * src = aos + (row*nb + kb)*18;
    __global uchar * dst = output + (kb*m + row)*18;
    for (int p = 0; p < 18; ++p) dst[p] = src[p];
}

// Conventional 2-D blocked AOS. Each 16-row by 16-K-block tile is contiguous,
// and blocks are K-major inside the tile.
__kernel void aos_to_tile16_aos(
        __global const uchar * aos,
        __global uchar * output,
        int m, int nb) {
    const int kb = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= m || kb >= nb) return;
    __global const uchar * src = aos + (row*nb + kb)*18;
    __global uchar * dst = output + tile16_block(row, kb, m, nb)*18;
    for (int p = 0; p < 18; ++p) dst[p] = src[p];
}

// Block-major Q4 with FP32 scales: 20 bytes/block instead of 18. This spends
// 11.1% more persistent memory to remove FP16 scale conversion from GEMV.
__kernel void aos_to_fp32_q4_tiled(
        __global const uchar * aos,
        __global float * scales,
        __global uchar * qplanes,
        int m, int nb) {
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int kb = get_group_id(0)*16 + lx;
    const int row = get_group_id(1)*16 + ly;
    __local uchar tile[16*16*18];
    __local uchar * mine = tile + (ly*16 + lx)*18;
    if (row < m && kb < nb) {
        __global const uchar * src = aos + (row*nb + kb)*18;
        for (int p = 0; p < 18; ++p) mine[p] = src[p];
    } else {
        for (int p = 0; p < 18; ++p) mine[p] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int out_row = get_group_id(1)*16 + lx;
    const int out_kb = get_group_id(0)*16 + ly;
    if (out_row < m && out_kb < nb) {
        __local uchar * src = tile + (lx*16 + ly)*18;
        const int dst = out_kb*m + out_row;
        const int nblocks = m*nb;
        const ushort dh = (ushort) src[0] | ((ushort) src[1] << 8);
        scales[dst] = fp16_bits_to_float(dh);
        for (int p = 0; p < 16; ++p) qplanes[p*nblocks + dst] = src[2+p];
    }
}

// Expanded block-major representation: FP32 scale plus 32 signed Q4 values,
// 36 bytes/block (2x raw). It removes nibble unpacking at the cost of bandwidth.
__kernel void aos_to_expanded_tiled(
        __global const uchar * aos,
        __global float * scales,
        __global char * qvalues,
        int m, int nb) {
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int kb = get_group_id(0)*16 + lx;
    const int row = get_group_id(1)*16 + ly;
    __local uchar tile[16*16*18];
    __local uchar * mine = tile + (ly*16 + lx)*18;
    if (row < m && kb < nb) {
        __global const uchar * src = aos + (row*nb + kb)*18;
        for (int p = 0; p < 18; ++p) mine[p] = src[p];
    } else {
        for (int p = 0; p < 18; ++p) mine[p] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int out_row = get_group_id(1)*16 + lx;
    const int out_kb = get_group_id(0)*16 + ly;
    if (out_row < m && out_kb < nb) {
        __local uchar * src = tile + (lx*16 + ly)*18;
        const int dst = out_kb*m + out_row;
        const int nblocks = m*nb;
        const ushort dh = (ushort) src[0] | ((ushort) src[1] << 8);
        scales[dst] = fp16_bits_to_float(dh);
        for (int p = 0; p < 16; ++p) {
            const uchar q = src[2+p];
            qvalues[p*nblocks + dst] = (char)((int)(q & 15) - 8);
            qvalues[(p+16)*nblocks + dst] = (char)((int)(q >> 4) - 8);
        }
    }
}

__kernel void gemv_aos(
        __global const uchar * aos,
        __global const float * x,
        __global float * y,
        int m, int nb) {
    const int row = get_global_id(0);
    if (row >= m) return;
    float sum = 0.0f;
    for (int kb = 0; kb < nb; ++kb) {
        __global const uchar * src = aos + (row*nb + kb)*18;
        const ushort dh = (ushort) src[0] | ((ushort) src[1] << 8);
        const float d = fp16_bits_to_float(dh);
        const int x0 = kb*32;
        for (int p = 0; p < 16; ++p) {
            const uchar q = src[2+p];
            sum = fma(d*((float)(q & 15) - 8.0f), x[x0+p], sum);
            sum = fma(d*((float)(q >> 4) - 8.0f), x[x0+p+16], sum);
        }
    }
    y[row] = sum;
}

__kernel void gemv_split(
        __global const ushort * scales,
        __global const uchar * qplanes,
        __global const float * x,
        __global float * y,
        int m, int nb, int layout) {
    const int row = get_global_id(0);
    if (row >= m) return;
    const int nblocks = m*nb;
    float sum = 0.0f;
    for (int kb = 0; kb < nb; ++kb) {
        const int src = physical_block(row, kb, m, nb, layout);
        const float d = fp16_bits_to_float(scales[src]);
        const int x0 = kb*32;
        for (int p = 0; p < 16; ++p) {
            const uchar q = qplanes[p*nblocks + src];
            sum = fma(d*((float)(q & 15) - 8.0f), x[x0+p], sum);
            sum = fma(d*((float)(q >> 4) - 8.0f), x[x0+p+16], sum);
        }
    }
    y[row] = sum;
}

// Promotion and the first GEMV share the only read of the AOS source. The
// promoted split layout remains resident for subsequent gemv_split calls.
__kernel void promote_gemv(
        __global const uchar * aos,
        __global ushort * scales,
        __global uchar * qplanes,
        __global const float * x,
        __global float * y,
        int m, int nb, int layout) {
    const int row = get_global_id(0);
    if (row >= m) return;
    const int nblocks = m*nb;
    float sum = 0.0f;
    for (int kb = 0; kb < nb; ++kb) {
        __global const uchar * src = aos + (row*nb + kb)*18;
        const ushort dh = (ushort) src[0] | ((ushort) src[1] << 8);
        const int dst = physical_block(row, kb, m, nb, layout);
        scales[dst] = dh;
        const float d = fp16_bits_to_float(dh);
        const int x0 = kb*32;
        for (int p = 0; p < 16; ++p) {
            const uchar q = src[2+p];
            qplanes[p*nblocks + dst] = q;
            sum = fma(d*((float)(q & 15) - 8.0f), x[x0+p], sum);
            sum = fma(d*((float)(q >> 4) - 8.0f), x[x0+p+16], sum);
        }
    }
    y[row] = sum;
}

// Tiled fused promotion: each workgroup handles 16 rows. For every 16x16
// row/K-block tile, threads first coalesce AOS reads, transpose in local memory,
// then coalesce block-major writes. Sixteen threads accumulate each output row.
__kernel void promote_block_gemv_tiled(
        __global const uchar * aos,
        __global ushort * scales,
        __global uchar * qplanes,
        __global const float * xvec,
        __global float * output,
        int m, int nb) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int row_base = get_group_id(1)*16;
    const int nblocks = m*nb;
    __local uchar tile[16*16*18];
    __local float partial[16*16];
    float sum = 0.0f;

    for (int kt = 0; kt < nb; kt += 16) {
        const int read_row = row_base + ly;
        const int read_kb = kt + lx;
        __local uchar * mine = tile + (ly*16 + lx)*18;
        if (read_row < m && read_kb < nb) {
            __global const uchar * src = aos + (read_row*nb + read_kb)*18;
            for (int p = 0; p < 18; ++p) mine[p] = src[p];
        } else {
            for (int p = 0; p < 18; ++p) mine[p] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        const int row = row_base + lx;
        const int kb = kt + ly;
        if (row < m && kb < nb) {
            __local uchar * src = tile + (lx*16 + ly)*18;
            const ushort dh = (ushort) src[0] | ((ushort) src[1] << 8);
            const int dst = kb*m + row;
            scales[dst] = dh;
            const float d = fp16_bits_to_float(dh);
            const int x0 = kb*32;
            for (int p = 0; p < 16; ++p) {
                const uchar q = src[2+p];
                qplanes[p*nblocks + dst] = q;
                sum = fma(d*((float)(q & 15) - 8.0f), xvec[x0+p], sum);
                sum = fma(d*((float)(q >> 4) - 8.0f), xvec[x0+p+16], sum);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    partial[ly*16 + lx] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ly == 0 && row_base + lx < m) {
        float total = 0.0f;
        for (int lane = 0; lane < 16; ++lane) total += partial[lane*16 + lx];
        output[row_base + lx] = total;
    }
}

// Same 16x16 compute mapping as the tiled fused kernel, but computes directly
// from AOS and leaves no persistent promoted layout.
__kernel void gemv_aos_tiled(
        __global const uchar * aos,
        __global const float * xvec,
        __global float * output,
        int m, int nb) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int row_base = get_group_id(1)*16;
    __local uchar tile[16*16*18];
    __local float partial[16*16];
    float sum = 0.0f;
    for (int kt = 0; kt < nb; kt += 16) {
        const int read_row = row_base + ly;
        const int read_kb = kt + lx;
        __local uchar * mine = tile + (ly*16 + lx)*18;
        if (read_row < m && read_kb < nb) {
            __global const uchar * src = aos + (read_row*nb + read_kb)*18;
            for (int p = 0; p < 18; ++p) mine[p] = src[p];
        } else {
            for (int p = 0; p < 18; ++p) mine[p] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int row = row_base + lx;
        const int kb = kt + ly;
        if (row < m && kb < nb) {
            __local uchar * src = tile + (lx*16 + ly)*18;
            const ushort dh = (ushort) src[0] | ((ushort) src[1] << 8);
            const float d = fp16_bits_to_float(dh);
            const int x0 = kb*32;
            for (int p = 0; p < 16; ++p) {
                const uchar q = src[2+p];
                sum = fma(d*((float)(q & 15) - 8.0f), xvec[x0+p], sum);
                sum = fma(d*((float)(q >> 4) - 8.0f), xvec[x0+p+16], sum);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    partial[ly*16 + lx] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ly == 0 && row_base + lx < m) {
        float total = 0.0f;
        for (int lane = 0; lane < 16; ++lane) total += partial[lane*16 + lx];
        output[row_base + lx] = total;
    }
}

// Same compute mapping and reduction as tiled direct/fused, consuming the
// already-persistent block-major split layout.
__kernel void gemv_block_tiled(
        __global const ushort * scales,
        __global const uchar * qplanes,
        __global const float * xvec,
        __global float * output,
        int m, int nb) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int row = get_group_id(1)*16 + lx;
    const int nblocks = m*nb;
    __local float partial[16*16];
    float sum = 0.0f;
    if (row < m) {
        for (int kb = ly; kb < nb; kb += 16) {
            const int src = kb*m + row;
            const float d = fp16_bits_to_float(scales[src]);
            const int x0 = kb*32;
            for (int p = 0; p < 16; ++p) {
                const uchar q = qplanes[p*nblocks + src];
                sum = fma(d*((float)(q & 15) - 8.0f), xvec[x0+p], sum);
                sum = fma(d*((float)(q >> 4) - 8.0f), xvec[x0+p+16], sum);
            }
        }
    }
    partial[ly*16 + lx] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ly == 0 && row < m) {
        float total = 0.0f;
        for (int lane = 0; lane < 16; ++lane) total += partial[lane*16 + lx];
        output[row] = total;
    }
}

// Row-major SOA: scales and quantized byte planes are separated, while Q4
// blocks remain row-major. The workgroup/reduction matches gemv_block_tiled.
__kernel void gemv_row_soa_tiled(
        __global const ushort * scales,
        __global const uchar * qplanes,
        __global const float * xvec,
        __global float * output,
        int m, int nb) {
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int row = get_group_id(1)*16 + lx;
    const int nblocks = m*nb;
    __local float partial[16*16];
    float sum = 0.0f;
    if (row < m) {
        for (int kb = ly; kb < nb; kb += 16) {
            const int src = row*nb + kb;
            const float d = fp16_bits_to_float(scales[src]);
            const int x0 = kb*32;
            for (int p = 0; p < 16; ++p) {
                const uchar q = qplanes[p*nblocks + src];
                sum = fma(d*((float)(q & 15) - 8.0f), xvec[x0+p], sum);
                sum = fma(d*((float)(q >> 4) - 8.0f), xvec[x0+p+16], sum);
            }
        }
    }
    partial[ly*16 + lx] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ly == 0 && row < m) {
        float total = 0.0f;
        for (int lane = 0; lane < 16; ++lane) total += partial[lane*16 + lx];
        output[row] = total;
    }
}

__kernel void gemv_kmajor_aos_tiled(
        __global const uchar * aos,
        __global const float * xvec,
        __global float * output,
        int m, int nb) {
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int row = get_group_id(1)*16 + lx;
    __local float partial[16*16];
    float sum = 0.0f;
    if (row < m) {
        for (int kb = ly; kb < nb; kb += 16) {
            __global const uchar * src = aos + (kb*m + row)*18;
            const ushort dh = (ushort) src[0] | ((ushort) src[1] << 8);
            const float d = fp16_bits_to_float(dh);
            const int x0 = kb*32;
            for (int p = 0; p < 16; ++p) {
                const uchar q = src[2+p];
                sum = fma(d*((float)(q & 15) - 8.0f), xvec[x0+p], sum);
                sum = fma(d*((float)(q >> 4) - 8.0f), xvec[x0+p+16], sum);
            }
        }
    }
    partial[ly*16 + lx] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ly == 0 && row < m) {
        float total = 0.0f;
        for (int lane = 0; lane < 16; ++lane) total += partial[lane*16 + lx];
        output[row] = total;
    }
}

__kernel void gemv_tile16_aos_tiled(
        __global const uchar * aos,
        __global const float * xvec,
        __global float * output,
        int m, int nb) {
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int row = get_group_id(1)*16 + lx;
    __local float partial[16*16];
    float sum = 0.0f;
    if (row < m) {
        for (int kb = ly; kb < nb; kb += 16) {
            __global const uchar * src = aos + tile16_block(row, kb, m, nb)*18;
            const ushort dh = (ushort) src[0] | ((ushort) src[1] << 8);
            const float d = fp16_bits_to_float(dh);
            const int x0 = kb*32;
            for (int p = 0; p < 16; ++p) {
                const uchar q = src[2+p];
                sum = fma(d*((float)(q & 15) - 8.0f), xvec[x0+p], sum);
                sum = fma(d*((float)(q >> 4) - 8.0f), xvec[x0+p+16], sum);
            }
        }
    }
    partial[ly*16 + lx] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ly == 0 && row < m) {
        float total = 0.0f;
        for (int lane = 0; lane < 16; ++lane) total += partial[lane*16 + lx];
        output[row] = total;
    }
}

// Consume a partially promoted tensor. Promotion advances in complete 16-row
// tiles, so every workgroup uniformly selects either the persistent block-major
// representation or the original AOS representation.
__kernel void gemv_mixed_tiled(
        __global const uchar * aos,
        __global const ushort * scales,
        __global const uchar * qplanes,
        __global const float * xvec,
        __global float * output,
        int m, int nb, int promoted_rows) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int row_base = get_group_id(1)*16;
    const int row = row_base + lx;
    const int nblocks = m*nb;
    __local uchar tile[16*16*18];
    __local float partial[16*16];
    float sum = 0.0f;

    if (row_base < promoted_rows) {
        if (row < m) {
            for (int kb = ly; kb < nb; kb += 16) {
                const int src = kb*m + row;
                const float d = fp16_bits_to_float(scales[src]);
                const int x0 = kb*32;
                for (int p = 0; p < 16; ++p) {
                    const uchar q = qplanes[p*nblocks + src];
                    sum = fma(d*((float)(q & 15) - 8.0f), xvec[x0+p], sum);
                    sum = fma(d*((float)(q >> 4) - 8.0f), xvec[x0+p+16], sum);
                }
            }
        }
    } else {
        for (int kt = 0; kt < nb; kt += 16) {
            const int read_row = row_base + ly;
            const int read_kb = kt + lx;
            __local uchar * mine = tile + (ly*16 + lx)*18;
            if (read_row < m && read_kb < nb) {
                __global const uchar * src = aos + (read_row*nb + read_kb)*18;
                for (int p = 0; p < 18; ++p) mine[p] = src[p];
            } else {
                for (int p = 0; p < 18; ++p) mine[p] = 0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            const int kb = kt + ly;
            if (row < m && kb < nb) {
                __local uchar * src = tile + (lx*16 + ly)*18;
                const ushort dh = (ushort) src[0] | ((ushort) src[1] << 8);
                const float d = fp16_bits_to_float(dh);
                const int x0 = kb*32;
                for (int p = 0; p < 16; ++p) {
                    const uchar q = src[2+p];
                    sum = fma(d*((float)(q & 15) - 8.0f), xvec[x0+p], sum);
                    sum = fma(d*((float)(q >> 4) - 8.0f), xvec[x0+p+16], sum);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    partial[ly*16 + lx] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ly == 0 && row < m) {
        float total = 0.0f;
        for (int lane = 0; lane < 16; ++lane) total += partial[lane*16 + lx];
        output[row] = total;
    }
}


__kernel void gemv_fp32_q4_tiled(
        __global const float * scales,
        __global const uchar * qplanes,
        __global const float * xvec,
        __global float * output,
        int m, int nb) {
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int row = get_group_id(1)*16 + lx;
    const int nblocks = m*nb;
    __local float partial[16*16];
    float sum = 0.0f;
    if (row < m) {
        for (int kb = ly; kb < nb; kb += 16) {
            const int src = kb*m + row;
            const float d = scales[src];
            const int x0 = kb*32;
            for (int p = 0; p < 16; ++p) {
                const uchar q = qplanes[p*nblocks + src];
                sum = fma(d*((float)(q & 15) - 8.0f), xvec[x0+p], sum);
                sum = fma(d*((float)(q >> 4) - 8.0f), xvec[x0+p+16], sum);
            }
        }
    }
    partial[ly*16 + lx] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ly == 0 && row < m) {
        float total = 0.0f;
        for (int lane = 0; lane < 16; ++lane) total += partial[lane*16 + lx];
        output[row] = total;
    }
}

__kernel void gemv_expanded_tiled(
        __global const float * scales,
        __global const char * qvalues,
        __global const float * xvec,
        __global float * output,
        int m, int nb) {
    const int lx = get_local_id(0), ly = get_local_id(1);
    const int row = get_group_id(1)*16 + lx;
    const int nblocks = m*nb;
    __local float partial[16*16];
    float sum = 0.0f;
    if (row < m) {
        for (int kb = ly; kb < nb; kb += 16) {
            const int src = kb*m + row;
            const float d = scales[src];
            const int x0 = kb*32;
            for (int p = 0; p < 32; ++p) {
                sum = fma(d*convert_float(qvalues[p*nblocks + src]), xvec[x0+p], sum);
            }
        }
    }
    partial[ly*16 + lx] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ly == 0 && row < m) {
        float total = 0.0f;
        for (int lane = 0; lane < 16; ++lane) total += partial[lane*16 + lx];
        output[row] = total;
    }
}
)CLC";

struct stats {
    double min = 0, med = 0, avg = 0, max = 0;
};

struct overlap_stats {
    stats wall;
    stats transform;
    stats compute;
};

static stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v.front(), v[v.size()/2],
            std::accumulate(v.begin(), v.end(), 0.0)/double(v.size()), v.back()};
}

static double event_ms(cl_event event) {
    cl_ulong begin = 0, end = 0;
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(begin), &begin, nullptr));
    CL_CHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
    return double(end - begin)*1e-6;
}

static double event_span_ms(cl_event a, cl_event b) {
    cl_ulong a0 = 0, a1 = 0, b0 = 0, b1 = 0;
    CL_CHECK(clGetEventProfilingInfo(a, CL_PROFILING_COMMAND_START, sizeof(a0), &a0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(a, CL_PROFILING_COMMAND_END, sizeof(a1), &a1, nullptr));
    CL_CHECK(clGetEventProfilingInfo(b, CL_PROFILING_COMMAND_START, sizeof(b0), &b0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(b, CL_PROFILING_COMMAND_END, sizeof(b1), &b1, nullptr));
    return double(std::max(a1, b1) - std::min(a0, b0))*1e-6;
}

static double event_span_ms3(cl_event a, cl_event b, cl_event c) {
    cl_ulong a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0;
    CL_CHECK(clGetEventProfilingInfo(a, CL_PROFILING_COMMAND_START, sizeof(a0), &a0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(a, CL_PROFILING_COMMAND_END, sizeof(a1), &a1, nullptr));
    CL_CHECK(clGetEventProfilingInfo(b, CL_PROFILING_COMMAND_START, sizeof(b0), &b0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(b, CL_PROFILING_COMMAND_END, sizeof(b1), &b1, nullptr));
    CL_CHECK(clGetEventProfilingInfo(c, CL_PROFILING_COMMAND_START, sizeof(c0), &c0, nullptr));
    CL_CHECK(clGetEventProfilingInfo(c, CL_PROFILING_COMMAND_END, sizeof(c1), &c1, nullptr));
    return double(std::max({a1, b1, c1}) - std::min({a0, b0, c0}))*1e-6;
}

struct mem {
    cl_mem p = nullptr;
    mem() = default;
    mem(cl_context context, size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE) {
        cl_int err = CL_SUCCESS;
        p = clCreateBuffer(context, flags, std::max<size_t>(size, 1), nullptr, &err);
        CL_CHECK(err);
    }
    mem(const mem &) = delete;
    ~mem() { if (p) clReleaseMemObject(p); }
};

struct kernel {
    cl_kernel p = nullptr;
    kernel(cl_program program, const char * name) {
        cl_int err = CL_SUCCESS;
        p = clCreateKernel(program, name, &err);
        CL_CHECK(err);
    }
    ~kernel() { if (p) clReleaseKernel(p); }
};

static void set_mem(cl_kernel k, cl_uint i, cl_mem value) {
    CL_CHECK(clSetKernelArg(k, i, sizeof(value), &value));
}

template <typename T>
static void set_value(cl_kernel k, cl_uint i, T value) {
    CL_CHECK(clSetKernelArg(k, i, sizeof(value), &value));
}

static cl_event enqueue(cl_command_queue q, cl_kernel k, size_t global, size_t local = 64) {
    global = ((global + local - 1)/local)*local;
    cl_event event = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, &event));
    return event;
}

static cl_event enqueue_2d(cl_command_queue q, cl_kernel k, size_t gx, size_t gy) {
    const size_t local[2] = {16, 16};
    const size_t global[2] = {((gx + 15)/16)*16, ((gy + 15)/16)*16};
    cl_event event = nullptr;
    CL_CHECK(clEnqueueNDRangeKernel(q, k, 2, nullptr, global, local, 0, nullptr, &event));
    return event;
}

static cl_program build_program(cl_context context, cl_device_id device) {
    const size_t length = std::strlen(kernel_source);
    cl_int err = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, &length, &err);
    CL_CHECK(err);
    err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t n = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
        std::string log(n, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
        clReleaseProgram(program);
        throw std::runtime_error("OpenCL build failed:\n" + log);
    }
    return program;
}

static void fill_weight(std::vector<uint8_t> & data, int64_t k, int64_t m) {
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, k);
    std::vector<float> row(static_cast<size_t>(k));
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t c = 0; c < k; ++c) {
            row[size_t(c)] = float(((c*17 + r*13) % 127) - 63)/64.0f;
        }
        const size_t written = ggml_quantize_chunk(
            GGML_TYPE_Q4_0, row.data(), data.data() + size_t(r)*row_bytes, 0, 1, k, nullptr);
        if (written != row_bytes) throw std::runtime_error("unexpected Q4_0 row size");
    }
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) result = std::max(result, double(std::abs(a[i] - b[i])));
    return result;
}

static std::vector<int> read_trace_pattern(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to open trace pattern: " + path);
    std::vector<int> result;
    std::string line;
    std::getline(input, line); // header
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const size_t comma = line.rfind(',');
        if (comma == std::string::npos) throw std::runtime_error("invalid trace pattern row");
        result.push_back(std::atoi(line.c_str() + comma + 1) != 0);
    }
    if (result.empty()) throw std::runtime_error("empty trace pattern: " + path);
    return result;
}

struct row_granularity_result {
    int parts = 0;
    int tile_rows = 0;
    double tile_mib = 0.0;
    double host_peak_mib = 0.0;
    double cl_buffer_peak_mib = 0.0;
    stats load_wall;
    stats cl_chain_wall;
    stats cl_host_driver_gap;
    stats cl_write_event;
    stats transform_event;
    stats compute_event;
    stats pipeline_wall;
    stats max_tile_wall;
    double max_abs_diff = 0.0;
};

static void write_load_file(const std::string & path, const std::vector<uint8_t> & data) {
    const int fd = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0) {
        throw std::runtime_error("failed to create load file: " + path +
                                 " errno=" + std::to_string(errno));
    }
    size_t written = 0;
    while (written < data.size()) {
        const ssize_t rc = write(fd, data.data() + written, data.size() - written);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) {
            const int saved_errno = errno ? errno : EIO;
            close(fd);
            throw std::runtime_error("failed to write load file errno=" +
                                     std::to_string(saved_errno));
        }
        written += size_t(rc);
    }
    if (fsync(fd) != 0) {
        const int saved_errno = errno;
        close(fd);
        throw std::runtime_error("failed to sync load file errno=" +
                                 std::to_string(saved_errno));
    }
    close(fd);
}

static std::vector<row_granularity_result> run_row_granularity_bench(
        cl_context context,
        cl_command_queue queue,
        cl_program program,
        int k,
        int m,
        int warmup,
        int iters,
        int max_parts,
        bool reverse_parts,
        bool stage_sync,
        const std::string & load_file,
        bool prepare_load_file,
        std::vector<uint8_t> & host_weight,
        const std::vector<float> & host_x) {
    if (load_file.empty()) {
        throw std::runtime_error("--granularity-only requires --load-file");
    }
    if (max_parts <= 0 || (max_parts & (max_parts - 1)) != 0 || m % max_parts != 0) {
        throw std::runtime_error("--max-row-parts must be a power of two that divides M");
    }
    if ((m / max_parts) % 16 != 0) {
        throw std::runtime_error("the smallest row tile must contain a multiple of 16 rows");
    }

    const int nb = k / 32;
    const size_t row_bytes = size_t(nb) * 18;
    const size_t weight_bytes = size_t(m) * row_bytes;
    constexpr size_t direct_alignment = 4096;
    if (prepare_load_file) write_load_file(load_file, host_weight);

    struct stat disk_stat {};
    const int disk_fd = open(load_file.c_str(), O_RDONLY | O_DIRECT);
    if (disk_fd < 0) {
        throw std::runtime_error("failed to open O_DIRECT load file: " + load_file +
                                 " errno=" + std::to_string(errno));
    }
    if (fstat(disk_fd, &disk_stat) != 0 || uint64_t(disk_stat.st_size) < weight_bytes) {
        close(disk_fd);
        throw std::runtime_error("load file is smaller than the matrix");
    }

    // The generated source is no longer needed after it has been persisted.
    // Releasing it keeps the measured host footprint equal to the reusable tile
    // staging buffer rather than silently retaining a full-matrix copy.
    host_weight.clear();
    host_weight.shrink_to_fit();

    std::vector<int> part_order;
    for (int parts = 1; parts <= max_parts; parts *= 2) part_order.push_back(parts);
    if (reverse_parts) std::reverse(part_order.begin(), part_order.end());

    std::vector<row_granularity_result> results;
    std::vector<std::vector<float>> outputs;
    for (int parts : part_order) {
        const int tile_rows = m / parts;
        const size_t tile_blocks = size_t(tile_rows) * size_t(nb);
        const size_t tile_weight_bytes = tile_blocks * 18;
        const size_t tile_scale_bytes = tile_blocks * 2;
        const size_t tile_q_bytes = tile_blocks * 16;
        if (tile_weight_bytes % direct_alignment != 0) {
            close(disk_fd);
            throw std::runtime_error("row-tile byte size is not O_DIRECT aligned");
        }

        void * aligned = nullptr;
        if (posix_memalign(&aligned, direct_alignment, tile_weight_bytes) != 0) {
            close(disk_fd);
            throw std::runtime_error("failed to allocate aligned row-tile staging buffer");
        }
        std::unique_ptr<void, decltype(&free)> staging(aligned, &free);

        mem tile_aos(context, tile_weight_bytes);
        mem tile_d(context, tile_scale_bytes);
        mem tile_q(context, tile_q_bytes);
        mem x(context, size_t(k) * sizeof(float));
        mem tile_y(context, size_t(tile_rows) * sizeof(float));
        CL_CHECK(clEnqueueWriteBuffer(queue, x.p, CL_TRUE, 0,
                                      size_t(k) * sizeof(float), host_x.data(),
                                      0, nullptr, nullptr));

        kernel transform(program, "aos_to_block_tiled");
        kernel compute(program, "gemv_block_tiled");
        set_mem(transform.p, 0, tile_aos.p);
        set_mem(transform.p, 1, tile_d.p);
        set_mem(transform.p, 2, tile_q.p);
        set_value(transform.p, 3, tile_rows);
        set_value(transform.p, 4, nb);
        set_mem(compute.p, 0, tile_d.p);
        set_mem(compute.p, 1, tile_q.p);
        set_mem(compute.p, 2, x.p);
        set_mem(compute.p, 3, tile_y.p);
        set_value(compute.p, 4, tile_rows);
        set_value(compute.p, 5, nb);

        std::vector<double> load_wall_samples, cl_chain_wall_samples, cl_host_driver_gap_samples;
        std::vector<double> cl_write_event_samples, transform_event_samples;
        std::vector<double> compute_event_samples;
        std::vector<double> pipeline_wall_samples, max_tile_wall_samples;
        auto wall_ms = [](const std::chrono::steady_clock::time_point & begin) {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
        };

        for (int trial = 0; trial < warmup + iters; ++trial) {
            double load_total = 0.0, cl_chain_wall_total = 0.0;
            double cl_write_event_total = 0.0, transform_event_total = 0.0;
            double compute_event_total = 0.0;
            double max_tile_wall = 0.0;
            const auto pipeline_begin = std::chrono::steady_clock::now();
            for (int tile = 0; tile < parts; ++tile) {
                const auto tile_begin = std::chrono::steady_clock::now();
                const off_t offset = off_t(size_t(tile) * tile_weight_bytes);

                const auto load_begin = std::chrono::steady_clock::now();
                ssize_t got = -1;
                do {
                    got = pread(disk_fd, staging.get(), tile_weight_bytes, offset);
                } while (got < 0 && errno == EINTR);
                load_total += wall_ms(load_begin);
                if (got != ssize_t(tile_weight_bytes)) {
                    close(disk_fd);
                    throw std::runtime_error("O_DIRECT row-tile read failed errno=" +
                                             std::to_string(errno ? errno : EIO));
                }

                const auto cl_chain_begin = std::chrono::steady_clock::now();
                cl_event upload = nullptr;
                CL_CHECK(clEnqueueWriteBuffer(queue, tile_aos.p, CL_FALSE, 0,
                                              tile_weight_bytes, staging.get(),
                                              0, nullptr, &upload));
                if (stage_sync) CL_CHECK(clWaitForEvents(1, &upload));

                cl_event transformed = enqueue_2d(queue, transform.p, nb, tile_rows);
                if (stage_sync) CL_CHECK(clWaitForEvents(1, &transformed));

                cl_event computed = enqueue_2d(queue, compute.p, 16, tile_rows);
                CL_CHECK(clWaitForEvents(1, &computed));
                cl_chain_wall_total += wall_ms(cl_chain_begin);
                cl_write_event_total += event_ms(upload);
                transform_event_total += event_ms(transformed);
                compute_event_total += event_ms(computed);
                clReleaseEvent(upload);
                clReleaseEvent(transformed);
                clReleaseEvent(computed);
                max_tile_wall = std::max(max_tile_wall, wall_ms(tile_begin));
            }
            const double pipeline_wall = wall_ms(pipeline_begin);
            if (trial >= warmup) {
                load_wall_samples.push_back(load_total);
                cl_chain_wall_samples.push_back(cl_chain_wall_total);
                cl_host_driver_gap_samples.push_back(
                    cl_chain_wall_total - cl_write_event_total -
                    transform_event_total - compute_event_total);
                cl_write_event_samples.push_back(cl_write_event_total);
                transform_event_samples.push_back(transform_event_total);
                compute_event_samples.push_back(compute_event_total);
                pipeline_wall_samples.push_back(pipeline_wall);
                max_tile_wall_samples.push_back(max_tile_wall);
            }
        }

        // Correctness is checked outside the timed loop so that D2H copies do
        // not contaminate the operator pipeline measurements.
        std::vector<float> output(static_cast<size_t>(m));
        for (int tile = 0; tile < parts; ++tile) {
            const off_t offset = off_t(size_t(tile) * tile_weight_bytes);
            ssize_t got = -1;
            do {
                got = pread(disk_fd, staging.get(), tile_weight_bytes, offset);
            } while (got < 0 && errno == EINTR);
            if (got != ssize_t(tile_weight_bytes)) {
                close(disk_fd);
                throw std::runtime_error("O_DIRECT correctness read failed");
            }
            CL_CHECK(clEnqueueWriteBuffer(queue, tile_aos.p, CL_TRUE, 0,
                                          tile_weight_bytes, staging.get(),
                                          0, nullptr, nullptr));
            cl_event transformed = enqueue_2d(queue, transform.p, nb, tile_rows);
            CL_CHECK(clWaitForEvents(1, &transformed));
            clReleaseEvent(transformed);
            cl_event computed = enqueue_2d(queue, compute.p, 16, tile_rows);
            CL_CHECK(clWaitForEvents(1, &computed));
            clReleaseEvent(computed);
            CL_CHECK(clEnqueueReadBuffer(queue, tile_y.p, CL_TRUE, 0,
                                         size_t(tile_rows) * sizeof(float),
                                         output.data() + size_t(tile) * size_t(tile_rows),
                                         0, nullptr, nullptr));
        }
        row_granularity_result result;
        result.parts = parts;
        result.tile_rows = tile_rows;
        result.tile_mib = double(tile_weight_bytes) / 1048576.0;
        result.host_peak_mib = result.tile_mib;
        result.cl_buffer_peak_mib = double(tile_weight_bytes + tile_scale_bytes + tile_q_bytes +
                                           size_t(k) * sizeof(float) +
                                           size_t(tile_rows) * sizeof(float)) / 1048576.0;
        result.load_wall = summarize(load_wall_samples);
        result.cl_chain_wall = summarize(cl_chain_wall_samples);
        result.cl_host_driver_gap = summarize(cl_host_driver_gap_samples);
        result.cl_write_event = summarize(cl_write_event_samples);
        result.transform_event = summarize(transform_event_samples);
        result.compute_event = summarize(compute_event_samples);
        result.pipeline_wall = summarize(pipeline_wall_samples);
        result.max_tile_wall = summarize(max_tile_wall_samples);
        results.push_back(result);
        outputs.push_back(std::move(output));
    }
    close(disk_fd);
    size_t reference_index = 0;
    while (reference_index < results.size() && results[reference_index].parts != 1) {
        ++reference_index;
    }
    if (reference_index == results.size()) {
        throw std::runtime_error("row-granularity sweep is missing the whole-matrix reference");
    }
    for (size_t i = 0; i < results.size(); ++i) {
        results[i].max_abs_diff = max_abs_diff(outputs[reference_index], outputs[i]);
    }
    return results;
}

int main(int argc, char ** argv) {
    int k = 4096, m = 4096, warmup = 10, iters = 50;
    int max_row_parts = 32;
    bool granularity_only = false;
    bool prepare_load_file = false;
    bool reverse_row_parts = false;
    bool granularity_stage_sync = false;
    std::string trace_pattern;
    std::string trace_policy = "greedy";
    std::string trace_transform = "tiled";
    std::string load_file;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * {
            if (++i >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[i];
        };
        if (arg == "--k") k = std::atoi(next());
        else if (arg == "--m") m = std::atoi(next());
        else if (arg == "--warmup") warmup = std::atoi(next());
        else if (arg == "--iters") iters = std::atoi(next());
        else if (arg == "--trace-pattern") trace_pattern = next();
        else if (arg == "--trace-policy") trace_policy = next();
        else if (arg == "--trace-transform") trace_transform = next();
        else if (arg == "--load-file") load_file = next();
        else if (arg == "--granularity-only") granularity_only = true;
        else if (arg == "--max-row-parts") max_row_parts = std::atoi(next());
        else if (arg == "--prepare-load-file") prepare_load_file = true;
        else if (arg == "--reverse-row-parts") reverse_row_parts = true;
        else if (arg == "--granularity-stage-sync") granularity_stage_sync = true;
        else if (arg == "--help") {
            std::printf("usage: %s [--k K] [--m M] [--warmup N] [--iters N] "
                        "[--load-file PATH] "
                        "[--granularity-only --max-row-parts N --prepare-load-file "
                        "--reverse-row-parts --granularity-stage-sync] "
                        "[--trace-pattern CSV --trace-policy generic|greedy "
                        "--trace-transform naive|tiled]\n", argv[0]);
            return 0;
        } else throw std::runtime_error("unknown argument: " + arg);
    }
    if (k <= 0 || m <= 0 || k%32 != 0 || warmup < 0 || iters <= 0) {
        throw std::runtime_error("K must be a positive multiple of 32; M/iters must be positive");
    }
    if (trace_policy != "generic" && trace_policy != "greedy") {
        throw std::runtime_error("trace policy must be generic or greedy");
    }
    if (trace_transform != "naive" && trace_transform != "tiled") {
        throw std::runtime_error("trace transform must be naive or tiled");
    }

    cl_uint np = 0;
    CL_CHECK(clGetPlatformIDs(0, nullptr, &np));
    if (!np) throw std::runtime_error("no OpenCL platform");
    std::vector<cl_platform_id> platforms(np);
    CL_CHECK(clGetPlatformIDs(np, platforms.data(), nullptr));
    cl_uint nd = 0;
    CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &nd));
    if (!nd) throw std::runtime_error("no OpenCL device");
    std::vector<cl_device_id> devices(nd);
    CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, nd, devices.data(), nullptr));
    const cl_device_id device = devices[0];

    cl_int err = CL_SUCCESS;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);
    const cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    CL_CHECK(err);
    cl_command_queue queue2 = clCreateCommandQueueWithProperties(context, device, properties, &err);
    CL_CHECK(err);
    cl_program program = build_program(context, device);

    char name[256] = {};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::printf("DEVICE name=%s\n", name);

    const int nb = k/32;
    const size_t nblocks = size_t(m)*size_t(nb);
    const size_t weight_bytes = nblocks*18;
    const size_t scale_bytes = nblocks*2;
    const size_t q_bytes = nblocks*16;
    const size_t tiled_blocks = size_t((m + 15)/16)*size_t((nb + 15)/16)*256;
    std::vector<uint8_t> host_weight(weight_bytes);
    std::vector<float> host_x(static_cast<size_t>(k));
    std::vector<float> host_y(static_cast<size_t>(m));
    std::vector<float> reference(static_cast<size_t>(m));
    fill_weight(host_weight, k, m);
    for (int i = 0; i < k; ++i) host_x[size_t(i)] = float((i%31)-15)/31.0f;

    if (granularity_only) {
        const std::vector<row_granularity_result> granularity = run_row_granularity_bench(
            context, queue, program, k, m, warmup, iters, max_row_parts, reverse_row_parts,
            granularity_stage_sync, load_file, prepare_load_file, host_weight, host_x);
        const auto whole_it = std::find_if(granularity.begin(), granularity.end(),
            [](const row_granularity_result & row) { return row.parts == 1; });
        if (whole_it == granularity.end()) {
            throw std::runtime_error("row-granularity sweep is missing parts=1");
        }
        const double whole_pipeline = whole_it->pipeline_wall.med;
        const double whole_max_tile = whole_it->max_tile_wall.med;
        const double whole_cl_buffer_peak = whole_it->cl_buffer_peak_mib;
        std::printf("GRANULARITY_CONFIG K=%d M=%d weight_mib=%.3f warmup=%d iters=%d "
                    "max_parts=%d order=%s io=O_DIRECT schedule=serial_one_slot sync=%s\n",
                    k, m, double(size_t(m) * size_t(k / 32) * 18) / 1048576.0,
                    warmup, iters, max_row_parts,
                    reverse_row_parts ? "descending" : "ascending",
                    granularity_stage_sync ? "each_stage" : "tile_end");
        for (const row_granularity_result & row : granularity) {
            std::printf(
                "ROW_GRANULARITY parts=%d tile_rows=%d tile_mib=%.3f "
                "host_peak_mib=%.3f cl_buffer_peak_mib=%.3f controlled_peak_mib=%.3f "
                "cl_buffer_peak_ratio=%.6f load_wall_med_ms=%.6f "
                "cl_chain_wall_med_ms=%.6f cl_host_driver_gap_med_ms=%.6f "
                "cl_write_event_med_ms=%.6f "
                "transform_event_med_ms=%.6f compute_event_med_ms=%.6f "
                "pipeline_wall_med_ms=%.6f overhead_vs_whole=%.6f "
                "max_tile_wall_med_ms=%.6f max_tile_reduction=%.6f max_abs_diff=%.9g\n",
                row.parts, row.tile_rows, row.tile_mib, row.host_peak_mib,
                row.cl_buffer_peak_mib, row.host_peak_mib + row.cl_buffer_peak_mib,
                row.cl_buffer_peak_mib / whole_cl_buffer_peak,
                row.load_wall.med, row.cl_chain_wall.med, row.cl_host_driver_gap.med,
                row.cl_write_event.med,
                row.transform_event.med, row.compute_event.med,
                row.pipeline_wall.med, row.pipeline_wall.med / whole_pipeline,
                row.max_tile_wall.med, whole_max_tile / row.max_tile_wall.med,
                row.max_abs_diff);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue2);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }

    mem aos(context, weight_bytes), x(context, size_t(k)*sizeof(float)), y(context, size_t(m)*sizeof(float));
    CL_CHECK(clEnqueueWriteBuffer(queue, aos.p, CL_TRUE, 0, weight_bytes, host_weight.data(), 0, nullptr, nullptr));
    CL_CHECK(clEnqueueWriteBuffer(queue, x.p, CL_TRUE, 0, size_t(k)*sizeof(float), host_x.data(), 0, nullptr, nullptr));

    if (!trace_pattern.empty()) {
        const std::vector<int> pattern = read_trace_pattern(trace_pattern);
        double promotion_event_ms = 0.0;
        double compute_event_ms = 0.0;
        double allocation_wall_ms = 0.0;
        double release_wall_ms = 0.0;
        double max_promotion_ms = 0.0;
        int promotions = 0, evictions = 0, high_executions = 0;
        {
            kernel convert(program, "aos_to_split");
            kernel tiled_convert(program, "aos_to_block_tiled");
            kernel tiled_direct(program, "gemv_aos_tiled");
            kernel tiled_block_gemv(program, "gemv_block_tiled");
            set_mem(tiled_direct.p, 0, aos.p); set_mem(tiled_direct.p, 1, x.p);
            set_mem(tiled_direct.p, 2, y.p); set_value(tiled_direct.p, 3, m);
            set_value(tiled_direct.p, 4, nb);

            for (int i = 0; i < warmup; ++i) {
                cl_event event = enqueue_2d(queue, tiled_direct.p, 16, m);
                CL_CHECK(clWaitForEvents(1, &event)); clReleaseEvent(event);
            }

            std::unique_ptr<mem> block_d;
            std::unique_ptr<mem> block_q;
            bool promoted = false;
            for (int high : pattern) {
                high_executions += high;
                if (trace_policy == "greedy" && high && !promoted) {
                    const auto alloc_begin = std::chrono::steady_clock::now();
                    block_d = std::make_unique<mem>(context, scale_bytes);
                    block_q = std::make_unique<mem>(context, q_bytes);
                    allocation_wall_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - alloc_begin).count();
                    cl_event event = nullptr;
                    if (trace_transform == "tiled") {
                        set_mem(tiled_convert.p, 0, aos.p); set_mem(tiled_convert.p, 1, block_d->p);
                        set_mem(tiled_convert.p, 2, block_q->p); set_value(tiled_convert.p, 3, m);
                        set_value(tiled_convert.p, 4, nb);
                        event = enqueue_2d(queue, tiled_convert.p, nb, m);
                    } else {
                        set_mem(convert.p, 0, aos.p); set_mem(convert.p, 1, block_d->p);
                        set_mem(convert.p, 2, block_q->p); set_value(convert.p, 3, m);
                        set_value(convert.p, 4, nb); set_value(convert.p, 5, 1);
                        event = enqueue(queue, convert.p, nblocks);
                    }
                    CL_CHECK(clWaitForEvents(1, &event));
                    const double elapsed = event_ms(event);
                    clReleaseEvent(event);
                    promotion_event_ms += elapsed;
                    max_promotion_ms = std::max(max_promotion_ms, elapsed);
                    ++promotions;
                    promoted = true;
                    set_mem(tiled_block_gemv.p, 0, block_d->p);
                    set_mem(tiled_block_gemv.p, 1, block_q->p);
                    set_mem(tiled_block_gemv.p, 2, x.p); set_mem(tiled_block_gemv.p, 3, y.p);
                    set_value(tiled_block_gemv.p, 4, m); set_value(tiled_block_gemv.p, 5, nb);
                } else if ((!high || trace_policy == "generic") && promoted) {
                    CL_CHECK(clFinish(queue));
                    const auto release_begin = std::chrono::steady_clock::now();
                    block_q.reset();
                    block_d.reset();
                    release_wall_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - release_begin).count();
                    ++evictions;
                    promoted = false;
                }

                cl_event event = nullptr;
                if (trace_policy == "greedy" && high && promoted) {
                    event = enqueue_2d(queue, tiled_block_gemv.p, 16, m);
                } else {
                    event = enqueue_2d(queue, tiled_direct.p, 16, m);
                }
                CL_CHECK(clWaitForEvents(1, &event));
                compute_event_ms += event_ms(event);
                clReleaseEvent(event);
            }
            if (promoted) {
                CL_CHECK(clFinish(queue));
                block_q.reset(); block_d.reset();
            }
        }
        std::printf(
            "TRACE_RESULT policy=%s transform=%s K=%d M=%d executions=%zu high_executions=%d "
            "promotions=%d evictions=%d promotion_event_ms=%.6f compute_event_ms=%.6f "
            "total_event_ms=%.6f allocation_wall_ms=%.6f release_wall_ms=%.6f max_promotion_ms=%.6f\n",
            trace_policy.c_str(), trace_transform.c_str(), k, m, pattern.size(), high_executions,
            promotions, evictions, promotion_event_ms, compute_event_ms,
            promotion_event_ms + compute_event_ms, allocation_wall_ms, release_wall_ms, max_promotion_ms);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue2);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }

    mem row_d(context, scale_bytes), row_q(context, q_bytes);
    mem block_d(context, scale_bytes), block_q(context, q_bytes);
    mem kmajor_aos(context, weight_bytes), tile16_aos(context, tiled_blocks*18);
    mem fp32_d(context, nblocks*4), fp32_q(context, nblocks*16);
    mem expanded_d(context, nblocks*4), expanded_q(context, nblocks*32);
    mem overlap_d(context, scale_bytes), overlap_q(context, q_bytes);
    mem prep_source(context, weight_bytes), prep_aos(context, std::max(weight_bytes, tiled_blocks*18));
    mem prep_d(context, nblocks*4), prep_q(context, nblocks*32);
    CL_CHECK(clEnqueueWriteBuffer(queue, prep_source.p, CL_TRUE, 0, weight_bytes,
                                  host_weight.data(), 0, nullptr, nullptr));

    kernel convert(program, "aos_to_split");
    kernel tiled_convert(program, "aos_to_block_tiled");
    kernel chunked_convert(program, "aos_to_block_tiled_chunk");
    kernel direct(program, "gemv_aos");
    kernel split(program, "gemv_split");
    kernel fused(program, "promote_gemv");
    kernel tiled_fused(program, "promote_block_gemv_tiled");
    kernel tiled_direct(program, "gemv_aos_tiled");
    kernel tiled_block_gemv(program, "gemv_block_tiled");
    kernel row_soa_gemv(program, "gemv_row_soa_tiled");
    kernel kmajor_aos_convert(program, "aos_to_kmajor_aos");
    kernel tile16_aos_convert(program, "aos_to_tile16_aos");
    kernel kmajor_aos_gemv(program, "gemv_kmajor_aos_tiled");
    kernel tile16_aos_gemv(program, "gemv_tile16_aos_tiled");
    kernel mixed_gemv(program, "gemv_mixed_tiled");
    kernel fp32_convert(program, "aos_to_fp32_q4_tiled");
    kernel expanded_convert(program, "aos_to_expanded_tiled");
    kernel fp32_gemv(program, "gemv_fp32_q4_tiled");
    kernel expanded_gemv(program, "gemv_expanded_tiled");

    auto configure_convert = [&](cl_mem d, cl_mem q, int layout) {
        set_mem(convert.p, 0, aos.p); set_mem(convert.p, 1, d); set_mem(convert.p, 2, q);
        set_value(convert.p, 3, m); set_value(convert.p, 4, nb); set_value(convert.p, 5, layout);
    };
    auto configure_split = [&](cl_mem d, cl_mem q, int layout) {
        set_mem(split.p, 0, d); set_mem(split.p, 1, q); set_mem(split.p, 2, x.p); set_mem(split.p, 3, y.p);
        set_value(split.p, 4, m); set_value(split.p, 5, nb); set_value(split.p, 6, layout);
    };
    auto configure_fused = [&](cl_mem d, cl_mem q, int layout) {
        set_mem(fused.p, 0, aos.p); set_mem(fused.p, 1, d); set_mem(fused.p, 2, q);
        set_mem(fused.p, 3, x.p); set_mem(fused.p, 4, y.p);
        set_value(fused.p, 5, m); set_value(fused.p, 6, nb); set_value(fused.p, 7, layout);
    };
    set_mem(direct.p, 0, aos.p); set_mem(direct.p, 1, x.p); set_mem(direct.p, 2, y.p);
    set_value(direct.p, 3, m); set_value(direct.p, 4, nb);
    set_mem(tiled_convert.p, 0, aos.p); set_mem(tiled_convert.p, 1, block_d.p);
    set_mem(tiled_convert.p, 2, block_q.p); set_value(tiled_convert.p, 3, m);
    set_value(tiled_convert.p, 4, nb);
    set_mem(chunked_convert.p, 0, aos.p); set_mem(chunked_convert.p, 1, overlap_d.p);
    set_mem(chunked_convert.p, 2, overlap_q.p); set_value(chunked_convert.p, 3, m);
    set_value(chunked_convert.p, 4, nb);
    set_mem(tiled_fused.p, 0, aos.p); set_mem(tiled_fused.p, 1, block_d.p);
    set_mem(tiled_fused.p, 2, block_q.p); set_mem(tiled_fused.p, 3, x.p);
    set_mem(tiled_fused.p, 4, y.p); set_value(tiled_fused.p, 5, m);
    set_value(tiled_fused.p, 6, nb);
    set_mem(tiled_direct.p, 0, aos.p); set_mem(tiled_direct.p, 1, x.p);
    set_mem(tiled_direct.p, 2, y.p); set_value(tiled_direct.p, 3, m);
    set_value(tiled_direct.p, 4, nb);
    set_mem(tiled_block_gemv.p, 0, block_d.p); set_mem(tiled_block_gemv.p, 1, block_q.p);
    set_mem(tiled_block_gemv.p, 2, x.p); set_mem(tiled_block_gemv.p, 3, y.p);
    set_value(tiled_block_gemv.p, 4, m); set_value(tiled_block_gemv.p, 5, nb);
    set_mem(row_soa_gemv.p, 0, row_d.p); set_mem(row_soa_gemv.p, 1, row_q.p);
    set_mem(row_soa_gemv.p, 2, x.p); set_mem(row_soa_gemv.p, 3, y.p);
    set_value(row_soa_gemv.p, 4, m); set_value(row_soa_gemv.p, 5, nb);
    set_mem(kmajor_aos_convert.p, 0, aos.p); set_mem(kmajor_aos_convert.p, 1, kmajor_aos.p);
    set_value(kmajor_aos_convert.p, 2, m); set_value(kmajor_aos_convert.p, 3, nb);
    set_mem(tile16_aos_convert.p, 0, aos.p); set_mem(tile16_aos_convert.p, 1, tile16_aos.p);
    set_value(tile16_aos_convert.p, 2, m); set_value(tile16_aos_convert.p, 3, nb);
    set_mem(kmajor_aos_gemv.p, 0, kmajor_aos.p); set_mem(kmajor_aos_gemv.p, 1, x.p);
    set_mem(kmajor_aos_gemv.p, 2, y.p); set_value(kmajor_aos_gemv.p, 3, m);
    set_value(kmajor_aos_gemv.p, 4, nb);
    set_mem(tile16_aos_gemv.p, 0, tile16_aos.p); set_mem(tile16_aos_gemv.p, 1, x.p);
    set_mem(tile16_aos_gemv.p, 2, y.p); set_value(tile16_aos_gemv.p, 3, m);
    set_value(tile16_aos_gemv.p, 4, nb);
    set_mem(mixed_gemv.p, 0, aos.p); set_mem(mixed_gemv.p, 1, block_d.p);
    set_mem(mixed_gemv.p, 2, block_q.p); set_mem(mixed_gemv.p, 3, x.p);
    set_mem(mixed_gemv.p, 4, y.p); set_value(mixed_gemv.p, 5, m);
    set_value(mixed_gemv.p, 6, nb);
    set_mem(fp32_convert.p, 0, aos.p); set_mem(fp32_convert.p, 1, fp32_d.p);
    set_mem(fp32_convert.p, 2, fp32_q.p); set_value(fp32_convert.p, 3, m);
    set_value(fp32_convert.p, 4, nb);
    set_mem(expanded_convert.p, 0, aos.p); set_mem(expanded_convert.p, 1, expanded_d.p);
    set_mem(expanded_convert.p, 2, expanded_q.p); set_value(expanded_convert.p, 3, m);
    set_value(expanded_convert.p, 4, nb);
    set_mem(fp32_gemv.p, 0, fp32_d.p); set_mem(fp32_gemv.p, 1, fp32_q.p);
    set_mem(fp32_gemv.p, 2, x.p); set_mem(fp32_gemv.p, 3, y.p);
    set_value(fp32_gemv.p, 4, m); set_value(fp32_gemv.p, 5, nb);
    set_mem(expanded_gemv.p, 0, expanded_d.p); set_mem(expanded_gemv.p, 1, expanded_q.p);
    set_mem(expanded_gemv.p, 2, x.p); set_mem(expanded_gemv.p, 3, y.p);
    set_value(expanded_gemv.p, 4, m); set_value(expanded_gemv.p, 5, nb);

    // Produce both persistent layouts once before steady-state timing.
    configure_convert(row_d.p, row_q.p, 0);
    cl_event e = enqueue(queue, convert.p, nblocks); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    configure_convert(block_d.p, block_q.p, 1);
    e = enqueue(queue, convert.p, nblocks); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    e = enqueue_2d(queue, kmajor_aos_convert.p, nb, m);
    CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    e = enqueue_2d(queue, tile16_aos_convert.p, nb, m);
    CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);

    auto time_kernel = [&](cl_kernel kern, size_t global) {
        std::vector<double> times;
        for (int i = 0; i < warmup + iters; ++i) {
            cl_event event = enqueue(queue, kern, global);
            CL_CHECK(clWaitForEvents(1, &event));
            if (i >= warmup) times.push_back(event_ms(event));
            clReleaseEvent(event);
        }
        return summarize(times);
    };
    auto time_kernel_2d = [&](cl_kernel kern, size_t gx, size_t gy) {
        std::vector<double> times;
        for (int i = 0; i < warmup + iters; ++i) {
            cl_event event = enqueue_2d(queue, kern, gx, gy);
            CL_CHECK(clWaitForEvents(1, &event));
            if (i >= warmup) times.push_back(event_ms(event));
            clReleaseEvent(event);
        }
        return summarize(times);
    };
    auto read_y = [&]() {
        CL_CHECK(clEnqueueReadBuffer(queue, y.p, CL_TRUE, 0, host_y.size()*sizeof(float), host_y.data(), 0, nullptr, nullptr));
        return host_y;
    };
    const stats direct_s = time_kernel(direct.p, m);
    e = enqueue(queue, direct.p, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    reference = read_y();

    configure_split(row_d.p, row_q.p, 0);
    const stats row_s = time_kernel(split.p, m);
    e = enqueue(queue, split.p, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double row_diff = max_abs_diff(reference, read_y());

    configure_split(block_d.p, block_q.p, 1);
    const stats block_s = time_kernel(split.p, m);
    e = enqueue(queue, split.p, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double block_diff = max_abs_diff(reference, read_y());

    auto measure_convert = [&](cl_mem d, cl_mem q, int layout) {
        configure_convert(d, q, layout);
        return time_kernel(convert.p, nblocks);
    };
    const stats row_convert = measure_convert(row_d.p, row_q.p, 0);
    const stats block_convert = measure_convert(block_d.p, block_q.p, 1);
    const stats block_tiled_convert = time_kernel_2d(tiled_convert.p, nb, m);
    const stats kmajor_aos_convert_s = time_kernel_2d(kmajor_aos_convert.p, nb, m);
    const stats tile16_aos_convert_s = time_kernel_2d(tile16_aos_convert.p, nb, m);
    configure_split(block_d.p, block_q.p, 1);
    e = enqueue(queue, split.p, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double block_tiled_diff = max_abs_diff(reference, read_y());
    const stats fp32_convert_s = time_kernel_2d(fp32_convert.p, nb, m);
    const stats expanded_convert_s = time_kernel_2d(expanded_convert.p, nb, m);

    configure_fused(row_d.p, row_q.p, 0);
    const stats row_fused = time_kernel(fused.p, m);
    e = enqueue(queue, fused.p, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double row_fused_diff = max_abs_diff(reference, read_y());
    configure_split(row_d.p, row_q.p, 0);
    e = enqueue(queue, split.p, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double row_persist_diff = max_abs_diff(reference, read_y());

    configure_fused(block_d.p, block_q.p, 1);
    const stats block_fused = time_kernel(fused.p, m);
    e = enqueue(queue, fused.p, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double block_fused_diff = max_abs_diff(reference, read_y());
    configure_split(block_d.p, block_q.p, 1);
    e = enqueue(queue, split.p, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double block_persist_diff = max_abs_diff(reference, read_y());

    const stats block_tiled_fused = time_kernel_2d(tiled_fused.p, 16, m);
    e = enqueue_2d(queue, tiled_fused.p, 16, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double block_tiled_fused_diff = max_abs_diff(reference, read_y());
    configure_split(block_d.p, block_q.p, 1);
    e = enqueue(queue, split.p, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double block_tiled_persist_diff = max_abs_diff(reference, read_y());

    const stats direct_tiled_s = time_kernel_2d(tiled_direct.p, 16, m);
    e = enqueue_2d(queue, tiled_direct.p, 16, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double direct_tiled_diff = max_abs_diff(reference, read_y());
    const stats row_soa_gemv_s = time_kernel_2d(row_soa_gemv.p, 16, m);
    e = enqueue_2d(queue, row_soa_gemv.p, 16, m);
    CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double row_soa_gemv_diff = max_abs_diff(reference, read_y());
    const stats kmajor_aos_gemv_s = time_kernel_2d(kmajor_aos_gemv.p, 16, m);
    e = enqueue_2d(queue, kmajor_aos_gemv.p, 16, m);
    CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double kmajor_aos_gemv_diff = max_abs_diff(reference, read_y());
    const stats tile16_aos_gemv_s = time_kernel_2d(tile16_aos_gemv.p, 16, m);
    e = enqueue_2d(queue, tile16_aos_gemv.p, 16, m);
    CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double tile16_aos_gemv_diff = max_abs_diff(reference, read_y());
    const stats block_tiled_gemv_s = time_kernel_2d(tiled_block_gemv.p, 16, m);
    e = enqueue_2d(queue, tiled_block_gemv.p, 16, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double block_tiled_gemv_diff = max_abs_diff(reference, read_y());
    struct mixed_result { int promoted_rows; stats timing; double diff; };
    std::vector<mixed_result> mixed_results;
    for (int quarter = 0; quarter <= 4; ++quarter) {
        const int promoted_rows = ((m*quarter/4)/16)*16;
        set_value(mixed_gemv.p, 7, promoted_rows);
        const stats timing = time_kernel_2d(mixed_gemv.p, 16, m);
        e = enqueue_2d(queue, mixed_gemv.p, 16, m);
        CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
        mixed_results.push_back({promoted_rows, timing, max_abs_diff(reference, read_y())});
    }
    const stats fp32_gemv_s = time_kernel_2d(fp32_gemv.p, 16, m);
    e = enqueue_2d(queue, fp32_gemv.p, 16, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double fp32_gemv_diff = max_abs_diff(reference, read_y());
    const stats expanded_gemv_s = time_kernel_2d(expanded_gemv.p, 16, m);
    e = enqueue_2d(queue, expanded_gemv.p, 16, m); CL_CHECK(clWaitForEvents(1, &e)); clReleaseEvent(e);
    const double expanded_gemv_diff = max_abs_diff(reference, read_y());

    // Dynamic budget-up proxy: materialize the next weight on queue2 while the
    // current operator computes on queue1. The destination is disjoint from all
    // compute layouts, so measured delay is resource contention, not dependency.
    set_mem(tiled_convert.p, 0, aos.p); set_mem(tiled_convert.p, 1, overlap_d.p);
    set_mem(tiled_convert.p, 2, overlap_q.p); set_value(tiled_convert.p, 3, m);
    set_value(tiled_convert.p, 4, nb);
    auto time_overlap = [&](cl_kernel compute_kernel) {
        std::vector<double> wall, transform_times, compute_times;
        for (int i = 0; i < warmup + iters; ++i) {
            cl_event transform_event = nullptr, compute_event = nullptr;
            if ((i & 1) == 0) {
                transform_event = enqueue_2d(queue2, tiled_convert.p, nb, m);
                compute_event = enqueue_2d(queue, compute_kernel, 16, m);
            } else {
                compute_event = enqueue_2d(queue, compute_kernel, 16, m);
                transform_event = enqueue_2d(queue2, tiled_convert.p, nb, m);
            }
            CL_CHECK(clFlush(queue)); CL_CHECK(clFlush(queue2));
            CL_CHECK(clWaitForEvents(1, &transform_event));
            CL_CHECK(clWaitForEvents(1, &compute_event));
            const double elapsed = event_span_ms(transform_event, compute_event);
            if (i >= warmup) {
                wall.push_back(elapsed);
                transform_times.push_back(event_ms(transform_event));
                compute_times.push_back(event_ms(compute_event));
            }
            clReleaseEvent(transform_event); clReleaseEvent(compute_event);
        }
        return overlap_stats {summarize(wall), summarize(transform_times), summarize(compute_times)};
    };
    const overlap_stats overlap_raw = time_overlap(tiled_direct.p);
    const overlap_stats overlap_block = time_overlap(tiled_block_gemv.p);
    const overlap_stats overlap_fp32 = time_overlap(fp32_gemv.p);
    const overlap_stats overlap_expanded = time_overlap(expanded_gemv.p);

    // Measure the cost and largest per-token interruption when the same
    // promotion is divided into independently schedulable row chunks.
    struct chunk_stats { int chunks; stats total; stats max_chunk; };
    std::vector<chunk_stats> chunk_results;
    const int row_tiles = (m + 15)/16;
    for (int requested_chunks : {1, 2, 4, 8, 16, 32}) {
        const int chunks = std::min(requested_chunks, row_tiles);
        const int tiles_per_chunk = (row_tiles + chunks - 1)/chunks;
        std::vector<double> totals, maxima;
        for (int trial = 0; trial < warmup + iters; ++trial) {
            std::vector<cl_event> events;
            for (int tile0 = 0; tile0 < row_tiles; tile0 += tiles_per_chunk) {
                const int row0 = tile0*16;
                const int rows = std::min(m - row0, tiles_per_chunk*16);
                set_value(chunked_convert.p, 5, row0);
                set_value(chunked_convert.p, 6, rows);
                events.push_back(enqueue_2d(queue, chunked_convert.p, nb, rows));
            }
            CL_CHECK(clWaitForEvents(1, &events.back()));
            double total = 0.0, maximum = 0.0;
            for (cl_event event : events) {
                const double duration = event_ms(event);
                total += duration;
                maximum = std::max(maximum, duration);
                clReleaseEvent(event);
            }
            if (trial >= warmup) { totals.push_back(total); maxima.push_back(maximum); }
        }
        chunk_results.push_back({chunks, summarize(totals), summarize(maxima)});
    }

    // Conventional-layout contention matrix. The A compute layout uses its own
    // resident buffers; every B preparation reads/writes disjoint buffers.
    // This isolates shared-device/bandwidth contention from data dependencies.
    using submit_fn = std::function<cl_event(cl_command_queue)>;
    auto time_submit = [&](const submit_fn & submit) {
        std::vector<double> times;
        for (int i = 0; i < warmup + iters; ++i) {
            cl_event event = submit(queue2);
            CL_CHECK(clWaitForEvents(1, &event));
            if (i >= warmup) times.push_back(event_ms(event));
            clReleaseEvent(event);
        }
        return summarize(times);
    };

    set_mem(convert.p, 0, prep_source.p); set_mem(convert.p, 1, prep_d.p);
    set_mem(convert.p, 2, prep_q.p); set_value(convert.p, 3, m);
    set_value(convert.p, 4, nb); set_value(convert.p, 5, 0);
    const submit_fn prep_row_soa = [&](cl_command_queue q) { return enqueue(q, convert.p, nblocks); };

    set_mem(tiled_convert.p, 0, prep_source.p); set_mem(tiled_convert.p, 1, prep_d.p);
    set_mem(tiled_convert.p, 2, prep_q.p); set_value(tiled_convert.p, 3, m);
    set_value(tiled_convert.p, 4, nb);
    const submit_fn prep_kmajor_soa = [&](cl_command_queue q) { return enqueue_2d(q, tiled_convert.p, nb, m); };

    set_mem(kmajor_aos_convert.p, 0, prep_source.p); set_mem(kmajor_aos_convert.p, 1, prep_aos.p);
    set_value(kmajor_aos_convert.p, 2, m); set_value(kmajor_aos_convert.p, 3, nb);
    const submit_fn prep_kmajor_aos = [&](cl_command_queue q) { return enqueue_2d(q, kmajor_aos_convert.p, nb, m); };

    set_mem(tile16_aos_convert.p, 0, prep_source.p); set_mem(tile16_aos_convert.p, 1, prep_aos.p);
    set_value(tile16_aos_convert.p, 2, m); set_value(tile16_aos_convert.p, 3, nb);
    const submit_fn prep_tile16_aos = [&](cl_command_queue q) { return enqueue_2d(q, tile16_aos_convert.p, nb, m); };

    set_mem(expanded_convert.p, 0, prep_source.p); set_mem(expanded_convert.p, 1, prep_d.p);
    set_mem(expanded_convert.p, 2, prep_q.p); set_value(expanded_convert.p, 3, m);
    set_value(expanded_convert.p, 4, nb);
    const submit_fn prep_expanded = [&](cl_command_queue q) { return enqueue_2d(q, expanded_convert.p, nb, m); };

    const submit_fn prep_upload = [&](cl_command_queue q) {
        cl_event event = nullptr;
        CL_CHECK(clEnqueueWriteBuffer(q, prep_aos.p, CL_FALSE, 0, weight_bytes,
                                      host_weight.data(), 0, nullptr, &event));
        return event;
    };

    struct prep_case { const char * name; double extra_mib; submit_fn submit; stats isolated; };
    std::vector<prep_case> preparations;
    auto add_prep = [&](const char * name, double extra_mib, const submit_fn & submit) {
        preparations.push_back({name, extra_mib, submit, time_submit(submit)});
    };
    add_prep("host_upload", double(weight_bytes)/1048576.0, prep_upload);
    add_prep("row_soa_transform", double(weight_bytes + scale_bytes + q_bytes)/1048576.0, prep_row_soa);
    add_prep("kmajor_soa_transform", double(weight_bytes + scale_bytes + q_bytes)/1048576.0, prep_kmajor_soa);
    add_prep("kmajor_aos_transform", double(2*weight_bytes)/1048576.0, prep_kmajor_aos);
    add_prep("tile16_aos_transform", double(weight_bytes + tiled_blocks*18)/1048576.0, prep_tile16_aos);
    add_prep("expanded_i8_transform", double(weight_bytes + nblocks*36)/1048576.0, prep_expanded);

    struct compute_case { const char * name; cl_kernel kernel; const stats * isolated; };
    const std::vector<compute_case> computations = {
        {"row_aos", tiled_direct.p, &direct_tiled_s},
        {"row_soa", row_soa_gemv.p, &row_soa_gemv_s},
        {"kmajor_aos", kmajor_aos_gemv.p, &kmajor_aos_gemv_s},
        {"tile16_aos", tile16_aos_gemv.p, &tile16_aos_gemv_s},
        {"kmajor_soa", tiled_block_gemv.p, &block_tiled_gemv_s},
        {"kmajor_soa_fp32_scale", fp32_gemv.p, &fp32_gemv_s},
        {"kmajor_expanded_i8", expanded_gemv.p, &expanded_gemv_s},
    };

    auto time_contention = [&](cl_kernel compute_kernel, const submit_fn & prep) {
        std::vector<double> wall, prep_times, compute_times;
        for (int i = 0; i < warmup + iters; ++i) {
            cl_event prep_event = nullptr, compute_event = nullptr;
            if ((i & 1) == 0) {
                prep_event = prep(queue2);
                compute_event = enqueue_2d(queue, compute_kernel, 16, m);
            } else {
                compute_event = enqueue_2d(queue, compute_kernel, 16, m);
                prep_event = prep(queue2);
            }
            CL_CHECK(clFlush(queue)); CL_CHECK(clFlush(queue2));
            CL_CHECK(clWaitForEvents(1, &prep_event));
            CL_CHECK(clWaitForEvents(1, &compute_event));
            if (i >= warmup) {
                wall.push_back(event_span_ms(prep_event, compute_event));
                prep_times.push_back(event_ms(prep_event));
                compute_times.push_back(event_ms(compute_event));
            }
            clReleaseEvent(prep_event); clReleaseEvent(compute_event);
        }
        return overlap_stats {summarize(wall), summarize(prep_times), summarize(compute_times)};
    };

    // A host-side shared-memory load proxy models the storage/load stage that
    // can overlap with GPU compute when the current budget provides a staging
    // buffer. Unlike two GPU kernels on this device, this pressure runs on a
    // separate engine and therefore exercises the shared LPDDR path.
    const size_t cpu_pressure_bytes = 128u * 1024u * 1024u;
    std::vector<uint8_t> cpu_pressure(cpu_pressure_bytes, 1);
    std::atomic<int> pressure_state {0};
    std::atomic<uint64_t> pressure_sink {0};
    auto time_cpu_pressure = [&](cl_kernel compute_kernel, const stats & isolated) {
        std::vector<double> under, wall;
        for (int i = 0; i < warmup + iters; ++i) {
            pressure_state.store(0, std::memory_order_release);
            std::thread pressure([&] {
                while (pressure_state.load(std::memory_order_acquire) == 0) std::this_thread::yield();
                uint64_t sum = 0;
                while (pressure_state.load(std::memory_order_acquire) == 1) {
                    for (size_t off = 0; off < cpu_pressure.size(); off += 64) sum += cpu_pressure[off];
                }
                pressure_sink.fetch_add(sum, std::memory_order_relaxed);
            });
            const auto wall_begin = std::chrono::steady_clock::now();
            pressure_state.store(1, std::memory_order_release);
            cl_event compute_event = enqueue_2d(queue, compute_kernel, 16, m);
            CL_CHECK(clWaitForEvents(1, &compute_event));
            pressure_state.store(2, std::memory_order_release);
            pressure.join();
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - wall_begin).count();
            if (i >= warmup) {
                under.push_back(event_ms(compute_event));
                wall.push_back(elapsed);
            }
            clReleaseEvent(compute_event);
        }
        (void) isolated;
        return std::pair<stats, stats> {summarize(under), summarize(wall)};
    };

    struct contention_result {
        const compute_case * compute;
        const prep_case * prep;
        overlap_stats timing;
    };
    std::vector<contention_result> contention_results;
    for (const prep_case & prep : preparations) {
        for (const compute_case & compute : computations) {
            contention_results.push_back({&compute, &prep, time_contention(compute.kernel, prep.submit)});
        }
    }

    // End-to-end A->B path: A computes while B is prepared, then B consumes
    // the actual prepared representation. Each target has a distinct
    // transition footprint and preparation rate, making the output directly
    // usable for current-budget mode selection.
    auto find_prep = [&](const char * name) -> const prep_case * {
        for (const prep_case & prep : preparations) {
            if (std::string(prep.name) == name) return &prep;
        }
        throw std::runtime_error(std::string("missing preparation case: ") + name);
    };
    struct pipeline_target {
        const char * name;
        const prep_case * prep;
        cl_kernel kernel;
        const stats * isolated;
        std::function<void()> configure;
    };
    std::vector<pipeline_target> pipeline_targets;
    pipeline_targets.push_back({"row_soa", find_prep("row_soa_transform"), row_soa_gemv.p,
                                &row_soa_gemv_s, [&] {
        set_mem(row_soa_gemv.p, 0, prep_d.p); set_mem(row_soa_gemv.p, 1, prep_q.p);
        set_mem(row_soa_gemv.p, 2, x.p); set_mem(row_soa_gemv.p, 3, y.p);
        set_value(row_soa_gemv.p, 4, m); set_value(row_soa_gemv.p, 5, nb);
    }});
    pipeline_targets.push_back({"tile16_aos", find_prep("tile16_aos_transform"), tile16_aos_gemv.p,
                                &tile16_aos_gemv_s, [&] {
        set_mem(tile16_aos_gemv.p, 0, prep_aos.p); set_mem(tile16_aos_gemv.p, 1, x.p);
        set_mem(tile16_aos_gemv.p, 2, y.p); set_value(tile16_aos_gemv.p, 3, m);
        set_value(tile16_aos_gemv.p, 4, nb);
    }});
    pipeline_targets.push_back({"kmajor_aos", find_prep("kmajor_aos_transform"), kmajor_aos_gemv.p,
                                &kmajor_aos_gemv_s, [&] {
        set_mem(kmajor_aos_gemv.p, 0, prep_aos.p); set_mem(kmajor_aos_gemv.p, 1, x.p);
        set_mem(kmajor_aos_gemv.p, 2, y.p); set_value(kmajor_aos_gemv.p, 3, m);
        set_value(kmajor_aos_gemv.p, 4, nb);
    }});
    pipeline_targets.push_back({"kmajor_soa", find_prep("kmajor_soa_transform"), tiled_block_gemv.p,
                                &block_tiled_gemv_s, [&] {
        set_mem(tiled_block_gemv.p, 0, prep_d.p); set_mem(tiled_block_gemv.p, 1, prep_q.p);
        set_mem(tiled_block_gemv.p, 2, x.p); set_mem(tiled_block_gemv.p, 3, y.p);
        set_value(tiled_block_gemv.p, 4, m); set_value(tiled_block_gemv.p, 5, nb);
    }});
    pipeline_targets.push_back({"expanded_i8", find_prep("expanded_i8_transform"), expanded_gemv.p,
                                &expanded_gemv_s, [&] {
        set_mem(expanded_gemv.p, 0, prep_d.p); set_mem(expanded_gemv.p, 1, prep_q.p);
        set_mem(expanded_gemv.p, 2, x.p); set_mem(expanded_gemv.p, 3, y.p);
        set_value(expanded_gemv.p, 4, m); set_value(expanded_gemv.p, 5, nb);
    }});
    auto time_pipeline = [&](const compute_case & a, const pipeline_target & b) {
        std::vector<double> spans, a_times, prep_times, b_times;
        for (int i = 0; i < warmup + iters; ++i) {
            b.configure();
            cl_event a_event = enqueue_2d(queue, a.kernel, 16, m);
            cl_event prep_event = b.prep->submit(queue2);
            CL_CHECK(clFlush(queue)); CL_CHECK(clFlush(queue2));
            CL_CHECK(clWaitForEvents(1, &a_event));
            CL_CHECK(clWaitForEvents(1, &prep_event));
            b.configure();
            cl_event b_event = enqueue_2d(queue, b.kernel, 16, m);
            CL_CHECK(clWaitForEvents(1, &b_event));
            if (i >= warmup) {
                spans.push_back(event_span_ms3(a_event, prep_event, b_event));
                a_times.push_back(event_ms(a_event));
                prep_times.push_back(event_ms(prep_event));
                b_times.push_back(event_ms(b_event));
            }
            clReleaseEvent(a_event); clReleaseEvent(prep_event); clReleaseEvent(b_event);
        }
        return std::array<stats, 4> {summarize(spans), summarize(a_times), summarize(prep_times), summarize(b_times)};
    };
    for (const pipeline_target & target : pipeline_targets) {
        for (const compute_case & a : computations) {
            const auto pipeline = time_pipeline(a, target);
            std::printf("PIPELINE A_layout=%s B_layout=%s prep_extra_mib=%.3f "
                        "A_iso_ms=%.6f prep_iso_ms=%.6f B_iso_ms=%.6f "
                        "span_ms=%.6f A_under_ms=%.6f prep_under_ms=%.6f B_ms=%.6f\n",
                        a.name, target.name, target.prep->extra_mib, a.isolated->med,
                        target.prep->isolated.med, target.isolated->med, pipeline[0].med,
                        pipeline[1].med, pipeline[2].med, pipeline[3].med);
        }
    }

    const double mib = double(weight_bytes)/1048576.0;
    std::printf("META K=%d M=%d blocks=%zu weight_mib=%.3f direct_peak_mib=%.3f promotion_peak_mib=%.3f\n",
                k, m, nblocks, mib, mib, 2*mib);
    auto result = [&](const char * variant, const char * layout, const char * coupling,
                      const stats & s, double diff) {
        std::printf("RESULT variant=%s layout=%s coupling=%s med_ms=%.6f avg_ms=%.6f min_ms=%.6f max_ms=%.6f max_abs_diff=%.9g\n",
                    variant, layout, coupling, s.med, s.avg, s.min, s.max, diff);
    };
    std::printf("LAYOUT layout=raw_aos persistent_mib=%.3f transition_peak_mib=%.3f working_local_kib=5.500\n",
                mib, mib);
    result("gemv", "raw_aos", "none", direct_s, 0.0);
    result("gemv_tiled", "raw_aos", "none", direct_tiled_s, direct_tiled_diff);
    result("gemv_tiled", "row_soa", "none", row_soa_gemv_s, row_soa_gemv_diff);
    result("materialize", "kmajor_aos", "separate", kmajor_aos_convert_s, 0.0);
    result("gemv_tiled", "kmajor_aos", "none", kmajor_aos_gemv_s, kmajor_aos_gemv_diff);
    result("materialize", "tile16_aos", "separate", tile16_aos_convert_s, 0.0);
    result("gemv_tiled", "tile16_aos", "none", tile16_aos_gemv_s, tile16_aos_gemv_diff);
    result("materialize", "row_split", "separate", row_convert, 0.0);
    result("gemv", "row_split", "separate", row_s, row_diff);
    result("first_gemv", "row_split", "fused_persist", row_fused, row_fused_diff);
    std::printf("CHECK layout=row_split after_fused_persist_max_abs_diff=%.9g\n", row_persist_diff);
    result("materialize", "block_split", "separate", block_convert, 0.0);
    result("materialize_tiled", "block_split", "separate", block_tiled_convert, block_tiled_diff);
    result("gemv", "block_split", "separate", block_s, block_diff);
    std::printf("LAYOUT layout=block_split persistent_mib=%.3f transition_peak_mib=%.3f working_local_kib=1.000\n",
                mib, 2*mib);
    result("gemv_tiled", "block_split", "separate", block_tiled_gemv_s, block_tiled_gemv_diff);
    const double mixed_raw_endpoint = mixed_results.front().timing.med;
    const double mixed_block_endpoint = mixed_results.back().timing.med;
    for (const mixed_result & mixed : mixed_results) {
        const double fraction = double(mixed.promoted_rows)/double(m);
        const double linear = (1.0 - fraction)*mixed_raw_endpoint + fraction*mixed_block_endpoint;
        std::printf("MIXED_LAYOUT promoted_fraction=%.6f med_ms=%.6f linear_expected_ms=%.6f "
                    "mixture_penalty=%.6f max_abs_diff=%.9g\n",
                    fraction, mixed.timing.med, linear, mixed.timing.med/linear, mixed.diff);
    }
    std::printf("LAYOUT layout=fp32_scale_q4 persistent_mib=%.3f transition_peak_mib=%.3f working_local_kib=1.000\n",
                double(nblocks*20)/1048576.0, double(weight_bytes + nblocks*20)/1048576.0);
    result("materialize_tiled", "fp32_scale_q4", "separate", fp32_convert_s, 0.0);
    result("gemv_tiled", "fp32_scale_q4", "separate", fp32_gemv_s, fp32_gemv_diff);
    std::printf("LAYOUT layout=expanded_q4 persistent_mib=%.3f transition_peak_mib=%.3f working_local_kib=1.000\n",
                double(nblocks*36)/1048576.0, double(weight_bytes + nblocks*36)/1048576.0);
    result("materialize_tiled", "expanded_q4", "separate", expanded_convert_s, 0.0);
    result("gemv_tiled", "expanded_q4", "separate", expanded_gemv_s, expanded_gemv_diff);
    for (const prep_case & prep : preparations) {
        std::printf("PREP_STAGE name=%s extra_mib=%.3f isolated_med_ms=%.6f\n",
                    prep.name, prep.extra_mib, prep.isolated.med);
    }
    for (const contention_result & result_row : contention_results) {
        const stats & compute = *result_row.compute->isolated;
        const stats & prep = result_row.prep->isolated;
        const overlap_stats & pair = result_row.timing;
        const double ideal = std::max(compute.med, prep.med);
        const double serial = compute.med + prep.med;
        const double denom = std::max(1e-9, std::min(compute.med, prep.med));
        const double overlap_eff = (serial - pair.wall.med)/denom;
        std::printf("CONTENTION compute_layout=%s prep=%s prep_extra_mib=%.3f "
                    "compute_iso_ms=%.6f prep_iso_ms=%.6f span_ms=%.6f ideal_ms=%.6f "
                    "serial_ms=%.6f overlap_eff=%.6f compute_slowdown=%.6f prep_slowdown=%.6f\n",
                    result_row.compute->name, result_row.prep->name, result_row.prep->extra_mib,
                    compute.med, prep.med, pair.wall.med, ideal, serial, overlap_eff,
                    pair.compute.med/compute.med, pair.transform.med/prep.med);
    }
    for (const compute_case & compute : computations) {
        const auto pressure = time_cpu_pressure(compute.kernel, *compute.isolated);
        std::printf("CPU_PRESSURE compute_layout=%s gpu_iso_ms=%.6f gpu_under_ms=%.6f "
                    "gpu_slowdown=%.6f wall_med_ms=%.6f\n",
                    compute.name, compute.isolated->med, pressure.first.med,
                    pressure.first.med/compute.isolated->med, pressure.second.med);
    }

    if (!load_file.empty()) {
        const int disk_fd = open(load_file.c_str(), O_RDONLY | O_DIRECT);
        if (disk_fd < 0) throw std::runtime_error("failed to open load file: " + load_file +
                                                   " errno=" + std::to_string(errno));
        struct stat disk_stat {};
        if (fstat(disk_fd, &disk_stat) != 0 || uint64_t(disk_stat.st_size) < weight_bytes) {
            close(disk_fd);
            throw std::runtime_error("load file is smaller than the weight size");
        }
        void * disk_buffer = nullptr;
        if (posix_memalign(&disk_buffer, 4096, weight_bytes) != 0) {
            close(disk_fd);
            throw std::runtime_error("failed to allocate aligned disk buffer");
        }
        std::atomic<int> disk_state {0};
        std::atomic<int> disk_error {0};
        auto time_disk_pressure = [&](cl_kernel compute_kernel, const stats & isolated) {
            std::vector<double> under, wall;
            for (int i = 0; i < warmup + iters; ++i) {
                disk_state.store(0, std::memory_order_release);
                disk_error.store(0, std::memory_order_release);
                std::thread loader([&] {
                    while (disk_state.load(std::memory_order_acquire) == 0) std::this_thread::yield();
                    while (disk_state.load(std::memory_order_acquire) == 1) {
                        const ssize_t got = pread(disk_fd, disk_buffer, weight_bytes, 0);
                        if (got != ssize_t(weight_bytes)) {
                            disk_error.store(errno ? errno : EIO, std::memory_order_release);
                            break;
                        }
                    }
                });
                const auto wall_begin = std::chrono::steady_clock::now();
                disk_state.store(1, std::memory_order_release);
                cl_event compute_event = enqueue_2d(queue, compute_kernel, 16, m);
                CL_CHECK(clWaitForEvents(1, &compute_event));
                disk_state.store(2, std::memory_order_release);
                loader.join();
                const double elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wall_begin).count();
                if (disk_error.load(std::memory_order_acquire) != 0) {
                    clReleaseEvent(compute_event);
                    free(disk_buffer); close(disk_fd);
                    throw std::runtime_error("O_DIRECT load failed during pressure run");
                }
                if (i >= warmup) {
                    under.push_back(event_ms(compute_event));
                    wall.push_back(elapsed);
                }
                clReleaseEvent(compute_event);
            }
            (void) isolated;
            return std::pair<stats, stats> {summarize(under), summarize(wall)};
        };
        for (const compute_case & compute : computations) {
            const auto pressure = time_disk_pressure(compute.kernel, *compute.isolated);
            std::printf("DISK_PRESSURE compute_layout=%s gpu_iso_ms=%.6f gpu_under_ms=%.6f "
                        "gpu_slowdown=%.6f wall_med_ms=%.6f\n",
                        compute.name, compute.isolated->med, pressure.first.med,
                        pressure.first.med/compute.isolated->med, pressure.second.med);
        }
        free(disk_buffer);
        close(disk_fd);
    }
    auto overlap_result = [&](const char * layout, const overlap_stats & overlap, const stats & compute) {
        const double ideal = std::max(block_tiled_convert.med, compute.med);
        const double serial = block_tiled_convert.med + compute.med;
        std::printf("OVERLAP transform=block_tiled compute_layout=%s wall_med_ms=%.6f ideal_ms=%.6f "
                    "serial_ms=%.6f extra_vs_serial_ms=%.6f competition=%.6f "
                    "transform_under_ms=%.6f transform_slowdown=%.6f compute_under_ms=%.6f compute_slowdown=%.6f\n",
                    layout, overlap.wall.med, ideal, serial, overlap.wall.med - serial, overlap.wall.med/ideal,
                    overlap.transform.med, overlap.transform.med/block_tiled_convert.med,
                    overlap.compute.med, overlap.compute.med/compute.med);
    };
    overlap_result("raw_aos", overlap_raw, direct_tiled_s);
    overlap_result("block_split", overlap_block, block_tiled_gemv_s);
    overlap_result("fp32_scale_q4", overlap_fp32, fp32_gemv_s);
    overlap_result("expanded_q4", overlap_expanded, expanded_gemv_s);
    for (const chunk_stats & chunk : chunk_results) {
        const double bulk = chunk_results.front().total.med;
        std::printf("CHUNK_PROMOTION chunks=%d total_transform_med_ms=%.6f "
                    "max_chunk_med_ms=%.6f overhead_vs_bulk=%.6f stall_reduction=%.6f\n",
                    chunk.chunks, chunk.total.med, chunk.max_chunk.med,
                    chunk.total.med/bulk, bulk/chunk.max_chunk.med);
    }
    result("first_gemv", "block_split", "fused_persist", block_fused, block_fused_diff);
    std::printf("CHECK layout=block_split after_fused_persist_max_abs_diff=%.9g\n", block_persist_diff);
    result("first_gemv_tiled", "block_split", "fused_persist", block_tiled_fused, block_tiled_fused_diff);
    std::printf("CHECK layout=block_split after_tiled_fused_persist_max_abs_diff=%.9g\n", block_tiled_persist_diff);

    // Paper-friendly total latency at representative reuse horizons.
    for (int h : {1, 4, 16, 64, 128}) {
        std::printf("HORIZON H=%d direct_ms=%.6f direct_tiled_ms=%.6f row_separate_ms=%.6f row_fused_ms=%.6f block_separate_ms=%.6f block_fused_ms=%.6f block_tiled_fused_ms=%.6f\n",
                    h, h*direct_s.med, h*direct_tiled_s.med,
                    row_convert.med + h*row_s.med, row_fused.med + (h-1)*row_s.med,
                    block_tiled_convert.med + h*block_tiled_gemv_s.med,
                    block_fused.med + (h-1)*block_tiled_gemv_s.med,
                    block_tiled_fused.med + (h-1)*block_tiled_gemv_s.med);
    }

    clReleaseProgram(program);
    clReleaseCommandQueue(queue2);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
