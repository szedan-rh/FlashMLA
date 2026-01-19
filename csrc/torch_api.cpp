// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Torch library registration for FlashMLA

#include <Python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "pytorch_shim.h"
#include "api/common.h"
#include "api/dense_decode.h"
#include "api/sparse_decode.h"
#include "api/sparse_fwd.h"


std::vector<at::Tensor>
get_mla_decoding_metadata(
    at::Tensor &seqlens_k,
    const int num_q_tokens_per_head_k,
    const int h_k,
    const std::optional<int> h_q,
    const bool is_fp8_kvcache,
    const std::optional<int> topk
) {
    TORCH_CHECK(seqlens_k.is_cuda(), "seqlens_k must be on CUDA device");
    TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    TORCH_CHECK(seqlens_k.is_contiguous(), "seqlens_k must be contiguous");
    
    const int batch_size = seqlens_k.size(0);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    
    at::cuda::CUDAGuard device_guard{(char)seqlens_k.get_device()};
    auto opts = seqlens_k.options();
    
    Arch arch = Arch();
    TORCH_CHECK(h_k > 0, "num_heads_k must be positive");
    int num_heads_q = h_q.value_or(h_k * num_q_tokens_per_head_k);
    TORCH_CHECK(num_heads_q > 0, "num_heads_q must be positive");
    const int heads_ratio = std::max(1, num_heads_q / h_k);
    int s_q = std::max(1, num_q_tokens_per_head_k / heads_ratio);
    
    int num_sm_parts;
    int fixed_overhead_num_blocks = 5;
    int block_size_n = 64;
    
    if (topk.has_value()) {
        // Sparse FP8 decode - formula differs by architecture
        // Must match the formulas in api/sparse_decode.h
        if (arch.is_sm100f()) {
            // SM100 head64/head64x2 use: num_sms / s_q
            // SM100 head128 uses: num_sms / s_q / 2
            // Use larger buffer (num_sms / s_q) to be safe for both
            num_sm_parts = std::max(arch.num_sms / s_q, 1);
        } else {
            // SM90 uses: num_sms / s_q / (h_q/64)
            const int heads_per_64 = std::max(1, num_heads_q / 64);
            num_sm_parts = std::max(arch.num_sms / s_q / heads_per_64, 1);
        }
    } else {
        num_sm_parts = std::max(arch.num_sms / h_k / cutlass::ceil_div(s_q * num_heads_q / h_k, 64), 1);
    }
    
    at::Tensor tile_scheduler_metadata = torch::empty(
        {num_sm_parts, DecodingSchedMetaSize / sizeof(int)}, opts.dtype(torch::kInt32));
    at::Tensor num_splits = torch::empty({batch_size + 1}, opts.dtype(torch::kInt32));
    
    GetDecodeSchedMetaParams params = {
        batch_size, s_q,
        block_size_n,
        fixed_overhead_num_blocks,
        topk.value_or(-1), -1,
        nullptr, nullptr,
        seqlens_k.data_ptr<int>(),
        (DecodingSchedMeta*)tile_scheduler_metadata.data_ptr(),
        num_splits.data_ptr<int>(),
        num_sm_parts,
        at::cuda::getCurrentCUDAStream().stream()
    };
    
    smxx::decode::run_get_decoding_sched_meta_kernel(params);
    
    return {tile_scheduler_metadata, num_splits};
}


std::vector<at::Tensor>
fwd_kvcache_mla(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits,                // batch_size + 1
    const bool &is_fp8,
    const std::optional<at::Tensor> &indices     // None, or batch_size x seqlen_q x topk
) {
    std::optional<at::Tensor> tile_scheduler_metadata_opt = tile_scheduler_metadata;
    std::optional<at::Tensor> num_splits_opt = num_splits;
    
    if (indices.has_value()) {
        auto result = sparse_attn_decode_interface(
            q,                         // q
            kcache,                    // kv
            indices.value(),           // indices
            std::nullopt,              // topk_length
            std::nullopt,              // attn_sink
            tile_scheduler_metadata_opt,
            num_splits_opt,
            std::nullopt,              // extra_kv
            std::nullopt,              // extra_indices
            std::nullopt,              // extra_topk_length
            head_size_v,
            softmax_scale
        );
        return {std::get<0>(result), std::get<1>(result)};
    }

    auto result = dense_attn_decode_interface(
        q, kcache, head_size_v, seqlens_k, block_table,
        softmax_scale, is_causal,
        tile_scheduler_metadata_opt, num_splits_opt
    );

    return {std::get<0>(result), std::get<1>(result)};
}

std::vector<at::Tensor>
sparse_prefill_fwd(
    const at::Tensor &q,                         // s_q x h_q x d_qk
    const at::Tensor &kv,                        // s_kv x h_kv x d_qk
    const at::Tensor &indices,                   // s_q x h_kv x topk
    float sm_scale,
    int d_v
) {
    return sparse_attn_prefill_interface(q, kv, indices, sm_scale, d_v, std::nullopt, std::nullopt);
}


TORCH_LIBRARY(_flashmla_C, m) {
    m.def("get_mla_decoding_metadata", make_pytorch_shim(&get_mla_decoding_metadata));
    m.impl("get_mla_decoding_metadata", torch::kCUDA, make_pytorch_shim(&get_mla_decoding_metadata));

    m.def("fwd_kvcache_mla", make_pytorch_shim(&fwd_kvcache_mla));
    m.impl("fwd_kvcache_mla", torch::kCUDA, make_pytorch_shim(&fwd_kvcache_mla));

    m.def("sparse_prefill_fwd", make_pytorch_shim(&sparse_prefill_fwd));
    m.impl("sparse_prefill_fwd", torch::kCUDA, make_pytorch_shim(&sparse_prefill_fwd));
}

PyMODINIT_FUNC PyInit__flashmla_C() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_flashmla_C", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
