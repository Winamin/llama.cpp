#pragma once

#include "ggml-vulkan-types.h"

struct vk_kv_codec_params {
    vk_subbuffer encoded {};
    vk_subbuffer dirty {};
    uint32_t tile_rows {}, tile_words {}, tiles_per_head {};
};

bool ggml_vk_kv_prepare(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * k, vk_kv_codec_params & params);
vk_subbuffer ggml_vk_kv_write(ggml_backend_vk_context * ctx, const ggml_tensor * dst);
void ggml_vk_kv_invalidate(ggml_backend_vk_context * ctx, const ggml_tensor * dst);
void ggml_vk_kv_reset(ggml_backend_vk_context * ctx);
void ggml_vk_kv_free(ggml_backend_vk_context * ctx);
