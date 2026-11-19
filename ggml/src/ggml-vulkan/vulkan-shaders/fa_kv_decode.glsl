layout(binding = 8) readonly buffer KC {uint32_t data_kv_encoded[];};
layout(binding = 8) readonly buffer KC16 {float16_t data_kv_values[];};

bool fa_kv_block_encoded(uint first_row, uint end_row) {
    if (first_row >= end_row) {
        return false;
    }
    const uint tile_rows = data_kv_encoded[0];
    const uint tiles_per_head = data_kv_encoded[1];
    const uint tile_words = data_kv_encoded[2];
    const uint head_base = 4u + ik2 * tiles_per_head * tile_words;
    const uint last_tile = (end_row - 1u) / tile_rows;
    for (uint tile = first_row / tile_rows; tile <= last_tile; ++tile) {
        if (data_kv_encoded[head_base + tile * tile_words] != 0) {
            return true;
        }
    }
    return false;
}

float16_t fa_kv_decode_k(uint row, uint axis) {
    if (row >= KV || axis >= HSK) {
        return float16_t(0.0);
    }
    const uint tile_rows = data_kv_encoded[0];
    const uint tiles_per_head = data_kv_encoded[1];
    const uint tile_words = data_kv_encoded[2];
    const uint base = 4u + (ik2 * tiles_per_head + row / tile_rows) * tile_words;
    const uint encoding = data_kv_encoded[base];
    if (encoding == 0) {
        return data_k16[g_k_off_elem + row * k_stride + axis];
    }
    const uint count = encoding >> 8;
    const uint bits = encoding & 255u;
    uint rank = 0;
    if (bits != 0) {
        const uint bit = ((row % tile_rows) * HSK + axis) * bits;
        const uint word = base + 2u + (count + 1u) / 2u + bit / 32u;
        const uint shift = bit & 31u;
        rank = data_kv_encoded[word] >> shift;
        if (shift + bits > 32u) {
            rank |= data_kv_encoded[word + 1u] << (32u - shift);
        }
        rank &= (1u << bits) - 1u;
    }
    return data_kv_values[(base + 2u) * 2u + rank];
}
