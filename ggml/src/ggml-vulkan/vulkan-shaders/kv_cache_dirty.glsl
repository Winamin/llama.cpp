#if defined(RT_KV_BINDING)
layout (binding = RT_KV_BINDING) buffer KV_DIRTY {
    uint kv_tile_rows;
    uint kv_tiles;
    uint kv_generation[];
};

void rt_kv_write(uint row) {
    const uint tile = row / kv_tile_rows;
    if (tile < kv_tiles) {
        atomicAdd(kv_generation[tile], 1);
    }
}
#endif
