#include "ggml-vulkan-common.h"
#include "ggml-vulkan-kv.h"

#include <cinttypes>
#include <cstdlib>

struct vk_kv_push_constants {
    uint32_t rows, head_width, heads, row_stride, head_stride, tile_rows, tiles_per_head, tile_words;
};

static_assert(sizeof(vk_kv_push_constants) == 32, "KV encoder push constants");

struct vk_kv_scene {
    vk_buffer_ref source;
    uint64_t offset {}, size {}, generation {};
    uint32_t capacity {}, rows {}, head_width {}, heads {}, row_stride {}, head_stride {};
    uint32_t tile_rows {}, tile_words {}, tiles {};
    vk_buffer encoded, dirty;
    bool initialized {}, invalidated {};
};

struct vk_kv_context {
    vk_device device;
    PFN_vkCreateAccelerationStructureKHR create_as {};
    PFN_vkDestroyAccelerationStructureKHR destroy_as {};
    PFN_vkGetAccelerationStructureBuildSizesKHR get_sizes {};
    PFN_vkCmdBuildAccelerationStructuresKHR build_as {};
    PFN_vkGetAccelerationStructureDeviceAddressKHR as_address {};
    VkAccelerationStructureKHR bottom {}, top {};
    vk_buffer bounds, instances, bottom_storage, top_storage, scratch;
    uint32_t scratch_alignment {};
    bool built {};
    vk::DescriptorSetLayout descriptor_layout;
    vk::PipelineLayout pipeline_layout;
    vk::ShaderModule module;
    vk::Pipeline pipeline;
    std::vector<vk::DescriptorPool> pools;
    std::vector<vk::DescriptorSet> sets;
    uint32_t set_idx {};
    std::vector<std::unique_ptr<vk_kv_scene>> scenes;

    ~vk_kv_context() {
        if (top)    { destroy_as(device->device, top, nullptr); }
        if (bottom) { destroy_as(device->device, bottom, nullptr); }
        for (auto pool : pools) { device->device.destroyDescriptorPool(pool); }
        device->device.destroyPipeline(pipeline);
        device->device.destroyShaderModule(module);
        device->device.destroyPipelineLayout(pipeline_layout);
        device->device.destroyDescriptorSetLayout(descriptor_layout);
    }
};

static vk_buffer ggml_vk_kv_buffer(vk_device & device, size_t size, vk::BufferUsageFlags usage = {}) {
    return ggml_vk_create_buffer_check(device, size, vk::MemoryPropertyFlagBits::eDeviceLocal, {}, usage);
}

static VkAccelerationStructureGeometryKHR ggml_vk_kv_geometry(const vk_buffer & buffer, bool instances) {
    VkAccelerationStructureGeometryKHR geometry {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = instances ? VK_GEOMETRY_TYPE_INSTANCES_KHR : VK_GEOMETRY_TYPE_AABBS_KHR;
    if (instances) {
        geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geometry.geometry.instances.data.deviceAddress = buffer->bda_addr;
    } else {
        geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
        geometry.geometry.aabbs.data.deviceAddress = buffer->bda_addr;
        geometry.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);
    }
    return geometry;
}

static VkAccelerationStructureBuildGeometryInfoKHR ggml_vk_kv_build_info(const VkAccelerationStructureGeometryKHR & geometry, bool top) {
    VkAccelerationStructureBuildGeometryInfoKHR build {};
    build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build.type = top ? VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR : VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = 1;
    build.pGeometries = &geometry;
    return build;
}

static VkAccelerationStructureKHR ggml_vk_kv_allocate_as(vk_kv_context & state, vk_buffer & storage, const VkAccelerationStructureGeometryKHR & geometry, uint32_t count, bool top, size_t & scratch) {
    auto build = ggml_vk_kv_build_info(geometry, top);
    VkAccelerationStructureBuildSizesInfoKHR sizes {};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    state.get_sizes(state.device->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build, &count, &sizes);
    scratch = std::max(scratch, size_t(sizes.buildScratchSize));
    storage = ggml_vk_kv_buffer(state.device, sizes.accelerationStructureSize, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR);
    VkAccelerationStructureCreateInfoKHR create {};
    create.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    create.type = build.type;
    create.buffer = storage->buffer;
    create.size = sizes.accelerationStructureSize;
    VkAccelerationStructureKHR as {};
    VK_CHECK(vk::Result(state.create_as(state.device->device, &create, nullptr, &as)), "create KV acceleration structure", state.device);
    return as;
}

static std::shared_ptr<vk_kv_context> ggml_vk_kv_context(vk_device & device) {
    auto state = std::make_shared<vk_kv_context>();
    state->device = device;
    state->create_as  = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device->device, "vkCreateAccelerationStructureKHR"));
    state->destroy_as = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device->device, "vkDestroyAccelerationStructureKHR"));
    state->get_sizes  = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device->device, "vkGetAccelerationStructureBuildSizesKHR"));
    state->build_as   = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device->device, "vkCmdBuildAccelerationStructuresKHR"));
    state->as_address = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(device->device, "vkGetAccelerationStructureDeviceAddressKHR"));
    VkPhysicalDeviceAccelerationStructurePropertiesKHR properties {};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 physical {};
    physical.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physical.pNext = &properties;
    vkGetPhysicalDeviceProperties2(device->physical_device, &physical);
    state->scratch_alignment = properties.minAccelerationStructureScratchOffsetAlignment;

    std::array<vk::DescriptorSetLayoutBinding, 4> bindings;
    for (uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i] = { i, i == 3 ? vk::DescriptorType::eAccelerationStructureKHR : vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute };
    }
    state->descriptor_layout = device->device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, bindings));
    const vk::PushConstantRange push(vk::ShaderStageFlagBits::eCompute, 0, sizeof(vk_kv_push_constants));
    state->pipeline_layout = device->device.createPipelineLayout(vk::PipelineLayoutCreateInfo({}, state->descriptor_layout, push));
    state->module = device->device.createShaderModule(vk::ShaderModuleCreateInfo({}, fa_kv_encode_len, reinterpret_cast<const uint32_t *>(fa_kv_encode_data)));
    const vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute, state->module, "main");
    state->pipeline = device->device.createComputePipeline({}, vk::ComputePipelineCreateInfo({}, stage, state->pipeline_layout)).value;

    const auto usage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    state->bounds = ggml_vk_kv_buffer(device, 256 * sizeof(VkAabbPositionsKHR), usage);
    state->instances = ggml_vk_kv_buffer(device, sizeof(VkAccelerationStructureInstanceKHR), usage);
    size_t scratch = 0;
    state->bottom = ggml_vk_kv_allocate_as(*state, state->bottom_storage, ggml_vk_kv_geometry(state->bounds, false), 256, false, scratch);
    state->top = ggml_vk_kv_allocate_as(*state, state->top_storage, ggml_vk_kv_geometry(state->instances, true), 1, true, scratch);
    state->scratch = ggml_vk_kv_buffer(device, scratch + state->scratch_alignment);
    return state;
}

static void ggml_vk_kv_barrier(vk::CommandBuffer cmd, VkPipelineStageFlags source, VkPipelineStageFlags dest, VkAccessFlags writes, VkAccessFlags reads) {
    VkMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = writes;
    barrier.dstAccessMask = reads;
    vkCmdPipelineBarrier(cmd, source, dest, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

static void ggml_vk_kv_build(vk_kv_context & state, vk::CommandBuffer cmd) {
    std::array<VkAabbPositionsKHR, 256> bounds;
    for (uint32_t i = 0; i < bounds.size(); ++i) {
        bounds[i] = { float(i) - 0.25f, -0.25f, 0.0f, float(i) + 0.25f, 0.25f, 0.1f };
    }
    VkAccelerationStructureDeviceAddressInfoKHR address {};
    address.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address.accelerationStructure = state.bottom;
    VkAccelerationStructureInstanceKHR instance {};
    for (uint32_t i = 0; i < 3; ++i) { instance.transform.matrix[i][i] = 1.0f; }
    instance.mask = 255;
    instance.accelerationStructureReference = state.as_address(state.device->device, &address);
    vkCmdUpdateBuffer(cmd, state.bounds->buffer, 0, sizeof(bounds), bounds.data());
    vkCmdUpdateBuffer(cmd, state.instances->buffer, 0, sizeof(instance), &instance);
    ggml_vk_kv_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                       VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    const VkDeviceAddress scratch = ggml_vk_align_size(state.scratch->bda_addr, state.scratch_alignment);
    for (uint32_t level = 0; level < 2; ++level) {
        auto geometry = ggml_vk_kv_geometry(level ? state.instances : state.bounds, level != 0);
        auto build = ggml_vk_kv_build_info(geometry, level != 0);
        build.dstAccelerationStructure = level ? state.top : state.bottom;
        build.scratchData.deviceAddress = scratch;
        VkAccelerationStructureBuildRangeInfoKHR range { level ? 1u : 256u, 0, 0, 0 };
        const auto * ranges = &range;
        state.build_as(cmd, 1, &build, &ranges);
        ggml_vk_kv_barrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           level ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                           level ? VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR :
                                   VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
    }
    state.built = true;
}

static vk::DescriptorSet ggml_vk_kv_set(vk_kv_context & state, const vk_subbuffer & source, const vk_subbuffer & encoded, const vk_subbuffer & dirty) {
    if (state.set_idx == state.sets.size()) {
        const std::array<vk::DescriptorPoolSize, 2> sizes {{ { vk::DescriptorType::eStorageBuffer, 192 }, { vk::DescriptorType::eAccelerationStructureKHR, 64 } }};
        state.pools.push_back(state.device->device.createDescriptorPool(vk::DescriptorPoolCreateInfo({}, 64, sizes)));
        std::array<vk::DescriptorSetLayout, 64> layouts;
        layouts.fill(state.descriptor_layout);
        const auto sets = state.device->device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo(state.pools.back(), layouts));
        state.sets.insert(state.sets.end(), sets.begin(), sets.end());
    }
    const auto set = state.sets[state.set_idx++];
    const std::array<vk::DescriptorBufferInfo, 3> buffers { source, encoded, dirty };
    std::array<vk::WriteDescriptorSet, 4> writes;
    for (uint32_t i = 0; i < buffers.size(); ++i) {
        writes[i] = { set, i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &buffers[i] };
    }
    const vk::AccelerationStructureKHR as(state.top);
    const vk::WriteDescriptorSetAccelerationStructureKHR acceleration(1, &as);
    writes[3] = { set, 3, 0, 1, vk::DescriptorType::eAccelerationStructureKHR };
    writes[3].pNext = &acceleration;
    state.device->device.updateDescriptorSets(writes, {});
    return set;
}

bool ggml_vk_kv_prepare(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * k, vk_kv_codec_params & params) {
    if (!ctx->device->rt_kv_compression || k->type != GGML_TYPE_F16 || k->ne[0] <= 0 || k->ne[0] > 4096 ||
        k->ne[1] <= 0 || k->ne[2] <= 0 || k->ne[3] != 1 || k->nb[0] != 2 || (k->nb[1] | k->nb[2]) % 2 != 0 ||
        !k->buffer || k->buffer->buft != &ctx->device->buffer_type || get_misalign_bytes(ctx, k) != 0) {
        return false;
    }
    const ggml_tensor * root = k;
    while (root->view_src) { root = root->view_src; }
    const uint64_t capacity = vk_tensor_offset(root) + root->view_offs == vk_tensor_offset(k) + k->view_offs ?
                              uint64_t(std::max(root->ne[1], k->ne[1])) : uint64_t(k->ne[1]);
    const uint32_t width = uint32_t(k->ne[0]);
    const uint32_t rows = std::min(16u, 4096u / width);
    const auto & limits = ctx->device->properties.limits;
    if (capacity > UINT32_MAX || k->nb[1] > UINT32_MAX || k->nb[2] > UINT32_MAX ||
        uint64_t(k->ne[2]) > limits.maxComputeWorkGroupCount[1] || limits.maxComputeSharedMemorySize < 33796) {
        return false;
    }
    const uint64_t tiles = CEIL_DIV(capacity, rows);
    if (tiles > limits.maxComputeWorkGroupCount[0]) {
        return false;
    }
    const uint64_t words = 2 + CEIL_DIV(uint64_t(rows) * width, 2);
    const uint64_t bytes_per_head = tiles * words * sizeof(uint32_t);
    const uint64_t bytes_limit = std::min<uint64_t>(limits.maxStorageBufferRange,
                                                  std::min(ctx->device->max_buffer_size, ctx->device->max_memory_allocation_size));
    const uint64_t last = uint64_t(k->ne[1] - 1) * (k->nb[1] / 2) + uint64_t(k->ne[2] - 1) * (k->nb[2] / 2) + width - 1;
    if (last > UINT32_MAX || bytes_limit < 4 * sizeof(uint32_t) ||
        uint64_t(k->ne[2]) > (bytes_limit - 4 * sizeof(uint32_t)) / bytes_per_head) {
        return false;
    }
    const uint64_t bytes = 4 * sizeof(uint32_t) + uint64_t(k->ne[2]) * bytes_per_head;
    if (!ctx->kv_codec) { ctx->kv_codec = ggml_vk_kv_context(ctx->device); }
    auto & state = *ctx->kv_codec;
    auto source = ggml_vk_tensor_subbuffer(ctx, k);
    vk_kv_scene * scene = nullptr;
    for (auto it = state.scenes.begin(); it != state.scenes.end();) {
        const auto buffer = (*it)->source.lock();
        if (!buffer) { it = state.scenes.erase(it); continue; }
        auto & entry = **it++;
        if (buffer == source.buffer && entry.offset == source.offset && entry.capacity == capacity && entry.head_width == width &&
            entry.heads == k->ne[2] && entry.row_stride == k->nb[1] && entry.head_stride == k->nb[2]) {
            scene = &entry;
            break;
        }
    }
    if (!scene) {
        auto entry = std::make_unique<vk_kv_scene>();
        entry->source = source.buffer;
        entry->offset = source.offset;
        entry->size = (capacity - 1) * k->nb[1] + uint64_t(k->ne[2] - 1) * k->nb[2] + uint64_t(width) * 2;
        entry->capacity = uint32_t(capacity);
        entry->head_width = width;
        entry->heads = uint32_t(k->ne[2]);
        entry->row_stride = uint32_t(k->nb[1]);
        entry->head_stride = uint32_t(k->nb[2]);
        entry->tile_rows = rows;
        entry->tile_words = uint32_t(words);
        entry->tiles = uint32_t(tiles);
        entry->encoded = ggml_vk_kv_buffer(ctx->device, size_t(bytes));
        entry->dirty = ggml_vk_kv_buffer(ctx->device, size_t(2 + tiles) * sizeof(uint32_t));
        source.buffer->kv_source = true;
        scene = entry.get();
        state.scenes.push_back(std::move(entry));
    }
    params = { { scene->encoded, 0, scene->encoded->size }, { scene->dirty, 0, scene->dirty->size }, rows, uint32_t(words), uint32_t(tiles) };
    const uint32_t previous_rows = scene->rows;
    scene->rows = uint32_t(k->ne[1]);
    ggml_vk_sync_buffers(ctx, subctx);
    auto cmd = subctx->s->buffer->buf;
    if (!state.built) { ggml_vk_kv_build(state, cmd); }
    const uint64_t generation = source.buffer->kv_generation.load(std::memory_order_relaxed);
    if (!scene->initialized || scene->invalidated || scene->generation != generation) {
        vkCmdFillBuffer(cmd, scene->encoded->buffer, 4 * sizeof(uint32_t), scene->encoded->size - 4 * sizeof(uint32_t), UINT32_MAX);
        if (!scene->initialized) {
            const uint32_t header[] = { rows, uint32_t(tiles), uint32_t(words), 0 };
            const uint32_t dirty[] = { rows, uint32_t(tiles) };
            vkCmdUpdateBuffer(cmd, scene->encoded->buffer, 0, sizeof(header), header);
            vkCmdUpdateBuffer(cmd, scene->dirty->buffer, 0, sizeof(dirty), dirty);
            vkCmdFillBuffer(cmd, scene->dirty->buffer, sizeof(dirty), scene->dirty->size - sizeof(dirty), 0);
        }
        ggml_vk_kv_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        scene->initialized = true;
        scene->invalidated = false;
        scene->generation = generation;
    } else if (scene->rows > previous_rows && previous_rows % rows != 0) {
        for (uint32_t head = 0; head < scene->heads; ++head) {
            const uint64_t seen = 5 + (uint64_t(head) * tiles + previous_rows / rows) * words;
            vkCmdFillBuffer(cmd, scene->encoded->buffer, seen * sizeof(uint32_t), sizeof(uint32_t), UINT32_MAX);
        }
        ggml_vk_kv_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }
    const vk_kv_push_constants push { uint32_t(k->ne[1]), width, uint32_t(k->ne[2]), uint32_t(k->nb[1] / 2),
                                     uint32_t(k->nb[2] / 2), rows, uint32_t(tiles), uint32_t(words) };
    const auto set = ggml_vk_kv_set(state, source, params.encoded, params.dirty);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, state.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, state.pipeline_layout, 0, set, {});
    cmd.pushConstants(state.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmd.dispatch(CEIL_DIV(push.rows, rows), push.heads, 1);
    ggml_vk_sync_buffers(ctx, subctx);
    return true;
}

static vk_subbuffer ggml_vk_kv_changed(ggml_backend_vk_context * ctx, const ggml_tensor * dst, bool incremental) {
    if (!ctx->kv_codec || !dst->buffer || dst->buffer->buft != &ctx->device->buffer_type) { return {}; }
    const auto * buffer = static_cast<const ggml_backend_vk_buffer_context *>(dst->buffer->context);
    const uint64_t offset = vk_tensor_offset(dst) + dst->view_offs;
    vk_subbuffer dirty {};
    for (auto & scene : ctx->kv_codec->scenes) {
        if (scene->source.lock() != buffer->dev_buffer || offset >= scene->offset + scene->size || scene->offset >= offset + ggml_nbytes(dst)) { continue; }
        if (incremental && !dirty.buffer && scene->initialized && offset == scene->offset && dst->type == GGML_TYPE_F16 &&
            ggml_is_contiguous(dst) && dst->ne[1] == scene->capacity && dst->ne[2] == 1 && dst->ne[3] == 1 && dst->nb[1] == scene->row_stride) {
            dirty = { scene->dirty, 0, scene->dirty->size };
        } else {
            scene->invalidated = true;
        }
    }
    return dirty;
}

vk_subbuffer ggml_vk_kv_write(ggml_backend_vk_context * ctx, const ggml_tensor * dst) {
    return ggml_vk_kv_changed(ctx, dst, true);
}

void ggml_vk_kv_invalidate(ggml_backend_vk_context * ctx, const ggml_tensor * dst) {
    ggml_vk_kv_changed(ctx, dst, false);
}

void ggml_vk_kv_reset(ggml_backend_vk_context * ctx) {
    if (ctx->kv_codec) { ctx->kv_codec->set_idx = 0; }
}

void ggml_vk_kv_free(ggml_backend_vk_context * ctx) {
    if (ctx->kv_codec && getenv("GGML_VK_RT_KV_COMPRESS_STATS")) {
        for (auto & scene : ctx->kv_codec->scenes) {
            uint32_t header[4] {};
            ggml_vk_buffer_read(scene->encoded, 0, header, sizeof(header));
            std::vector<uint32_t> encodings(size_t(scene->heads) * scene->tiles);
            ggml_vk_buffer_read_2d(scene->encoded, sizeof(header), encodings.data(),
                                   size_t(scene->tile_words) * sizeof(uint32_t), sizeof(uint32_t), sizeof(uint32_t), encodings.size());
            uint32_t compressed_tiles = 0, first_compressed_row = UINT32_MAX;
            uint64_t raw_bytes = 0, payload_bytes = 0;
            for (size_t i = 0; i < encodings.size(); ++i) {
                const uint32_t row = uint32_t(i % scene->tiles) * scene->tile_rows;
                if (row >= scene->rows) { continue; }
                const uint64_t values = uint64_t(std::min(scene->tile_rows, scene->rows - row)) * scene->head_width;
                raw_bytes += values * sizeof(uint16_t);
                const uint32_t encoding = encodings[i];
                if (encoding == 0 || encoding == UINT32_MAX) {
                    payload_bytes += values * sizeof(uint16_t);
                } else {
                    ++compressed_tiles;
                    first_compressed_row = std::min(first_compressed_row, row);
                    payload_bytes += (2 + CEIL_DIV(encoding >> 8, 2) + CEIL_DIV(values * (encoding & 255u), 32)) * sizeof(uint32_t);
                }
            }
            GGML_LOG_INFO("ggml_vulkan: KV compression rows=%u heads=%u tile_rows=%u compressed_updates=%u"
                          " compressed_tiles=%u first_compressed_row=%u raw_bytes=%" PRIu64 " payload_bytes=%" PRIu64 "\n",
                          scene->rows, scene->heads, header[0], header[3], compressed_tiles, first_compressed_row, raw_bytes, payload_bytes);
        }
    }
    ctx->kv_codec.reset();
}
