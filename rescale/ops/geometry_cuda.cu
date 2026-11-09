#include <torch/extension.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <cmath>

namespace {
constexpr int threads = 256;

bool supported_float(torch::Tensor value) {
    return value.scalar_type() == torch::kFloat32 || value.scalar_type() == torch::kFloat16 ||
           value.scalar_type() == torch::kBFloat16;
}

void spatial_size(const std::vector<int64_t>& size) {
    TORCH_CHECK(size.size() == 2 && size[0] > 0 && size[1] > 0 && size[0] <= INT_MAX && size[1] <= INT_MAX,
                "Spatial size must have two positive int32 dimensions");
}

void metadata(torch::Tensor boxes, torch::Tensor sizes, torch::Tensor source) {
    TORCH_CHECK(boxes.is_cuda() && boxes.device() == source.device() && boxes.scalar_type() == torch::kFloat32 &&
                boxes.is_contiguous() && boxes.dim() == 2 && boxes.size(0) == source.size(0) && boxes.size(1) == 4,
                "Boxes must be contiguous CUDA FP32 [B,4] on the source device");
    TORCH_CHECK(sizes.is_cuda() && sizes.device() == source.device() && sizes.scalar_type() == torch::kInt64 &&
                sizes.is_contiguous() && sizes.dim() == 2 && sizes.size(0) == source.size(0) && sizes.size(1) == 2,
                "Sizes must be contiguous CUDA int64 [B,2] on the source device");
}

__device__ int64_t label_at(const void* source, int64_t index, int dtype) {
    switch (static_cast<at::ScalarType>(dtype)) {
        case at::ScalarType::Byte: return static_cast<const uint8_t*>(source)[index];
        case at::ScalarType::Char: return static_cast<const int8_t*>(source)[index];
        case at::ScalarType::Short: return static_cast<const int16_t*>(source)[index];
        case at::ScalarType::Int: return static_cast<const int32_t*>(source)[index];
        case at::ScalarType::Long: return static_cast<const int64_t*>(source)[index];
        case at::ScalarType::Bool: return static_cast<const bool*>(source)[index];
        default: return 0;
    }
}

__device__ float float_at(const void* source, int64_t index, int dtype) {
    if (dtype == static_cast<int>(at::ScalarType::Float)) return static_cast<const float*>(source)[index];
    if (dtype == static_cast<int>(at::ScalarType::Half)) return static_cast<float>(static_cast<const at::Half*>(source)[index]);
    return static_cast<float>(static_cast<const at::BFloat16*>(source)[index]);
}

__device__ float probability(const void* source, int dtype, int64_t offset, int height, int width, float y, float x) {
    x = fminf(fmaxf(x, 0.f), width - 1.f); y = fminf(fmaxf(y, 0.f), height - 1.f);
    int x0 = static_cast<int>(floorf(x)), y0 = static_cast<int>(floorf(y));
    int x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
    float wx = x - x0, wy = y - y0;
    return float_at(source, offset + static_cast<int64_t>(y0) * width + x0, dtype) * ((1 - wy) * (1 - wx))
         + float_at(source, offset + static_cast<int64_t>(y0) * width + x1, dtype) * ((1 - wy) * wx)
         + float_at(source, offset + static_cast<int64_t>(y1) * width + x0, dtype) * (wy * (1 - wx))
         + float_at(source, offset + static_cast<int64_t>(y1) * width + x1, dtype) * (wy * wx);
}

template <typename scalar_t>
__global__ void resized_crop(const void* image, int image_dtype, const void* mask, int mask_dtype,
    const float* boxes, const int64_t* indices, const float* black,
    scalar_t* output, int64_t* targets, bool* valid_output, int64_t count,
    int source_batch, int channels, int height, int width, int out_h, int out_w,
    float inv_out_h, float inv_out_w) {
    int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= count) return;
    int xout = p % out_w, yout = (p / out_w) % out_h;
    int64_t roi = p / (static_cast<int64_t>(out_h) * out_w);
    int64_t batch = indices[roi];
    const float* box = boxes + roi * 4;
    bool ok = batch >= 0 && batch < source_batch &&
              isfinite(box[0]) && isfinite(box[1]) && isfinite(box[2]) && isfinite(box[3]) &&
              box[2] > box[0] && box[3] > box[1];
    float x = 0.f, y = 0.f;
    if (ok) {
        x = box[0] + (xout + .5f) * inv_out_w * (box[2] - box[0]) - .5f;
        y = box[1] + (yout + .5f) * inv_out_h * (box[3] - box[1]) - .5f;
        ok = x >= -.5f && x < width - .5f && y >= -.5f && y < height - .5f;
        if (ok) {
            x = fminf(fmaxf(x, 0.f), width - 1.f);
            y = fminf(fmaxf(y, 0.f), height - 1.f);
        }
    }
    if (valid_output) valid_output[p] = ok;
    for (int channel = 0; channel < channels; ++channel) {
        float value = black[channel];
        if (ok) {
            int64_t offset = (batch * channels + channel) * height * width;
            value = probability(image, image_dtype, offset, height, width, y, x);
        }
        output[(roi * channels + channel) * out_h * out_w + yout * out_w + xout] = static_cast<scalar_t>(value);
    }
    if (targets) {
        int64_t target = 0;
        if (ok) {
            int ix = static_cast<int>(floorf(x + .5f)), iy = static_cast<int>(floorf(y + .5f));
            target = label_at(mask, batch * height * width + static_cast<int64_t>(iy) * width + ix, mask_dtype);
        }
        targets[p] = target;
    }
}

template <typename scalar_t>
__global__ void paste_single(const void* base, int base_dtype, const void* crop, int crop_dtype,
    const float* boxes, const int64_t* sizes, scalar_t* output, bool* coverage, int64_t count,
    int channels, int canvas_h, int canvas_w, int base_h, int base_w, int crop_h, int crop_w) {
    int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= count) return;
    int x = p % canvas_w, y = (p / canvas_w) % canvas_h;
    int64_t batch = p / (static_cast<int64_t>(canvas_h) * canvas_w);
    int64_t actual_h = sizes[batch * 2], actual_w = sizes[batch * 2 + 1];
    bool native_valid = actual_h > 0 && actual_w > 0 && actual_h <= canvas_h && actual_w <= canvas_w && x < actual_w && y < actual_h;
    const float* box = boxes + batch * 4;
    float bx = 0.f, by = 0.f;
    bool covered = false;
    if (native_valid) {
        bx = (x + .5f) * (static_cast<float>(base_w) / static_cast<float>(actual_w));
        by = (y + .5f) * (static_cast<float>(base_h) / static_cast<float>(actual_h));
        covered = isfinite(box[0]) && isfinite(box[1]) && isfinite(box[2]) && isfinite(box[3]) &&
                  box[2] > box[0] && box[3] > box[1] && bx >= box[0] && bx < box[2] && by >= box[1] && by < box[3];
    }
    coverage[p] = covered;
    for (int channel = 0; channel < channels; ++channel) {
        float value = 0.f;
        if (native_valid) {
            if (covered) {
                // Interpolate probabilities, not labels or logits, before a
                // caller applies its final threshold or argmax.
                float cx = (bx - box[0]) / (box[2] - box[0]) * crop_w - .5f;
                float cy = (by - box[1]) / (box[3] - box[1]) * crop_h - .5f;
                value = probability(crop, crop_dtype, (batch * channels + channel) * crop_h * crop_w, crop_h, crop_w, cy, cx);
            } else value = probability(base, base_dtype, (batch * channels + channel) * base_h * base_w, base_h, base_w, by - .5f, bx - .5f);
        }
        output[(batch * channels + channel) * canvas_h * canvas_w + y * canvas_w + x] = static_cast<scalar_t>(value);
    }
}
__global__ void init_components(const int* labels, int* parent, int* stats,
                                int count, int height, int width) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= count) return;
  parent[p] = labels[p] > 0 ? p : -1;
  int64_t offset = static_cast<int64_t>(p) * 5;
  stats[offset] = width;
  stats[offset + 1] = height;
  stats[offset + 2] = stats[offset + 3] = stats[offset + 4] = 0;
}

__device__ int find_root(int* parent, int node) {
  int next = atomicAdd(parent + node, 0);
  while (next != node) {
    node = next;
    next = atomicAdd(parent + node, 0);
  }
  return node;
}

// join_components has completed on this stream: no parent writes remain.
// Every pixel needs its root only once, so a separate path-compression buffer
// and launch would add traffic without removing any later root queries.
__device__ int find_root_readonly(const int* parent, int node) {
  int next = parent[node];
  while (next != node) {
    node = next;
    next = parent[node];
  }
  return node;
}

// Parent IDs strictly decrease. Each successful atomic link joins components;
// retries follow concurrently changed roots until the two sets really coincide.
// There is no fixed propagation count or dependence on component diameter.
__device__ void unite(int* parent, int a, int b) {
  while (true) {
    a = find_root(parent, a);
    b = find_root(parent, b);
    if (a == b) return;
    int high = max(a, b), low = min(a, b);
    int previous = atomicMin(parent + high, low);
    if (previous == high) return;
    a = previous;
    b = low;
  }
}

__global__ void join_components(const int* labels, int* parent, int count, int height, int width) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= count || labels[p] <= 0) return;
  int x = p % width, y = (p / width) % height;
  int category = labels[p];
  if (x > 0 && labels[p - 1] == category) unite(parent, p, p - 1);
  if (y > 0) {
    if (labels[p - width] == category) unite(parent, p, p - width);
    if (x > 0 && labels[p - width - 1] == category) unite(parent, p, p - width - 1);
    if (x + 1 < width && labels[p - width + 1] == category) unite(parent, p, p - width + 1);
  }
}

__global__ void reduce_components(const int* labels, const int* parent, int* stats,
                                  int count, int height, int width) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= count || labels[p] <= 0) return;
  int root = find_root_readonly(parent, p);
  int x = p % width, y = (p / width) % height;
  int xmin = x, ymin = y, xmax = x + 1, ymax = y + 1, area = 1;
#if __CUDA_ARCH__ >= 800
  // Hardware integer warp reductions (Ampere+) aggregate only identical roots.
  // Background/tail lanes are excluded. Older GPUs keep the exact atomic path.
  unsigned group = __match_any_sync(__activemask(), root);
  xmin = __reduce_min_sync(group, xmin);
  ymin = __reduce_min_sync(group, ymin);
  xmax = __reduce_max_sync(group, xmax);
  ymax = __reduce_max_sync(group, ymax);
  area = __popc(group);
  if ((threadIdx.x & 31) != __ffs(group) - 1) return;
#endif
  int64_t offset = static_cast<int64_t>(root) * 5;
  atomicMin(stats + offset, xmin);
  atomicMin(stats + offset + 1, ymin);
  atomicMax(stats + offset + 2, xmax);
  atomicMax(stats + offset + 3, ymax);
  atomicAdd(stats + offset + 4, area);
}

__global__ void pack_components(const int* stats, const int* labels, const int64_t* roots,
                                float* boxes, int64_t* batches,
                                int64_t count, int64_t pixel_count, int height, int width,
                                float margin_ratio) {
  int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= count) return;
  int64_t root = roots[p];
  // Public bindings must never turn malformed metadata into an out-of-bounds read.
  if (root < 0 || root >= pixel_count) {
    boxes[p * 4] = boxes[p * 4 + 1] = boxes[p * 4 + 2] = boxes[p * 4 + 3] = 0.f;
    batches[p] = -1;
    return;
  }
  const int* stat = stats + root * 5;
  float x0 = static_cast<float>(stat[0]), y0 = static_cast<float>(stat[1]);
  float x1 = static_cast<float>(stat[2]), y1 = static_cast<float>(stat[3]);
  float dx = (x1 - x0) * margin_ratio, dy = (y1 - y0) * margin_ratio;
  boxes[p * 4] = fmaxf(x0 - dx, 0.f);
  boxes[p * 4 + 1] = fmaxf(y0 - dy, 0.f);
  boxes[p * 4 + 2] = fminf(x1 + dx, static_cast<float>(width));
  boxes[p * 4 + 3] = fminf(y1 + dy, static_cast<float>(height));
  batches[p] = root / (static_cast<int64_t>(height) * width);
}

// Each ROI visits its native rectangle, instead of the full padded image.
// Atomics resolve expanded-box overlaps; FP32 counts are written once per pixel.
__global__ void accumulate_rois(const void* crops, int dtype, const float* boxes,
    const int64_t* indices, const int64_t* sizes, float* sums, float* counts,
    int batch_size, int channels, int canvas_h, int canvas_w, int base_h, int base_w,
    int crop_h, int crop_w) {
    int roi = blockIdx.x / 8;
    int64_t batch = indices[roi];
    if (batch < 0 || batch >= batch_size) return;
    int64_t native_h = sizes[batch * 2], native_w = sizes[batch * 2 + 1];
    const float* box = boxes + static_cast<int64_t>(roi) * 4;
    if (native_h <= 0 || native_w <= 0 || native_h > canvas_h || native_w > canvas_w ||
        !isfinite(box[0]) || !isfinite(box[1]) || !isfinite(box[2]) || !isfinite(box[3]) ||
        box[2] <= box[0] || box[3] <= box[1]) return;
    float scale_x = static_cast<float>(native_w) / base_w;
    float scale_y = static_cast<float>(native_h) / base_h;
    int x_start = static_cast<int>(fminf(fmaxf(floorf(box[0] * scale_x), 0.f), static_cast<float>(native_w)));
    int y_start = static_cast<int>(fminf(fmaxf(floorf(box[1] * scale_y), 0.f), static_cast<float>(native_h)));
    int x_end = static_cast<int>(fminf(fmaxf(ceilf(box[2] * scale_x), 0.f), static_cast<float>(native_w)));
    int y_end = static_cast<int>(fminf(fmaxf(ceilf(box[3] * scale_y), 0.f), static_cast<float>(native_h)));
    int rect_w = max(0, x_end - x_start), rect_h = max(0, y_end - y_start);
    int64_t pixels = static_cast<int64_t>(rect_w) * rect_h;
    for (int64_t position = (blockIdx.x % 8) * blockDim.x + threadIdx.x;
         position < pixels * channels; position += blockDim.x * 8) {
        int channel = position / pixels;
        int64_t local = position % pixels;
        int x = x_start + local % rect_w, y = y_start + local / rect_w;
        float bx = (x + .5f) * (static_cast<float>(base_w) / static_cast<float>(native_w));
        float by = (y + .5f) * (static_cast<float>(base_h) / static_cast<float>(native_h));
        if (bx < box[0] || bx >= box[2] || by < box[1] || by >= box[3]) continue;
        float cx = (bx - box[0]) / (box[2] - box[0]) * crop_w - .5f;
        float cy = (by - box[1]) / (box[3] - box[1]) * crop_h - .5f;
        int64_t source_offset = (static_cast<int64_t>(roi) * channels + channel) * crop_h * crop_w;
        float value = probability(crops, dtype, source_offset, crop_h, crop_w, cy, cx);
        int64_t pixel = static_cast<int64_t>(y) * canvas_w + x;
        atomicAdd(sums + (batch * channels + channel) * canvas_h * canvas_w + pixel, value);
        if (channel == 0) atomicAdd(counts + batch * canvas_h * canvas_w + pixel, 1.f);
    }
}
// One thread owns a native pixel across all classes. Votes are read once;
// normalization, optional fallback and padding writes share a single pass.
__global__ void finalize_rois(float* sums, const float* counts, const int64_t* sizes,
    const void* base, int base_dtype, bool* coverage, int64_t pixels,
    int channels, int canvas_h, int canvas_w) {
    int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    int64_t plane_size = static_cast<int64_t>(canvas_h) * canvas_w;
    int64_t batch = p / plane_size, pixel = p % plane_size;
    int x = pixel % canvas_w, y = pixel / canvas_w;
    int64_t native_h = sizes[batch * 2], native_w = sizes[batch * 2 + 1];
    bool native_valid = native_h > 0 && native_w > 0 && native_h <= canvas_h &&
                        native_w <= canvas_w && x < native_w && y < native_h;
    float votes = counts[p];
    bool covered = native_valid && votes > 0.f && isfinite(votes);
    if (coverage) coverage[p] = covered;
    for (int channel = 0; channel < channels; ++channel) {
        int64_t index = (batch * channels + channel) * plane_size + pixel;
        float value = 0.f;
        if (covered) value = sums[index] / votes;
        else if (native_valid) {
            if (base) value = float_at(base, index, base_dtype);
            else if (channels > 1 && channel == 0) value = 1.f;
        }
        sums[index] = value;
    }
}
} // namespace

std::vector<torch::Tensor> crop_resized_cuda(torch::Tensor image, torch::Tensor mask,
    torch::Tensor boxes, torch::Tensor indices, torch::Tensor black,
    std::vector<int64_t> output_size, torch::Tensor output_like, bool return_valid) {
    TORCH_CHECK(image.is_cuda() && image.is_contiguous() && supported_float(image) &&
                image.dim() == 4 && image.size(0) > 0 && image.size(1) > 0 &&
                image.size(2) > 0 && image.size(3) > 0,
                "Resized image must be nonempty contiguous CUDA BCHW FP32/FP16/BF16");
    spatial_size(output_size);
    TORCH_CHECK(image.size(0) <= INT_MAX && image.size(1) <= INT_MAX &&
                image.size(2) <= INT_MAX && image.size(3) <= INT_MAX,
                "Resized source dimensions exceed int32 indexing range");
    for (const auto& value : {mask, boxes, indices, black, output_like})
        TORCH_CHECK(value.is_cuda() && value.device() == image.device() && value.is_contiguous(),
                    "All resized crop tensors must be contiguous on one CUDA device");
    TORCH_CHECK(boxes.scalar_type() == torch::kFloat32 && boxes.dim() == 2 && boxes.size(1) == 4,
                "Boxes must be FP32 [N,4]");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64 && indices.dim() == 1 &&
                indices.numel() == boxes.size(0), "Indices must be int64 [N]");
    TORCH_CHECK(black.scalar_type() == torch::kFloat32 && black.dim() == 1 &&
                black.numel() == image.size(1), "Normalized black must be FP32 [C]");
    TORCH_CHECK(supported_float(output_like), "Output dtype must be FP32/FP16/BF16");
    bool has_mask = mask.numel() != 0;
    TORCH_CHECK(!has_mask || (mask.dim() == 3 && mask.size(0) == image.size(0) &&
                mask.size(1) == image.size(2) && mask.size(2) == image.size(3) &&
                at::isIntegralType(mask.scalar_type(), true)), "Mask must be integral [B,H,W]");
    c10::cuda::CUDAGuard guard(image.device());
    auto output = torch::empty({boxes.size(0), image.size(1), output_size[0], output_size[1]}, output_like.options());
    auto targets = torch::empty(has_mask ? std::vector<int64_t>{boxes.size(0), output_size[0], output_size[1]}
        : std::vector<int64_t>{0}, image.options().dtype(torch::kInt64));
    auto valid = torch::empty(return_valid ? std::vector<int64_t>{boxes.size(0), 1, output_size[0], output_size[1]}
        : std::vector<int64_t>{0}, image.options().dtype(torch::kBool));
    int64_t count = boxes.size(0) * output_size[0] * output_size[1];
    if (!count) return {output, targets, valid};
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, output.scalar_type(), "recursive_resized_crop", [&] {
        resized_crop<scalar_t><<<(count + threads - 1) / threads, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            image.data_ptr(), static_cast<int>(image.scalar_type()), has_mask ? mask.data_ptr() : nullptr,
            static_cast<int>(mask.scalar_type()), boxes.data_ptr<float>(), indices.data_ptr<int64_t>(),
            black.data_ptr<float>(), output.data_ptr<scalar_t>(), has_mask ? targets.data_ptr<int64_t>() : nullptr,
            return_valid ? valid.data_ptr<bool>() : nullptr, count, image.size(0), image.size(1),
            image.size(2), image.size(3), output_size[0], output_size[1],
            static_cast<float>(1.0 / output_size[0]), static_cast<float>(1.0 / output_size[1]));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, targets, valid};
}

std::vector<torch::Tensor> paste_probabilities_cuda(torch::Tensor base, torch::Tensor crop,
    torch::Tensor boxes, torch::Tensor sizes, std::vector<int64_t> canvas_size) {
    TORCH_CHECK(base.is_cuda() && supported_float(base) && base.is_contiguous() && base.dim() == 4 &&
                base.size(0) > 0 && base.size(1) > 0 && base.size(2) > 0 && base.size(3) > 0, "Base probabilities must be nonempty contiguous CUDA BCHW FP32/FP16/BF16");
    TORCH_CHECK(crop.is_cuda() && crop.device() == base.device() && supported_float(crop) && crop.is_contiguous() && crop.dim() == 4 &&
                crop.size(0) == base.size(0) && crop.size(1) == base.size(1) && crop.size(2) > 0 && crop.size(3) > 0, "Crop probabilities must match base batch/channels/device");
    spatial_size(canvas_size); metadata(boxes, sizes, base);
    c10::cuda::CUDAGuard guard(base.device());
    auto output = torch::empty({base.size(0), base.size(1), canvas_size[0], canvas_size[1]}, base.options());
    auto covered = torch::empty({base.size(0), 1, canvas_size[0], canvas_size[1]}, base.options().dtype(torch::kBool));
    int64_t count = base.size(0) * canvas_size[0] * canvas_size[1];
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, output.scalar_type(), "recursive_probability_paste", [&] {
        paste_single<scalar_t><<<(count + threads - 1) / threads, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            base.data_ptr(), static_cast<int>(base.scalar_type()), crop.data_ptr(), static_cast<int>(crop.scalar_type()), boxes.data_ptr<float>(), sizes.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(), covered.data_ptr<bool>(), count, base.size(1), canvas_size[0], canvas_size[1], base.size(2), base.size(3), crop.size(2), crop.size(3));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, covered};
}

torch::Tensor component_stats_cuda(torch::Tensor labels) {
  TORCH_CHECK(labels.is_cuda() && labels.is_contiguous() && labels.scalar_type() == torch::kInt32 &&
              labels.dim() == 3, "Labels must be contiguous CUDA int32 [B,H,W]");
  TORCH_CHECK(labels.numel() <= INT_MAX, "Component label indexing exceeds int32 range");
  c10::cuda::CUDAGuard guard(labels.device());
  int count = labels.numel(), height = labels.size(1), width = labels.size(2);
  auto parent = torch::empty({count}, labels.options());
  auto stats = torch::empty({count, 5}, labels.options());
  if (!count) return stats;
  auto stream = c10::cuda::getCurrentCUDAStream();
  int blocks = static_cast<int>((static_cast<int64_t>(count) + threads - 1) / threads);
  init_components<<<blocks, threads, 0, stream>>>(labels.data_ptr<int>(), parent.data_ptr<int>(),
                                                 stats.data_ptr<int>(), count, height, width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  join_components<<<blocks, threads, 0, stream>>>(labels.data_ptr<int>(), parent.data_ptr<int>(), count, height, width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  reduce_components<<<blocks, threads, 0, stream>>>(labels.data_ptr<int>(), parent.data_ptr<int>(),
                                                   stats.data_ptr<int>(), count, height, width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return stats;
}

std::vector<torch::Tensor> component_boxes_cuda(torch::Tensor stats, torch::Tensor labels,
                                               torch::Tensor roots, double margin_ratio) {
  TORCH_CHECK(labels.is_cuda() && stats.is_cuda() && roots.is_cuda() &&
              labels.device() == stats.device() && labels.device() == roots.device(), "Component devices differ");
  TORCH_CHECK(labels.scalar_type() == torch::kInt32 && stats.scalar_type() == torch::kInt32 &&
              roots.scalar_type() == torch::kInt64, "Invalid component metadata dtypes");
  TORCH_CHECK(labels.is_contiguous() && stats.is_contiguous() && roots.is_contiguous(),
              "Component metadata must be contiguous");
  TORCH_CHECK(labels.dim() == 3 && stats.dim() == 2 && stats.size(0) == labels.numel() &&
              stats.size(1) == 5 && roots.dim() == 1, "Invalid component metadata shapes");
  TORCH_CHECK(std::isfinite(margin_ratio) && margin_ratio >= 0., "Component margin must be finite and nonnegative");
  TORCH_CHECK(labels.size(1) > 0 && labels.size(2) > 0, "Component spatial dimensions must be positive");
  c10::cuda::CUDAGuard guard(labels.device());
  auto boxes = torch::empty({roots.numel(), 4}, labels.options().dtype(torch::kFloat32));
  auto batches = torch::empty({roots.numel()}, roots.options());
  if (roots.numel()) {
    int blocks = static_cast<int>((roots.numel() + threads - 1) / threads);
    pack_components<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        stats.data_ptr<int>(), labels.data_ptr<int>(), roots.data_ptr<int64_t>(),
        boxes.data_ptr<float>(), batches.data_ptr<int64_t>(),
        roots.numel(), labels.numel(), labels.size(1), labels.size(2), static_cast<float>(margin_ratio));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {boxes, batches};
}

void accumulate_crop_probabilities_cuda(torch::Tensor crops, torch::Tensor boxes,
    torch::Tensor indices, std::vector<int64_t> image_size, torch::Tensor sizes,
    torch::Tensor sums, torch::Tensor counts) {
    spatial_size(image_size);
    TORCH_CHECK(crops.is_cuda() && crops.is_contiguous() && crops.dim() == 4 && supported_float(crops) &&
                crops.size(1) > 0 && crops.size(2) > 0 && crops.size(3) > 0, "Crops must be CUDA [N,C,H,W] FP32/FP16/BF16");
    for (const auto& value : {boxes, indices, sizes, sums, counts})
        TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.device() == crops.device(), "Accumulation tensors must be contiguous on one CUDA device");
    TORCH_CHECK(boxes.scalar_type() == torch::kFloat32 && boxes.dim() == 2 && boxes.size(0) == crops.size(0) && boxes.size(1) == 4,
                "Boxes must be FP32 [N,4]");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64 && indices.dim() == 1 && indices.numel() == crops.size(0), "Indices must be int64 [N]");
    TORCH_CHECK(sums.scalar_type() == torch::kFloat32 && sums.dim() == 4 && sums.size(1) == crops.size(1) &&
                sums.size(0) > 0 && sums.size(2) > 0 && sums.size(3) > 0, "Sums must be FP32 [B,C,Hpad,Wpad]");
    TORCH_CHECK(counts.scalar_type() == torch::kFloat32 && counts.dim() == 4 && counts.size(0) == sums.size(0) && counts.size(1) == 1 &&
                counts.size(2) == sums.size(2) && counts.size(3) == sums.size(3), "Counts must be FP32 [B,1,Hpad,Wpad]");
    TORCH_CHECK(sizes.scalar_type() == torch::kInt64 && sizes.dim() == 2 && sizes.size(0) == sums.size(0) && sizes.size(1) == 2, "Sizes must be int64 [B,2]");
    c10::cuda::CUDAGuard guard(crops.device());
    if (!crops.size(0)) return;
    at::globalContext().alertNotDeterministic("recursive_seg::accumulate_crop_probabilities");
    accumulate_rois<<<crops.size(0) * 8, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        crops.data_ptr(), static_cast<int>(crops.scalar_type()), boxes.data_ptr<float>(), indices.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(), sums.data_ptr<float>(), counts.data_ptr<float>(), sums.size(0), sums.size(1),
        sums.size(2), sums.size(3), image_size[0], image_size[1], crops.size(2), crops.size(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor finalize_crop_probabilities_cuda(torch::Tensor sums, torch::Tensor counts,
    torch::Tensor sizes, torch::Tensor base, bool return_coverage) {
    TORCH_CHECK(sums.is_cuda() && sums.is_contiguous() && sums.scalar_type() == torch::kFloat32 &&
                sums.dim() == 4 && sums.size(0) > 0 && sums.size(1) > 0 &&
                sums.size(2) > 0 && sums.size(3) > 0, "Sums must be nonempty contiguous CUDA FP32 BCHW");
    for (const auto& value : {counts, sizes, base})
        TORCH_CHECK(value.is_cuda() && value.is_contiguous() && value.device() == sums.device(),
                    "Finalization tensors must be contiguous on one CUDA device");
    TORCH_CHECK(counts.scalar_type() == torch::kFloat32 && counts.dim() == 4 &&
                counts.size(0) == sums.size(0) && counts.size(1) == 1 &&
                counts.size(2) == sums.size(2) && counts.size(3) == sums.size(3),
                "Counts must be FP32 B1HW matching sums");
    TORCH_CHECK(sizes.scalar_type() == torch::kInt64 && sizes.dim() == 2 &&
                sizes.size(0) == sums.size(0) && sizes.size(1) == 2, "Sizes must be int64 [B,2]");
    bool has_base = base.numel() > 0;
    TORCH_CHECK(!has_base || (supported_float(base) && base.sizes() == sums.sizes()),
                "Base probabilities must match sums shape and be FP32/FP16/BF16");
    TORCH_CHECK(!sums.is_alias_of(counts) && (!has_base || !sums.is_alias_of(base)),
                "In-place sums must not share storage with counts or base probabilities");
    TORCH_CHECK(sums.size(1) <= INT_MAX && sums.size(2) <= INT_MAX && sums.size(3) <= INT_MAX,
                "Finalization dimensions exceed int32 indexing range");
    c10::cuda::CUDAGuard guard(sums.device());
    auto coverage = return_coverage
        ? torch::empty({sums.size(0), 1, sums.size(2), sums.size(3)}, sums.options().dtype(torch::kBool))
        : torch::empty({0}, sums.options().dtype(torch::kBool));
    int64_t pixels = counts.numel();
    finalize_rois<<<(pixels + threads - 1) / threads, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        sums.data_ptr<float>(), counts.data_ptr<float>(), sizes.data_ptr<int64_t>(),
        has_base ? base.data_ptr() : nullptr, static_cast<int>(base.scalar_type()),
        return_coverage ? coverage.data_ptr<bool>() : nullptr, pixels,
        sums.size(1), sums.size(2), sums.size(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return coverage;
}
