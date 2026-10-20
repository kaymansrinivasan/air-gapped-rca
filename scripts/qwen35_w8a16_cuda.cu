// SPDX-License-Identifier: Apache-2.0
// Quantization-grid emulation. GEMM/Conv remain FP32, not native W8A16 MACs.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define Q6A_HD __host__ __device__
#else
#define Q6A_HD
#endif

static_assert(ORT_API_VERSION == 27, "Build against ONNX Runtime 1.27 headers");
template <typename Q, bool Encode> using Input = std::conditional_t<Encode, float, Q>;
template <typename Q, bool Encode> using Output = std::conditional_t<Encode, Q, float>;

template <typename Q, bool Encode>
Q6A_HD Output<Q, Encode> Convert(Input<Q, Encode> value, float scale, Q zero) {
  if constexpr (Encode) {
    const float scaled = value / scale;
    const float rounded = nearbyintf(scaled);
    const float shifted = rounded + static_cast<float>(zero);
    const float clipped = fminf(static_cast<float>(std::numeric_limits<Q>::max()),
                               fmaxf(static_cast<float>(std::numeric_limits<Q>::min()), shifted));
    return static_cast<Q>(clipped);
  } else {
    const float centered = static_cast<float>(value) - static_cast<float>(zero);
    return centered * scale;
  }
}
#ifdef __CUDACC__
template <typename Q, bool Encode>
__global__ void Qdq(const Input<Q, Encode>* input, Output<Q, Encode>* output,
                    const float* scales, const Q* zeros, size_t size,
                    size_t scale_count, size_t axis_stride) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    const size_t channel = scale_count == 1 ? 0 : (i / axis_stride) % scale_count;
    output[i] = Convert<Q, Encode>(input[i], scales[channel], zeros[channel]);
  }
}
#endif
template <typename Q, bool Encode>
struct Kernel {
  int64_t axis = 1;
  explicit Kernel(const OrtKernelInfo* info) {
    auto* status = Ort::GetApi().KernelInfoGetAttribute_int64(info, "axis", &axis);
    if (status) Ort::GetApi().ReleaseStatus(status);
  }
  OrtStatus* ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext ctx(raw);
      const auto x = ctx.GetInput(0), s = ctx.GetInput(1), z = ctx.GetInput(2);
      const auto info = x.GetTensorTypeAndShapeInfo();
      const auto shape = info.GetShape();
      const size_t count = info.GetElementCount();
      const size_t channels = s.GetTensorTypeAndShapeInfo().GetElementCount();
      if (!channels || channels != z.GetTensorTypeAndShapeInfo().GetElementCount())
        throw std::runtime_error("Scale/zero-point lengths differ");
      size_t stride = 1;
      if (channels != 1) {
        const int64_t rank = static_cast<int64_t>(shape.size());
        const int64_t a = axis < 0 ? rank + axis : axis;
        if (a < 0 || a >= rank || static_cast<size_t>(shape[a]) != channels)
          throw std::runtime_error("Per-axis quantization shape mismatch");
        for (size_t j = static_cast<size_t>(a) + 1; j < shape.size(); ++j)
          stride *= static_cast<size_t>(shape[j]);
      }
      auto y = ctx.GetOutput(0, shape);
      if (!count) return nullptr;
      const auto* input = x.GetTensorData<Input<Q, Encode>>();
      auto* output = y.GetTensorMutableData<Output<Q, Encode>>();
      const float* scales = s.GetTensorData<float>();
      const Q* zeros = z.GetTensorData<Q>();
#ifdef __CUDACC__
      auto stream = static_cast<cudaStream_t>(ctx.GetGPUComputeStream());
      Qdq<Q, Encode><<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
          input, output, scales, zeros, count, channels, stride);
      const auto error = cudaGetLastError();
      if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
#else
      for (size_t i = 0; i < count; ++i) {
        const size_t channel = channels == 1 ? 0 : (i / stride) % channels;
        output[i] = Convert<Q, Encode>(input[i], scales[channel], zeros[channel]);
      }
#endif
      return nullptr;
    } catch (const std::exception& error) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
    }
  }
};
template <typename Q, bool Encode>
struct Op : Ort::CustomOpBase<Op<Q, Encode>, Kernel<Q, Encode>, true> {
  static constexpr ONNXTensorElementDataType type = sizeof(Q) == 2 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  const char* GetName() const {
    if constexpr (sizeof(Q) == 2) return Encode ? "QuantizeU16" : "DequantizeU16";
    return Encode ? "QuantizeU8" : "DequantizeU8";
  }
  const char* GetExecutionProviderType() const {
#ifdef __CUDACC__
    return "CUDAExecutionProvider";
#else
    return "CPUExecutionProvider";  // Local differential reference only.
#endif
  }
  size_t GetInputTypeCount() const { return 3; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    return i == 1 || (i == 0 && Encode) ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : type;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return Encode ? type : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  static OrtStatusPtr InferOutputShape(Ort::ShapeInferContext& context) {
    return context.SetOutputShape(0, context.GetInputShape(0),
        Encode ? type : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT).release();
  }
  OrtStatus* CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** result) const noexcept {
    try { *result = new Kernel<Q, Encode>(info); return nullptr; }
    catch (const std::exception& error) { return api.CreateStatus(ORT_FAIL, error.what()); }
  }
};
extern "C" __attribute__((visibility("default")))
OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(ORT_API_VERSION);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "ONNX Runtime API 27 required");
  Ort::InitApi(api);
  try {
    static Op<uint16_t, true> q16; static Op<uint16_t, false> dq16;
    static Op<uint8_t, true> q8; static Op<uint8_t, false> dq8;
    static const auto domain = [] {
      auto d = std::make_unique<Ort::CustomOpDomain>("airgap.quant");
      d->Add(&q16); d->Add(&dq16); d->Add(&q8); d->Add(&dq8); return d;
    }();
    Ort::ThrowOnError(api->AddCustomOpDomain(options, *domain)); return nullptr;
  } catch (const std::exception& error) { return api->CreateStatus(ORT_FAIL, error.what()); }
}
