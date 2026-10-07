#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace {
void check(cublasStatus_t s) { TORCH_CHECK(s == CUBLAS_STATUS_SUCCESS, "cuBLASLt status ", int(s)); }
template <typename T, auto Destroy> struct Resource {
    T value = nullptr;
    ~Resource() { if (value) Destroy(value); }
};

// Compare the full public configuration, including reduction/custom options.
// Equal compute dtype alone does not imply the same accumulation order.
std::vector<uint32_t> configuration(const cublasLtMatmulAlgo_t& algorithm) {
    std::vector<uint32_t> result;
    for (auto attr : {CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID,
                      CUBLASLT_ALGO_CONFIG_STAGES_ID, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                      CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                      CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
                      CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID}) {
        uint32_t value = 0;
        size_t bytes = 0;
        check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attr, nullptr, 0, &bytes));
        TORCH_CHECK(bytes <= sizeof(value), "Unexpected cuBLASLt configuration size");
        check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attr, &value, bytes, &bytes));
        result.push_back(value);
    }
    return result;
}

class Plan {
    Resource<cublasLtMatmulDesc_t, cublasLtMatmulDescDestroy> op_;
    Resource<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy> a_, b_, c_;
    at::Tensor workspace_;
    int64_t m_, n_, k_;
    bool kn_, gelu_;
    std::set<std::vector<uint32_t>> seen_;
    std::vector<cublasLtMatmulAlgo_t> candidates_;
    cublasLtMatmulAlgo_t best_;
    float best_ms_ = 0;

    void add(cublasLtHandle_t handle, const cublasLtMatmulAlgo_t& algorithm) {
        cublasLtMatmulHeuristicResult_t result{};
        auto status = cublasLtMatmulAlgoCheck(handle, op_.value, a_.value, b_.value,
                                            c_.value, c_.value, &algorithm, &result);
        if (status == CUBLAS_STATUS_SUCCESS && result.state == CUBLAS_STATUS_SUCCESS &&
            result.workspaceSize <= size_t(workspace_.numel()) && seen_.insert(configuration(algorithm)).second)
            candidates_.push_back(algorithm);
    }
    cublasStatus_t launch(const at::Tensor& x, const at::Tensor& weight, at::Tensor& out,
                         const cublasLtMatmulAlgo_t& algorithm) {
        float alpha = 1, beta = 0;
        return cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), op_.value, &alpha,
            weight.data_ptr(), a_.value, x.data_ptr(), b_.value, &beta,
            out.data_ptr(), c_.value, out.data_ptr(), c_.value, &algorithm,
            workspace_.data_ptr(), workspace_.numel(), c10::cuda::getCurrentCUDAStream());
    }
    // Preparation runs once per device/shape/layout, before Graph capture.
    // Keep the GEMM contract, candidate search and timing policy separate.
    void configure(const at::Tensor& bias) {
        check(cublasLtMatmulDescCreate(&op_.value, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        auto transpose = kn_ ? CUBLAS_OP_N : CUBLAS_OP_T;
        auto epilogue = gelu_ ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS;
        auto bias_type = CUDA_R_16BF;
        auto bias_ptr = bias.data_ptr();
        check(cublasLtMatmulDescSetAttribute(op_.value, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
        check(cublasLtMatmulDescSetAttribute(op_.value, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
        check(cublasLtMatmulDescSetAttribute(op_.value, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));
        check(cublasLtMatmulDescSetAttribute(op_.value, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
        check(cublasLtMatrixLayoutCreate(&a_.value, CUDA_R_16BF, kn_ ? n_ : k_, kn_ ? k_ : n_, kn_ ? n_ : k_));
        check(cublasLtMatrixLayoutCreate(&b_.value, CUDA_R_16BF, k_, m_, k_));
        check(cublasLtMatrixLayoutCreate(&c_.value, CUDA_R_16BF, n_, m_, n_));
    }

    void collect_candidates(cublasLtHandle_t handle, bool exhaustive) {
        Resource<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceDestroy> pref;
        check(cublasLtMatmulPreferenceCreate(&pref.value));
        size_t bytes = workspace_.numel();
        check(cublasLtMatmulPreferenceSetAttribute(pref.value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
        std::array<cublasLtMatmulHeuristicResult_t,64> heuristics{};
        int count = 0;
        check(cublasLtMatmulAlgoGetHeuristic(handle, op_.value, a_.value, b_.value, c_.value, c_.value,
                                             pref.value, heuristics.size(), heuristics.data(), &count));
        for (int i=0; i<count; ++i) add(handle, heuristics[i].algo);
        if (exhaustive) {
            const std::vector<int> ids{6,31};
            const std::vector<uint32_t> tiles{uint32_t(CUBLASLT_MATMUL_TILE_64x64),
                     uint32_t(CUBLASLT_MATMUL_TILE_128x64), uint32_t(CUBLASLT_MATMUL_TILE_128x128),
                     uint32_t(CUBLASLT_MATMUL_TILE_256x128), uint32_t(CUBLASLT_MATMUL_TILE_128x256)};
            const std::vector<uint32_t> pipeline_stages{uint32_t(CUBLASLT_MATMUL_STAGES_64x3), uint32_t(CUBLASLT_MATMUL_STAGES_64x4),
                     uint32_t(CUBLASLT_MATMUL_STAGES_64x5), uint32_t(CUBLASLT_MATMUL_STAGES_64x6)};
            const std::vector<uint32_t> splits{0u,2u,3u,4u,8u};
            for (int id : ids) for (uint32_t tile : tiles) for (uint32_t stages : pipeline_stages)
            for (uint32_t split : splits) for (uint32_t swizzle : {0u,1u}) {
                cublasLtMatmulAlgo_t algorithm;
                if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF,
                                          CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, id, &algorithm) != CUBLAS_STATUS_SUCCESS) continue;
                uint32_t reduction = split > 1 ? CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE : CUBLASLT_REDUCTION_SCHEME_NONE;
                bool valid = true;
                for (auto [key,value] : std::array<std::pair<cublasLtMatmulAlgoConfigAttributes_t,uint32_t>,5>{{
                        {CUBLASLT_ALGO_CONFIG_TILE_ID,tile},{CUBLASLT_ALGO_CONFIG_STAGES_ID,stages},
                        {CUBLASLT_ALGO_CONFIG_SPLITK_NUM,split},{CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,reduction},
                        {CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,swizzle}}})
                    valid &= cublasLtMatmulAlgoConfigSetAttribute(&algorithm,key,&value,sizeof(value)) == CUBLAS_STATUS_SUCCESS;
                if (valid) add(handle,algorithm);
            }
        }
        TORCH_CHECK(!candidates_.empty(), "No valid BF16 GEMM algorithms");
    }

    void tune(const at::Tensor& x, const at::Tensor& weight) {
        at::Tensor out = at::empty({m_,n_},x.options());
        cudaEvent_t start, stop;
        C10_CUDA_CHECK(cudaEventCreate(&start)); C10_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = c10::cuda::getCurrentCUDAStream();
        const auto* device = at::cuda::getCurrentDeviceProperties();
        const bool cache_pressure = device->major == 8 && device->minor == 9;
        at::Tensor scrub;
        if (cache_pressure) scrub = at::empty({128 << 20}, x.options().dtype(at::kByte));
        best_ms_ = std::numeric_limits<float>::infinity();
        for (auto& algorithm : candidates_) {
            auto status = launch(x,weight,out,algorithm);
            if (status != CUBLAS_STATUS_SUCCESS) { continue; }
            float ms;
            if (cache_pressure) {
                // Repeatedly timing one warm weight favors Ada's large L2.
                // Clear a scratch bank before each sample to approximate the
                // intervening layers; the scratch work is outside the timer.
                std::array<float,5> samples{};
                for (auto& sample : samples) {
                    scrub.zero_();
                    C10_CUDA_CHECK(cudaEventRecord(start,stream));
                    check(launch(x,weight,out,algorithm));
                    C10_CUDA_CHECK(cudaEventRecord(stop,stream)); C10_CUDA_CHECK(cudaEventSynchronize(stop));
                    C10_CUDA_CHECK(cudaEventElapsedTime(&sample,start,stop));
                }
                std::sort(samples.begin(),samples.end()); ms = samples[2];
            } else {
                C10_CUDA_CHECK(cudaEventRecord(start,stream));
                for (int i=0;i<16;++i) check(launch(x,weight,out,algorithm));
                C10_CUDA_CHECK(cudaEventRecord(stop,stream)); C10_CUDA_CHECK(cudaEventSynchronize(stop));
                C10_CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop)); ms /= 16;
            }
            if (ms < best_ms_) { best_ms_=ms; best_=algorithm; }
        }
        C10_CUDA_CHECK(cudaEventDestroy(start)); C10_CUDA_CHECK(cudaEventDestroy(stop));
        TORCH_CHECK(std::isfinite(best_ms_), "No candidate GEMM could execute on the current device");
    }

public:
    Plan(at::Tensor x, at::Tensor weight, at::Tensor bias, bool exhaustive, bool gelu = false)
        : m_(x.size(0)), n_(weight.size(0)), k_(x.size(1)), kn_(weight.stride(0) == 1), gelu_(gelu) {
        TORCH_CHECK(x.is_cuda() && weight.device()==x.device() && bias.device()==x.device());
        TORCH_CHECK(x.scalar_type()==at::kBFloat16 && weight.scalar_type()==x.scalar_type() && bias.scalar_type()==x.scalar_type());
        TORCH_CHECK(x.dim()==2 && weight.dim()==2 && bias.dim()==1 && bias.numel()==n_ && weight.size(1)==k_);
        const c10::cuda::CUDAGuard guard(x.device());
        TORCH_CHECK(c10::cuda::currentStreamCaptureStatusMayInitCtx()==c10::cuda::CaptureStatus::None,
                    "GEMM tuning must finish before CUDA Graph capture");
        workspace_ = at::empty({16 << 20}, x.options().dtype(at::kByte));
        auto handle = at::cuda::getCurrentCUDABlasLtHandle();
        configure(bias);
        collect_candidates(handle, exhaustive);
        tune(x, weight);
    }

    // The hot path only binds current tensors and launches the selected GEMM.
    at::Tensor run(at::Tensor x, at::Tensor weight, at::Tensor bias) {
        const c10::cuda::CUDAGuard guard(x.device());
        TORCH_CHECK(x.device()==workspace_.device() && weight.device()==x.device() && bias.device()==x.device());
        TORCH_CHECK(x.scalar_type()==at::kBFloat16 && weight.scalar_type()==x.scalar_type() && bias.scalar_type()==x.scalar_type());
        TORCH_CHECK(x.is_contiguous() && x.dim()==2 && x.size(0)==m_ && x.size(1)==k_);
        TORCH_CHECK(weight.dim()==2 && weight.size(0)==n_ && weight.size(1)==k_);
        TORCH_CHECK(weight.stride(0)==(kn_?1:k_) && weight.stride(1)==(kn_?n_:1));
        TORCH_CHECK(bias.is_contiguous() && bias.numel()==n_);
        auto ptr=bias.data_ptr();
        check(cublasLtMatmulDescSetAttribute(op_.value,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&ptr,sizeof(ptr)));
        auto out=at::empty({m_,n_},x.options()); check(launch(x,weight,out,best_)); return out;
    }
    std::vector<uint32_t> selected_configuration() const { return configuration(best_); }
};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
    pybind11::class_<Plan>(module,"Plan").def(pybind11::init<at::Tensor,at::Tensor,at::Tensor,bool,bool>(),
        pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
        pybind11::arg("exhaustive"), pybind11::arg("gelu") = false)
        .def("run",&Plan::run).def("selected_configuration",&Plan::selected_configuration);
}
