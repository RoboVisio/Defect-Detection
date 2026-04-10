// defect_dll.cpp
// 灰度图先提取中间 ROI，再切分为左右两个工件分别推理，
// 最终结果回写到未切分的整体 ROI 坐标系。
#define DEFECT_DLL_EXPORTS
#include "defect_dll.h"

#include "HalconCpp.h"
#include "onnxruntime_cxx_api.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

using namespace HalconCpp;
using clk = std::chrono::steady_clock;

void Preprocess_Color(HObject ho_Image, HObject *ho_ColorROI);
void Preprocess_Gray_WithROI(HObject ho_Image, HObject *ho_GrayROI,
                             HTuple *r1, HTuple *c1, HTuple *r2, HTuple *c2);
void Preprocess_Gray_LR(HObject ho_Image,
                        HObject *ho_Left, HObject *ho_Right,
                        HTuple *l_r1, HTuple *l_c1, HTuple *l_r2, HTuple *l_c2,
                        HTuple *r_r1, HTuple *r_c1, HTuple *r_r2, HTuple *r_c2);

namespace {

// =============================================================================
// 单个工件在流水线中的中间状态
struct SidePending {
    std::vector<float> tensor;
    int r1 = 0;
    int c1 = 0;
    int r2 = 0;
    int c2 = 0;
    bool ok = true;
    float pred_score = 0.f;
    std::vector<float> map;
    int map_h = 0;
    int map_w = 0;
};

struct Job {
    DefectImage img;
    clk::time_point t_submit;

    SidePending left;
    SidePending right;

    float t_pre_ms = 0;
    float t_inf_ms = 0;

    std::mutex              done_mu;
    std::condition_variable done_cv;
    bool                    done = false;
    void mark_done() {
        { std::lock_guard<std::mutex> lk(done_mu); done = true; }
        done_cv.notify_one();
    }
    void wait_done() {
        std::unique_lock<std::mutex> lk(done_mu);
        done_cv.wait(lk, [&]{ return done; });
    }
};
using JobPtr = std::shared_ptr<Job>;

// =============================================================================
// 阻塞队列
// =============================================================================
template <typename T>
class BlockingQueue {
public:
    void push(T v) {
        { std::lock_guard<std::mutex> lk(mu_); q_.push_back(std::move(v)); }
        cv_.notify_one();
    }
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&]{ return stopped_ || !q_.empty(); });
        if (q_.empty()) return false;
        out = std::move(q_.front()); q_.pop_front();
        return true;
    }
    bool pop_for(T& out, int us) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait_for(lk, std::chrono::microseconds(us),
                     [&]{ return stopped_ || !q_.empty(); });
        if (q_.empty()) return false;
        out = std::move(q_.front()); q_.pop_front();
        return true;
    }
    int try_drain(std::vector<T>& out, int max_n) {
        std::lock_guard<std::mutex> lk(mu_);
        int n = 0;
        while (!q_.empty() && n < max_n) {
            out.push_back(std::move(q_.front())); q_.pop_front(); ++n;
        }
        return n;
    }
    void stop() {
        { std::lock_guard<std::mutex> lk(mu_); stopped_ = true; }
        cv_.notify_all();
    }
private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<T> q_;
    bool stopped_ = false;
};

// =============================================================================
// ONNX Runtime 封装
// =============================================================================
class OrtModel {
public:
    bool load(const std::string& path, bool use_gpu, int dev_id) {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "defect");

        const auto try_create = [&](bool enable_gpu) -> bool {
            try {
                session_.reset();
                input_names_.clear();
                output_names_.clear();

                Ort::SessionOptions opts;
                opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                opts.SetIntraOpNumThreads(0);
                if (enable_gpu) {
                    OrtCUDAProviderOptions cuda{};
                    cuda.device_id = dev_id;
                    cuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
                    cuda.do_copy_in_default_stream = 1;
                    opts.AppendExecutionProvider_CUDA(cuda);
                }
#ifdef _WIN32
                std::wstring wpath(path.begin(), path.end());
                session_ = std::make_unique<Ort::Session>(*env_, wpath.c_str(), opts);
#else
                session_ = std::make_unique<Ort::Session>(*env_, path.c_str(), opts);
#endif
                Ort::AllocatorWithDefaultOptions alloc;
                for (size_t i = 0; i < session_->GetInputCount(); ++i) {
                    input_names_.push_back(session_->GetInputNameAllocated(i, alloc).get());
                }
                for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
                    output_names_.push_back(session_->GetOutputNameAllocated(i, alloc).get());
                }
                auto info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
                auto shp  = info.GetShape();
                if (shp.size() == 4) {
                    in_c_ = (int)(shp[1] > 0 ? shp[1] : 3);
                    in_h_ = (int)(shp[2] > 0 ? shp[2] : 256);
                    in_w_ = (int)(shp[3] > 0 ? shp[3] : 256);
                }
                return true;
            } catch (const Ort::Exception& e) {
                last_err_ = e.what();
                session_.reset();
                input_names_.clear();
                output_names_.clear();
                return false;
            }
        };

        if (use_gpu && try_create(true)) {
            return true;
        }
        return try_create(false);
    }
    int H() const { return in_h_; }
    int W() const { return in_w_; }
    int C() const { return in_c_; }

    bool run(int N, const float* input,
             std::vector<float>& pred_scores,
             std::vector<float>& anomaly_maps,
             int& map_h, int& map_w)
    {
        try {
            std::vector<int64_t> shape{N, in_c_, in_h_, in_w_};
            auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto in = Ort::Value::CreateTensor<float>(
                mem, const_cast<float*>(input),
                (size_t)N * in_c_ * in_h_ * in_w_,
                shape.data(), shape.size());

            std::vector<const char*> in_names_c, out_names_c;
            for (auto& s : input_names_)  in_names_c.push_back(s.c_str());
            for (auto& s : output_names_) out_names_c.push_back(s.c_str());

            auto outs = session_->Run(Ort::RunOptions{nullptr},
                                      in_names_c.data(), &in, 1,
                                      out_names_c.data(), out_names_c.size());

            pred_scores.assign(N, 0.f);
            anomaly_maps.clear();
            map_h = map_w = 0;
            for (size_t i = 0; i < outs.size(); ++i) {
                const auto& name = output_names_[i];
                auto info = outs[i].GetTensorTypeAndShapeInfo();
                auto shp  = info.GetShape();
                const float* p = outs[i].GetTensorData<float>();
                size_t sz = info.GetElementCount();
                if (name.find("score") != std::string::npos && shp.size() <= 2) {
                    for (int n = 0; n < N && n < (int)sz; ++n) pred_scores[n] = p[n];
                } else if (name.find("anomaly") != std::string::npos) {
                    if (shp.size() == 4) { map_h=(int)shp[2]; map_w=(int)shp[3]; }
                    else if (shp.size() == 3) { map_h=(int)shp[1]; map_w=(int)shp[2]; }
                    anomaly_maps.assign(p, p + sz);
                }
            }
            if (!anomaly_maps.empty() && map_h && map_w) {
                size_t per = (size_t)map_h * map_w;
                for (int n = 0; n < N; ++n) {
                    if (pred_scores[n] != 0.f) continue;
                    const float* mp = anomaly_maps.data() + n * per;
                    float mx = 0.f;
                    for (size_t k = 0; k < per; ++k) if (mp[k] > mx) mx = mp[k];
                    pred_scores[n] = mx;
                }
            }
            return true;
        } catch (const Ort::Exception& e) { last_err_ = e.what(); return false; }
    }
private:
    std::unique_ptr<Ort::Env>     env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string>      input_names_, output_names_;
    int in_c_ = 3, in_h_ = 256, in_w_ = 256;
    std::string last_err_;
};

// =============================================================================
// 全局上下文
struct Ctx {
    DefectConfig cfg{};
    std::atomic<bool> running{false};
    OrtModel gray_model, color_model;
    BlockingQueue<JobPtr> in_gray, in_color;
    BlockingQueue<JobPtr> ready_gray, ready_color;
    std::vector<std::thread> preproc_threads;
    std::thread infer_gray, infer_color;
};
std::unique_ptr<Ctx> g_ctx;

// =============================================================================
// HALCON 工具函数
// =============================================================================
HObject build_halcon_image(const DefectImage& img) {
    HObject h;
    if (img.channels == 1)
        GenImage1(&h, "byte", img.width, img.height, (Hlong)(size_t)img.data);
    else
        GenImageInterleaved(&h, (Hlong)(size_t)img.data, "rgb",
                            img.width, img.height, -1, "byte",
                            img.width, img.height, 0, 0, -1, 0);
    return h;
}

void halcon_to_tensor(HObject img, int dst_h, int dst_w, int dst_c,
                      std::vector<float>& out)
{
    HObject zoomed;
    ZoomImageSize(img, &zoomed, dst_w, dst_h, "constant");
    HTuple ch; CountChannels(zoomed, &ch);

    out.assign((size_t)dst_c * dst_h * dst_w, 0.f);
    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float stdv[3] = {0.229f, 0.224f, 0.225f};

    auto fill = [&](HObject plane, int c) {
        HTuple ptr, type, w, h;
        GetImagePointer1(plane, &ptr, &type, &w, &h);
        const uint8_t* p = reinterpret_cast<const uint8_t*>((size_t)(Hlong)ptr);
        float* d = out.data() + (size_t)c * dst_h * dst_w;
        const float m = mean[c], s = stdv[c];
        const int N = dst_h * dst_w;
        for (int i = 0; i < N; ++i) d[i] = ((float)p[i] / 255.f - m) / s;
    };
    if ((int)ch == 1 && dst_c == 3) {
        for (int c = 0; c < 3; ++c) fill(zoomed, c);
    } else if ((int)ch == 3) {
        HObject r, g, b;
        Decompose3(zoomed, &r, &g, &b);
        fill(r, 0); fill(g, 1); fill(b, 2);
    } else {
        HTuple ptr, type, w, h;
        GetImagePointer1(zoomed, &ptr, &type, &w, &h);
        const uint8_t* p = reinterpret_cast<const uint8_t*>((size_t)(Hlong)ptr);
        for (int i = 0; i < dst_h * dst_w; ++i) out[i] = (float)p[i] / 255.f;
    }
}

// 双线性缩放: src(sh, sw) -> dst(dh, dw)
static void bilinear_resize(const float* src, int sh, int sw,
                            float* dst, int dh, int dw)
{
    if (sh == dh && sw == dw) {
        std::memcpy(dst, src, sizeof(float) * sh * sw); return;
    }
    const float fx = (float)sw / dw, fy = (float)sh / dh;
    for (int y = 0; y < dh; ++y) {
        float sy = (y + 0.5f) * fy - 0.5f;
        int   y0 = (int)std::floor(sy);
        float ty = sy - y0;
        int   y1 = std::min(std::max(y0,     0), sh - 1);
        int   y2 = std::min(std::max(y0 + 1, 0), sh - 1);
        for (int x = 0; x < dw; ++x) {
            float sx = (x + 0.5f) * fx - 0.5f;
            int   x0 = (int)std::floor(sx);
            float tx = sx - x0;
            int   x1 = std::min(std::max(x0,     0), sw - 1);
            int   x2 = std::min(std::max(x0 + 1, 0), sw - 1);
            float v00 = src[y1*sw+x1], v01 = src[y1*sw+x2];
            float v10 = src[y2*sw+x1], v11 = src[y2*sw+x2];
            float v0 = v00*(1-tx) + v01*tx;
            float v1 = v10*(1-tx) + v11*tx;
            dst[y*dw+x] = v0*(1-ty) + v1*ty;
        }
    }
}

// 将模型输出的 anomaly map (sub_h, sub_w) 贴回整体 ROI buffer 的指定子矩形。
// roi_buf: 整个中间 ROI 的 float 缓冲区，尺寸为 roi_h * roi_w。
// off_r/c: 子矩形相对 ROI 左上角的偏移。
// sub_h/w: 子矩形的实际像素尺寸。
static void paste_sub_into_roi(const float* sub_map, int map_h, int map_w,
                               float* roi_buf, int roi_w,
                               int off_r, int off_c, int sub_h, int sub_w)
{
    if (sub_h <= 0 || sub_w <= 0) return;
    std::vector<float> resized((size_t)sub_h * sub_w);
    bilinear_resize(sub_map, map_h, map_w, resized.data(), sub_h, sub_w);
    for (int y = 0; y < sub_h; ++y)
        std::memcpy(roi_buf + (size_t)(off_r + y) * roi_w + off_c,
                    resized.data() + (size_t)y * sub_w,
                    sizeof(float) * sub_w);
}

// =============================================================================
// 预处理线程
// =============================================================================
void preprocess_loop() {
    auto& ctx = *g_ctx;
    while (ctx.running.load()) {
        JobPtr job;
        if (!ctx.in_gray.pop_for(job, 200)) {
            if (!ctx.in_color.pop_for(job, 200)) continue;
        }
        if (!job) continue;

        auto t0 = clk::now();
        try {
            HObject orig = build_halcon_image(job->img);
            int H, W, C;
            if (job->img.camera_type == DEFECT_CAM_COLOR) {
                HObject roi_img;
                Preprocess_Color(orig, &roi_img);
                job->left.r1 = 0;
                job->left.c1 = 0;
                job->left.r2 = job->img.height - 1;
                job->left.c2 = job->img.width - 1;
                job->right.ok = false;
                H = ctx.color_model.H();
                W = ctx.color_model.W();
                C = ctx.color_model.C();
                halcon_to_tensor(roi_img, H, W, C, job->left.tensor);
            } else {
                HObject leftImg, rightImg;
                HTuple lr1,lc1,lr2,lc2, rr1,rc1,rr2,rc2;
                Preprocess_Gray_LR(orig, &leftImg, &rightImg,
                                   &lr1,&lc1,&lr2,&lc2,
                                   &rr1,&rc1,&rr2,&rc2);
                job->left.r1=(int)(Hlong)lr1; job->left.c1=(int)(Hlong)lc1;
                job->left.r2=(int)(Hlong)lr2; job->left.c2=(int)(Hlong)lc2;
                job->right.r1=(int)(Hlong)rr1; job->right.c1=(int)(Hlong)rc1;
                job->right.r2=(int)(Hlong)rr2; job->right.c2=(int)(Hlong)rc2;
                H = ctx.gray_model.H(); W = ctx.gray_model.W(); C = ctx.gray_model.C();
                halcon_to_tensor(leftImg,  H, W, C, job->left.tensor);
                halcon_to_tensor(rightImg, H, W, C, job->right.tensor);
            }
        } catch (HException&) {
            job->left.ok = false;
            job->right.ok = false;
        }
        job->t_pre_ms =
            std::chrono::duration<float, std::milli>(clk::now() - t0).count();

        if (job->img.camera_type == DEFECT_CAM_COLOR)
            ctx.ready_color.push(std::move(job));
        else
            ctx.ready_gray.push(std::move(job));
    }
}

// =============================================================================
// 推理线程。灰度图每张图展开为左右两个样本，彩色图暂时仍是单样本。
// =============================================================================
void inference_loop(BlockingQueue<JobPtr>& q, OrtModel& model, bool is_color) {
    auto& ctx = *g_ctx;
    const int max_batch_jobs = is_color ? ctx.cfg.color_max_batch
                                        : ctx.cfg.gray_max_batch;
    const int max_wait_us = ctx.cfg.max_wait_us > 0 ? ctx.cfg.max_wait_us : 1000;
    const int H = model.H(), W = model.W(), C = model.C();
    const int per_sample = C * H * W;

    while (ctx.running.load()) {
        std::vector<JobPtr> jobs;
        JobPtr first;
        if (!q.pop(first)) continue;
        jobs.push_back(std::move(first));

        auto t_start = clk::now();
        while ((int)jobs.size() < max_batch_jobs) {
            int rest = max_wait_us -
                (int)std::chrono::duration_cast<std::chrono::microseconds>(
                    clk::now() - t_start).count();
            if (rest <= 0) break;
            std::vector<JobPtr> drained;
            int got = q.try_drain(drained, max_batch_jobs - (int)jobs.size());
            if (got > 0) {
                for (auto& j : drained) jobs.push_back(std::move(j));
                continue;
            }
            JobPtr j;
            if (q.pop_for(j, std::min(rest, 500))) jobs.push_back(std::move(j));
        }

        // 组 batch，左右工件分别作为独立样本。
        std::vector<float>     batch;
        std::vector<int>       sample_owner;  // 姣忎釜鏍锋湰灞炰簬鍝釜 job
        std::vector<int>       sample_side;   // 0=left, 1=right
        batch.reserve(jobs.size() * 2 * per_sample);

        for (int i = 0; i < (int)jobs.size(); ++i) {
            auto& j = jobs[i];
            for (int side = 0; side < 2; ++side) {
                SidePending& sp = (side == 0) ? j->left : j->right;
                if (!sp.ok || (int)sp.tensor.size() != per_sample) continue;
                size_t off = batch.size();
                batch.resize(off + per_sample);
                std::memcpy(batch.data() + off, sp.tensor.data(),
                            sizeof(float) * per_sample);
                sample_owner.push_back(i);
                sample_side.push_back(side);
            }
        }

        auto t_inf0 = clk::now();
        std::vector<float> scores, maps;
        int map_h = 0, map_w = 0;
        bool ok = false;
        const int N = (int)sample_owner.size();
        if (N > 0) ok = model.run(N, batch.data(), scores, maps, map_h, map_w);
        float t_inf_ms =
            std::chrono::duration<float, std::milli>(clk::now() - t_inf0).count();
        float per_job_inf = t_inf_ms / std::max(1, (int)jobs.size());

        // 回填推理结果。
        size_t per_map = (size_t)map_h * map_w;
        for (int s = 0; s < N; ++s) {
            auto& j = jobs[sample_owner[s]];
            SidePending& sp = (sample_side[s] == 0) ? j->left : j->right;
            sp.pred_score = ok ? scores[s] : 0.f;
            sp.map_h = map_h; sp.map_w = map_w;
            if (ok && per_map) {
                sp.map.assign(maps.data() + s * per_map,
                              maps.data() + (s + 1) * per_map);
            }
        }

        for (auto& j : jobs) {
            j->t_inf_ms = per_job_inf;
            j->mark_done();
        }
    }
}

// =============================================================================
// fill_result: 将 Job 内部状态写入对外的 DefectResult。
// =============================================================================
void fill_result(const Job& j, float threshold, bool enable_map, DefectResult* out)
{
    std::memset(out, 0, sizeof(*out));
    auto t_post0 = clk::now();

    out->image_id  = j.img.image_id;
    out->camera_id = j.img.camera_id;
    out->threshold = threshold;

    auto fill_side = [&](const SidePending& s, WorkpieceResult& w) {
        w.confidence = s.pred_score;
        w.ok_ng = (s.pred_score >= threshold) ? DEFECT_VERDICT_NG : DEFECT_VERDICT_OK;
        w.roi_row1 = s.r1; w.roi_col1 = s.c1;
        w.roi_row2 = s.r2; w.roi_col2 = s.c2;
    };
    fill_side(j.left,  out->left);
    fill_side(j.right, out->right);

    bool overall_ng =
        (j.left.ok  && out->left.ok_ng  == DEFECT_VERDICT_NG) ||
        (j.right.ok && out->right.ok_ng == DEFECT_VERDICT_NG);
    out->overall_ok_ng = overall_ng ? DEFECT_VERDICT_NG : DEFECT_VERDICT_OK;

    // 整体 ROI 由左半和右半合并得到。
    int R1 = j.left.r1,  C1 = j.left.c1;
    int R2 = j.right.r2, C2 = j.right.c2;
    if (!j.right.ok) { R2 = j.left.r2; C2 = j.left.c2; }
    if (!j.left.ok)  { R1 = j.right.r1; C1 = j.right.c1; }
    out->roi_row1 = R1; out->roi_col1 = C1;
    out->roi_row2 = R2; out->roi_col2 = C2;

    // anomaly map 回贴到未切分的整体 ROI 空间。
        const int roi_w = C2 - C1 + 1;
        const int roi_h = R2 - R1 + 1;
        if (roi_w > 0 && roi_h > 0) {
            float* buf = (float*)std::calloc((size_t)roi_w * roi_h, sizeof(float));
            if (buf) {
                if (j.left.ok && !j.left.map.empty()) {
                    int off_r = j.left.r1 - R1;
                    int off_c = j.left.c1 - C1;
                    int sub_h = j.left.r2 - j.left.r1 + 1;
                    int sub_w = j.left.c2 - j.left.c1 + 1;
                    paste_sub_into_roi(j.left.map.data(), j.left.map_h, j.left.map_w,
                                       buf, roi_w, off_r, off_c, sub_h, sub_w);
                }
                if (j.right.ok && !j.right.map.empty()) {
                    int off_r = j.right.r1 - R1;
                    int off_c = j.right.c1 - C1;
                    int sub_h = j.right.r2 - j.right.r1 + 1;
                    int sub_w = j.right.c2 - j.right.c1 + 1;
                    paste_sub_into_roi(j.right.map.data(), j.right.map_h, j.right.map_w,
                                       buf, roi_w, off_r, off_c, sub_h, sub_w);
                }
                out->anomaly_map = buf;
                out->map_width   = roi_w;
                out->map_height  = roi_h;
            }
        }
    }

    out->time_preprocess_ms = j.t_pre_ms;
    out->time_infer_ms      = j.t_inf_ms;
    out->time_post_ms       =
        std::chrono::duration<float, std::milli>(clk::now() - t_post0).count();
    out->time_total_ms      =
        std::chrono::duration<float, std::milli>(clk::now() - j.t_submit).count();
}

} // namespace

// =============================================================================
// 对外 API
// =============================================================================
extern "C" {

DEFECT_API int Algo_Init(const DefectConfig* cfg) {
    if (!cfg || !cfg->gray_onnx_path || !cfg->color_onnx_path)
        return DEFECT_ERR_BAD_ARG;
    if (g_ctx) return DEFECT_OK;

    g_ctx = std::make_unique<Ctx>();
    g_ctx->cfg = *cfg;
    auto& c = g_ctx->cfg;
    if (c.gray_max_batch     <= 0) c.gray_max_batch     = 16;
    if (c.color_max_batch    <= 0) c.color_max_batch    = 8;
    if (c.max_wait_us        <= 0) c.max_wait_us        = 1000;
    if (c.preprocess_threads <= 0) c.preprocess_threads = std::max(2u, std::thread::hardware_concurrency() / 2);
    if (c.gray_threshold     <= 0) c.gray_threshold     = 0.5f;
    if (c.color_threshold    <= 0) c.color_threshold    = 0.5f;

    SetHcppInterfaceStringEncodingIsUtf8(false);
    try { SetSystem("parallelize_operators", "false"); } catch (...) {}

    if (!g_ctx->gray_model.load(c.gray_onnx_path,  c.use_gpu != 0, c.gpu_device_id) ||
        !g_ctx->color_model.load(c.color_onnx_path, c.use_gpu != 0, c.gpu_device_id)) {
        g_ctx.reset();
        return DEFECT_ERR_INTERNAL;
    }

    g_ctx->running = true;
    for (int i = 0; i < c.preprocess_threads; ++i)
        g_ctx->preproc_threads.emplace_back(preprocess_loop);
    g_ctx->infer_gray  = std::thread(inference_loop, std::ref(g_ctx->ready_gray),
                                     std::ref(g_ctx->gray_model),  false);
    g_ctx->infer_color = std::thread(inference_loop, std::ref(g_ctx->ready_color),
                                     std::ref(g_ctx->color_model), true);
    return DEFECT_OK;
}

DEFECT_API int Algo_Release() {
    if (!g_ctx) return DEFECT_OK;
    g_ctx->running = false;
    g_ctx->in_gray.stop();   g_ctx->in_color.stop();
    g_ctx->ready_gray.stop();g_ctx->ready_color.stop();
    for (auto& t : g_ctx->preproc_threads) if (t.joinable()) t.join();
    if (g_ctx->infer_gray.joinable())  g_ctx->infer_gray.join();
    if (g_ctx->infer_color.joinable()) g_ctx->infer_color.join();
    g_ctx.reset();
    return DEFECT_OK;
}

DEFECT_API int Algo_ProcessImage(const DefectImage* image, DefectResult* result) {
    if (!g_ctx)  return DEFECT_ERR_NOT_INIT;
    if (!image || !result || !image->data) return DEFECT_ERR_BAD_ARG;

    auto job = std::make_shared<Job>();
    job->img = *image;
    job->t_submit = clk::now();
    if (image->camera_type == DEFECT_CAM_COLOR) g_ctx->in_color.push(job);
    else                                        g_ctx->in_gray.push(job);
    job->wait_done();

    const float thr = (image->camera_type == DEFECT_CAM_COLOR)
                    ? g_ctx->cfg.color_threshold
                    : g_ctx->cfg.gray_threshold;
    fill_result(*job, thr, g_ctx->cfg.enable_anomaly_map != 0, result);
    return DEFECT_OK;
}

DEFECT_API void Algo_ReleaseResult(DefectResult* r) {
    if (!r) return;
    if (r->anomaly_map) { std::free(r->anomaly_map); r->anomaly_map = nullptr; }
}

} // extern "C"

