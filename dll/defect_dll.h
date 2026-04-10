#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  ifdef DEFECT_DLL_EXPORTS
#    define DEFECT_API __declspec(dllexport)
#  else
#    define DEFECT_API __declspec(dllimport)
#  endif
#else
#  define DEFECT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DefectCameraType {
    DEFECT_CAM_GRAY = 0,
    DEFECT_CAM_COLOR = 1
} DefectCameraType;

typedef enum DefectStatus {
    DEFECT_OK = 0,
    DEFECT_ERR_NOT_INIT = -1,
    DEFECT_ERR_BAD_ARG = -2,
    DEFECT_ERR_TIMEOUT = -4,
    DEFECT_ERR_INTERNAL = -99
} DefectStatus;

typedef enum DefectVerdict {
    DEFECT_VERDICT_OK = 0,
    DEFECT_VERDICT_NG = 1
} DefectVerdict;

typedef struct DefectImage {
    int64_t image_id;
    int32_t camera_id;
    DefectCameraType camera_type;
    int64_t timestamp_ns;
    const uint8_t* data;
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t stride;
} DefectImage;

typedef struct WorkpieceResult {
    DefectVerdict ok_ng;
    float confidence;
    int32_t roi_row1;
    int32_t roi_col1;
    int32_t roi_row2;
    int32_t roi_col2;
} WorkpieceResult;

typedef struct DefectResult {
    int64_t image_id;
    int32_t camera_id;
    DefectVerdict overall_ok_ng;
    float threshold;
    WorkpieceResult left;
    WorkpieceResult right;
    int32_t roi_row1;
    int32_t roi_col1;
    int32_t roi_row2;
    int32_t roi_col2;
    float* anomaly_map;
    int32_t map_width;
    int32_t map_height;
    float time_preprocess_ms;
    float time_infer_ms;
    float time_post_ms;
    float time_total_ms;
} DefectResult;

typedef struct DefectConfig {
    const char* gray_onnx_path;
    const char* color_onnx_path;
    int32_t gray_max_batch;
    int32_t color_max_batch;
    int32_t max_wait_us;
    int32_t preprocess_threads;
    int32_t use_gpu;
    int32_t gpu_device_id;
    float gray_threshold;
    float color_threshold;
    int32_t enable_anomaly_map;
} DefectConfig;

DEFECT_API int Algo_Init(const DefectConfig* cfg);
DEFECT_API int Algo_Release();
DEFECT_API int Algo_ProcessImage(const DefectImage* image, DefectResult* result);
DEFECT_API void Algo_ReleaseResult(DefectResult* result);

#ifdef __cplusplus
}
#endif
