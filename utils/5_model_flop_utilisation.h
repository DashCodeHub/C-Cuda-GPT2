/*
File provides utilities to estimates GPU theoretical performance (FLOPS)
 and monitor real-time GPU utilization.
 - Estimate  how much of your compute power GPU theoretical performace (FLOPS) and 
 - Measure how much the GPU is actually doing while training
 - Diagnosing GPU throttling (power cap, thermal cap)
 - The flops utilisation is defined as 

                                                        (Actual FLOPs during training)
                        Model Flops Utilisation (MFU) = -------------------------------
                                                          (Max Possible FLOPS of GPU)
*/

#ifndef MFU_H
#define MFU_H

// basic imports
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// NVML can be used to query GPU temperature , power, clocks, etc.
#if __has_include(<nvml.h>) // check if NVIDIA Management Library is available
#define USE_NVML 1
#include <nvml.h> // if available then use NVML
#else
#define USE_NVML 0
#endif

// Defines integer constants for different floating-point precision modes 
// (matches the PrecisionMode enum we saw earlier `3_cublas_common_utils.h`)
#define MFUH_PRECISION_FP32 0
#define MFUH_PRECISION_FP16 1
#define MFUH_PRECISION_BF16 2

// defining a small nvml function 
// After every NVML call, it checks if the call succeeded, if not, it prints an error and exits
#if USE_NVML
inline void nvml_check(nvmlReturn_t status, const char *file, int line) {
    if (status != NVML_SUCCESS) {
        printf("[NVML ERROR] at file %s:%d:\n%s\n", file, line, nvmlErrorString(status));
        exit(EXIT_FAILURE);
    }
};
#define nvmlCheck(err) (nvml_check(err, __FILE__, __LINE__))
#endif

/*
PerfData contains:
    Theoretical FLOPS for different datatypes, clock speed and Number of Tensor Cores

*/
typedef struct {
    float TF_32;       // tensor-core performance 32 bit
    float BF_16_32;    // bf16 with 32 bit accumulate
    float FP_16_32;    // fp16 with 32 bit accumulate
    float FP_16_16;    // fp16 with 16 bit accumulate
    float FP_8_32;     // and so on
    float FP_8_16;
    float CLOCK;        // clock frequency from the spec sheet
    float CORES;        // #TCs from the spec sheet
} PerfData;

// stati hardcoded tables for different GPU generations
/*
An example about the connection between above struct and values below
lets take for ADA: {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f}

Field	Meaning	Value
TF_32    =>	   TensorFloat32 peak TFLOPS                        =>	82.6 TFLOPS
BF_16_32 =>	   BFloat16 input, FP32 accumulate peak TFLOPS	    =>  165.2 TFLOPS
FP_16_32 =>	   FP16 input, FP32 accumulate peak TFLOPS	        =>  165.2 TFLOPS
FP_16_16 =>	   FP16 input, FP16 accumulate peak TFLOPS	        =>  330.3 TFLOPS
FP_8_32	 =>    FP8 input, FP32 accumulate peak TFLOPS	        =>  330.3 TFLOPS
FP_8_16	 =>    FP8 input, FP16 accumulate peak TFLOPS	        =>  660.6 TFLOPS
CLOCK	 =>    Clock speed (boost)	                            =>  2520 MHz
CORES	 =>    Number of Tensor Cores	                        =>  512 cores
*/

static const PerfData VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
static const PerfData AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
static const PerfData AMPERE_CONSUMER = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
static const PerfData HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
static const PerfData ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

typedef struct {
    const char* name;
    const PerfData* perf_data;
    float new_cores;
    float new_mhz;
} GPUEntry;

// A lookup table that maps specific GPU model names to: which generation of GPU (`PerfData` pointer)
// Actual number of Tensor Cores, and Actual clock speed. because even if the GPU have different clock/core
// I also added the data for NVIDIA L40S which I am initially running on 
static GPUEntry gpu_db[] = {
    {"Tesla V100-SXM2-16GB", &VOLTA, 640, 1530},
    {"Tesla V100-PCIE-32GB", &VOLTA, 640, 1530},
    {"NVIDIA A100-PCIE-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-PCIE-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA RTX A2000", &AMPERE_CONSUMER, 104, 1200},
    {"NVIDIA RTX A4000", &AMPERE_CONSUMER, 192, 1560},
    {"NVIDIA RTX A4500", &AMPERE_CONSUMER, 224, 1650},
    {"NVIDIA RTX A5000", &AMPERE_CONSUMER, 256, 1695},
    {"NVIDIA RTX A5500", &AMPERE_CONSUMER, 320, 1770},
    {"NVIDIA RTX A6000", &AMPERE_CONSUMER, 336, 1800},
    {"NVIDIA GeForce RTX 3090 Ti", &AMPERE_CONSUMER, 336, 1860},
    {"NVIDIA GeForce RTX 3090", &AMPERE_CONSUMER, 328, 1695},
    {"NVIDIA GeForce RTX 3080 Ti", &AMPERE_CONSUMER, 320, 1665},
    {"NVIDIA GeForce RTX 3080", &AMPERE_CONSUMER, 272, 1710},
    {"NVIDIA GeForce RTX 3070 Ti", &AMPERE_CONSUMER, 192, 1770},
    {"NVIDIA GeForce RTX 3070", &AMPERE_CONSUMER, 184, 1725},
    {"NVIDIA GeForce RTX 3060 Ti", &AMPERE_CONSUMER, 152, 1665},
    {"NVIDIA GeForce RTX 3060", &AMPERE_CONSUMER, 112, 1777},
    {"NVIDIA RTX A2000 ADA", &ADA, 88, 2130},
    {"NVIDIA RTX A4000 ADA", &ADA, 192, 2175},
    {"NVIDIA RTX A4500 ADA", &ADA, 224, 2580},
    {"NVIDIA RTX A5000 ADA", &ADA, 400, 2550},
    {"NVIDIA RTX A5880 ADA", &ADA, 440, 2460},
    {"NVIDIA RTX A6000 ADA", &ADA, 568, 2505},
    {"NVIDIA RTX L40S", &ADA, 568, 1980},  // <--- I am adding this extra line for L40S, has 568 Tensor Cores and 1980 MhZ boost clock
    {"NVIDIA GeForce RTX 4090", &ADA, 512, 2520},
    {"NVIDIA GeForce RTX 4080 SUPER", &ADA, 320, 2550},
    {"NVIDIA GeForce RTX 4080", &ADA, 304, 2505},
    {"NVIDIA GeForce RTX 4070 Ti SUPER", &ADA, 264, 2610},
    {"NVIDIA GeForce RTX 4070 Ti", &ADA, 240, 2610},
    {"NVIDIA GeForce RTX 4070 SUPER", &ADA, 224, 2475},
    {"NVIDIA GeForce RTX 4070", &ADA, 184, 2475},
    {"NVIDIA GeForce RTX 4070", &ADA, 184, 2475},
    {"NVIDIA GeForce RTX 4060 Ti", &ADA, 136, 2535},
    {"NVIDIA GeForce RTX 4060", &ADA, 96, 2460},
    {"NVIDIA H100 PCIe", &HOPPER, 456, 1620},
    {"NVIDIA H100 80GB HBM3", &HOPPER, 528, 1830}, // HBM3 = SXM5
};

/*
function below estimates the theoretical peak FLOPS of a given precision (FP32, FP16, BF16)
based on hardware database (gpu_db) and simple linear scaling. This theoretical value is then 
used to compute Model FLOP utilisation (MFU).
*/
float get_flops_promised(const char* device, int precision_mode) {
   // check if the given `precision_mode` is valid: Only FP32, FP16, BF16 are allowed
    if (!(precision_mode == MFUH_PRECISION_FP32 || precision_mode == MFUH_PRECISION_FP16 || precision_mode == MFUH_PRECISION_BF16)) {
        fprintf(stderr, "Invalid precision mode: %d\n", precision_mode);
        return -1.0f; // If not valid -> print error and return -1.0f
    }

    
    int num_gpu_entries = sizeof(gpu_db) / sizeof(gpu_db[0]);
    // loop through GPU database (`gpu_db[]`)
    for (int i = 0; i < num_gpu_entries; i++) {
        // Find the entry whose name matches the input `device` string e.g. (NVIDIA RTX 4090 or NVIDIA RTX L40S)
        if (strcmp(gpu_db[i].name, device) == 0) {
            const PerfData* perf_data = gpu_db[i].perf_data;

            // look inside the PerfData struct associated with thi GPU.
            float value = -1.0f;
            if (precision_mode == MFUH_PRECISION_BF16) { value = perf_data->BF_16_32; }
            if (precision_mode == MFUH_PRECISION_FP32) { value = perf_data->TF_32; }
            if (precision_mode == MFUH_PRECISION_FP16) { value = perf_data->FP_16_32; }

            // If the selected value is negative (e.g., no BF16 on V100), print error and return -1.0f
            if (value < 0.0f) {
                fprintf(stderr, "No data for GPU %s and precision mode %d\n", device, precision_mode);
                return -1.0f;
            }

            /*
            Adjust the flop value because:
                - Different GPUs of the same architecture have:
                    * Different number of Tensor Cores (`new_cores`)
                    * Different clock speeds (`new_mhz`)
                - So we linearly scale the original FLOPS value using:
                    * adjusted FLOPS = base Flops x (actual tensor cores / reference tensor cores) x (actual clock speed / reference clock speed)
                - This scaling is critical to account for the differnces between models
            */
            float new_cores = gpu_db[i].new_cores;
            float new_mhz = gpu_db[i].new_mhz;
            float adjusted = value * (new_cores / perf_data->CORES) * (new_mhz / perf_data->CLOCK);
            return adjusted; // returns the adjusted theoretical flops
        }
    }

    return -1.0f; // If GPU not found at all
}

struct GPUUtilInfo {
    unsigned int clock;
    unsigned int max_clock;
    unsigned int power;
    unsigned int power_limit;
    unsigned int fan;
    unsigned int temperature;
    unsigned int temp_slowdown;

    float gpu_utilization;
    float mem_utilization;
    const char* throttle_reason;
};

/*
below function does two things:
    - Initialize NVML (NVIDIA Management Library) if it hasn't been done yet
    - Get a handle (pointer-like object) to the GPU device (GPU index 0)
So any time we call this funtion, you get back a valid `nvmlDevice_t` that you can use to
query temperature, clock_speed, memory usage, etc.
*/
#if USE_NVML
nvmlDevice_t nvml_get_device() {
    static bool needs_init = true; // NVML has not been initialized yet
    static nvmlDevice_t device; // static variable to store the GPU handle once it's created
    if(needs_init) {
        needs_init = false;
        nvmlCheck(nvmlInit());
        nvmlCheck(nvmlDeviceGetHandleByIndex_v2(0, &device)); // to get a handle to GPU at index 0
    }
    return device;
}

// interpret and translate a set of bit flags returned by NVML into a human-readable
// string that explains why GPU clock speed is slowed down
const char* get_throttle_reason(unsigned long long bits) {
    if(bits & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        return "power cap";
    } else if (bits & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        return "thermal cap";
    } else if (bits & (nvmlClocksThrottleReasonAll)) {
        return "other cap";
    } else {
        return "no cap";
    }
}

/*
Collects real-time live status information about the GPU (like clock speed, power usage, 
temperature, utilization, throttling) using NVML API, and returns it nicely in a GPUUtilInfo 
struct.
This allows us to monitor the GPU during model training or inference.
*/
GPUUtilInfo get_gpu_utilization_info() {
    GPUUtilInfo info;
    nvmlDevice_t device = nvml_get_device();
    nvmlCheck(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &info.clock));
    nvmlCheck(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &info.max_clock));
    nvmlCheck(nvmlDeviceGetPowerManagementLimit(device, &info.power_limit));
    nvmlCheck(nvmlDeviceGetPowerUsage(device, &info.power));
    nvmlCheck(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temperature));
    nvmlCheck(nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &info.temp_slowdown));
    unsigned long long throttle;
    nvmlCheck(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle));
    info.throttle_reason = get_throttle_reason(throttle);
    nvmlCheck(nvmlDeviceGetFanSpeed(device, &info.fan));

    constexpr const int BUFFER_LIMIT = 128; // Collect upto 128 recent GPU utilization samples
    nvmlSample_t buffer[BUFFER_LIMIT];
    nvmlValueType_t v_type;
    unsigned int sample_count = BUFFER_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_count, buffer));
    float gpu_utilization = 0.f;
    for(unsigned i = 0; i < sample_count; ++i) {
        gpu_utilization += (float)buffer[i].sampleValue.uiVal;
    }
    gpu_utilization /= (float)sample_count; // average all samples to get smoother and more accurate utilization

    // Same process for memory usage
    sample_count = BUFFER_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_MEMORY_UTILIZATION_SAMPLES, 0, &v_type, &sample_count, buffer));
    float mem_utilization = 0.f;
    for(unsigned i = 0; i < sample_count; ++i) {
        mem_utilization += (float)buffer[i].sampleValue.uiVal;
    }
    mem_utilization /= (float)sample_count; // avearage

    // finally store the average in the `info` struct
    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;
    return info;
}
#else
GPUUtilInfo get_gpu_utilization_info() { // this is what happens without NVML support
    fprintf(stderr, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    exit(EXIT_FAILURE);
}
#endif
#endif // MFU_H.
