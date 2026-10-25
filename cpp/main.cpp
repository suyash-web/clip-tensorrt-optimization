
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << "\n";
        }
    }
};

void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

std::vector<char> readBytes(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open: " + filename);

    auto size = file.tellg();
    if (size <= 0) throw std::runtime_error("Empty file: " + filename);

    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0);
    file.read(data.data(), size);
    if (!file) throw std::runtime_error("Cannot read: " + filename);

    return data;
}

std::vector<float> readFloats(
    const std::string& filename,
    size_t expectedCount
) {
    auto bytes = readBytes(filename);

    if (bytes.size() != expectedCount * sizeof(float)) {
        throw std::runtime_error("Unexpected tensor size: " + filename);
    }

    std::vector<float> values(expectedCount);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

constexpr int IMAGE_SIZE = 224;

const float IMAGE_MEAN[3] = {
    0.48145466f,
    0.4578275f,
    0.40821073f
};

const float IMAGE_STD[3] = {
    0.26862954f,
    0.26130258f,
    0.27577711f
};


// Cubic convolution kernel (Keys, a = -0.5)
static double cubicKernel(double x) {
    x = std::abs(x);

    if (x < 1.0) {
        return (1.5 * x - 2.5) * x * x + 1.0;
    }
    if (x < 2.0) {
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    }
    return 0.0;
}

static unsigned char toByte(double x) {
    return static_cast<unsigned char>(
        std::max(0.0, std::min(255.0, std::floor(x + 0.5)))
    );
}


static cv::Mat resizeAxis(
    const cv::Mat& src,
    int newSize,
    bool horizontal
) {
    const int oldSize = horizontal ? src.cols : src.rows;
    const double scale = double(oldSize) / newSize;
    const double filterScale = std::max(1.0, scale);
    const double support = 2.0 * filterScale;

    cv::Mat dst = horizontal
        ? cv::Mat(src.rows, newSize, CV_8UC3)
        : cv::Mat(newSize, src.cols, CV_8UC3);

    // Process independent output positions concurrently
    cv::parallel_for_(
        cv::Range(0, newSize),
        [&](const cv::Range& range) {

            for (int outIndex = range.start;
                 outIndex < range.end;
                 ++outIndex) {

                const double center = (outIndex + 0.5) * scale;

                const int first = std::max(
                    0,
                    int(std::floor(center - support + 0.5))
                );

                const int last = std::min(
                    oldSize,
                    int(std::floor(center + support + 0.5))
                );

                std::vector<double> weights(last - first);
                double weightSum = 0.0;

                for (int i = first; i < last; ++i) {
                    double w = cubicKernel(
                        (i - center + 0.5) / filterScale
                    );
                    weights[i - first] = w;
                    weightSum += w;
                }

                if (std::abs(weightSum) < 1e-12)
                    throw std::runtime_error(
                        "Invalid bicubic weight sum"
                    );

                for (double& w : weights)
                    w /= weightSum;

                const int otherSize =
                    horizontal ? src.rows : src.cols;

                for (int other = 0;
                     other < otherSize;
                     ++other) {

                    double sums[3] = {0.0, 0.0, 0.0};

                    for (int i = first; i < last; ++i) {
                        const cv::Vec3b pixel = horizontal
                            ? src.at<cv::Vec3b>(other, i)
                            : src.at<cv::Vec3b>(i, other);

                        const double weight = weights[i - first];

                        for (int c = 0; c < 3; ++c)
                            sums[c] += weight * pixel[c];
                    }

                    cv::Vec3b outputPixel;

                    for (int c = 0; c < 3; ++c)
                        outputPixel[c] = toByte(sums[c]);

                    if (horizontal)
                        dst.at<cv::Vec3b>(other, outIndex) =
                            outputPixel;
                    else
                        dst.at<cv::Vec3b>(outIndex, other) =
                            outputPixel;
                }
            }
        }
    );

    return dst;
}


static cv::Mat pillowStyleResize(
    const cv::Mat& src,
    int width,
    int height
) {
    // Pillow-style separable filtering: horizontal first, vertical second
    cv::Mat result = src;

    if (result.cols != width) {
        result = resizeAxis(result, width, true);
    }

    if (result.rows != height) {
        result = resizeAxis(result, height, false);
    }

    return result;
}

std::vector<float> preprocess(const std::string& image_path) {

    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);

    if (bgr.empty()) {
        throw std::runtime_error(
            "Cannot read image: " + image_path
        );
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    const int width = rgb.cols;
    const int height = rgb.rows;

    // Resize preserving aspect ratio: shortest edge becomes 224 pixels
    int new_width;
    int new_height;

    if (width <= height) {
        new_width = IMAGE_SIZE;
        new_height = static_cast<int>(
            static_cast<double>(height) * IMAGE_SIZE / width
        );
    } else {
        new_height = IMAGE_SIZE;
        new_width = static_cast<int>(
            static_cast<double>(width) * IMAGE_SIZE / height
        );
    }

    cv::Mat resized;

    resized = pillowStyleResize(rgb, new_width, new_height);

    // Center crop to 224x224
    const int x = (new_width - IMAGE_SIZE) / 2;
    const int y = (new_height - IMAGE_SIZE) / 2;

    cv::Rect crop_region(
        x, y, IMAGE_SIZE, IMAGE_SIZE
    );

    cv::Mat cropped = resized(crop_region);

    // Output tensor layout: NCHW, FP32, RGB
    std::vector<float> tensor(
        3 * IMAGE_SIZE * IMAGE_SIZE
    );

    const int plane_size = IMAGE_SIZE * IMAGE_SIZE;

    for (int row = 0; row < IMAGE_SIZE; ++row) {
        for (int col = 0; col < IMAGE_SIZE; ++col) {

            cv::Vec3b pixel = cropped.at<cv::Vec3b>(
                row, col
            );

            int offset = row * IMAGE_SIZE + col;

            for (int channel = 0; channel < 3; ++channel) {

                float value = static_cast<float>(
                    pixel[channel]
                ) / 255.0f;

                tensor[
                    channel * plane_size + offset
                ] = (
                    value - IMAGE_MEAN[channel]
                ) / IMAGE_STD[channel];
            }
        }
    }

    return tensor;
}

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::cerr
                << "Usage: " << argv[0]
                << " <engine> <text_embeddings.bin> <image1.jpg>"
                << " [image2.jpg ...]\n";
            return 1;
        }

        Logger logger;

        auto engineData = readBytes(argv[1]);

        std::unique_ptr<nvinfer1::IRuntime> runtime(
            nvinfer1::createInferRuntime(logger)
        );

        if (!runtime)
            throw std::runtime_error("Runtime creation failed");

        std::unique_ptr<nvinfer1::ICudaEngine> engine(
            runtime->deserializeCudaEngine(
                engineData.data(), engineData.size()
            )
        );

        if (!engine)
            throw std::runtime_error("Engine loading failed");

        std::unique_ptr<nvinfer1::IExecutionContext> context(
            engine->createExecutionContext()
        );

        if (!context ||
            !context->setInputShape(
                "pixel_values", nvinfer1::Dims4{1,3,224,224}
            )) {
            throw std::runtime_error("Context setup failed");
        }

        if (engine->getTensorDataType("pixel_values")
                != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType("image_features")
                != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("Expected FP32 I/O tensors");
        }

        auto outShape = context->getTensorShape("image_features");
        if (outShape.nbDims != 2 ||
            outShape.d[0] != 1 || outShape.d[1] != 512) {
            throw std::runtime_error("Unexpected output dimensions");
        }

        auto text = readFloats(argv[2], 3 * 512);
        const char* labels[] = {"daytime", "night", "dawn/dusk"};

        void* deviceInput = nullptr;
        void* deviceOutput = nullptr;
        cudaStream_t stream = nullptr;
        cudaEvent_t gpuStart = nullptr;
        cudaEvent_t gpuEnd = nullptr;

        const size_t inputBytes = 3 * 224 * 224 * sizeof(float);
        const size_t outputBytes = 512 * sizeof(float);

        cudaCheck(cudaMalloc(&deviceInput, inputBytes));
        cudaCheck(cudaMalloc(&deviceOutput, outputBytes));
        cudaCheck(cudaStreamCreate(&stream));
        cudaCheck(cudaEventCreate(&gpuStart));
        cudaCheck(cudaEventCreate(&gpuEnd));

        if (!context->setTensorAddress(
                "pixel_values", deviceInput) ||
            !context->setTensorAddress(
                "image_features", deviceOutput)) {
            throw std::runtime_error("Tensor binding failed");
        }

        // CSV output supports automated evaluation
        std::cout << std::fixed << std::setprecision(8);
        std::cout
            << "index,prediction,s_daytime,s_night,s_dawn_dusk,"
            << "preprocess_ms,gpu_ms,end_to_end_ms\n";

        for (int imageIndex = 3; imageIndex < argc; ++imageIndex) {
            using Clock = std::chrono::steady_clock;

            auto totalStart = Clock::now();

            auto prepStart = Clock::now();
            auto input = preprocess(argv[imageIndex]);
            auto prepEnd = Clock::now();

            if (input.size() != 3 * 224 * 224)
                throw std::runtime_error("Wrong input size");

            cudaCheck(cudaMemcpyAsync(
                deviceInput, input.data(), inputBytes,
                cudaMemcpyHostToDevice, stream
            ));

            cudaCheck(cudaEventRecord(gpuStart, stream));

            if (!context->enqueueV3(stream))
                throw std::runtime_error("TensorRT execution failed");

            cudaCheck(cudaEventRecord(gpuEnd, stream));

            std::vector<float> embedding(512);

            cudaCheck(cudaMemcpyAsync(
                embedding.data(), deviceOutput, outputBytes,
                cudaMemcpyDeviceToHost, stream
            ));

            cudaCheck(cudaStreamSynchronize(stream));

            float gpuMs = 0.0f;
            cudaCheck(cudaEventElapsedTime(
                &gpuMs, gpuStart, gpuEnd
            ));

            double norm = 0.0;
            for (float v : embedding)
                norm += double(v) * double(v);

            norm = std::sqrt(norm);

            if (!std::isfinite(norm) || norm <= 0.0)
                throw std::runtime_error("Invalid embedding norm");

            double scores[3] = {0.0, 0.0, 0.0};

            for (int c = 0; c < 3; ++c) {
                for (int k = 0; k < 512; ++k) {
                    scores[c] += (
                        double(embedding[k]) / norm
                    ) * double(text[c * 512 + k]);
                }
            }

            int prediction = int(
                std::max_element(scores, scores + 3) - scores
            );

            auto totalEnd = Clock::now();

            double preprocessingMs =
                std::chrono::duration<double, std::milli>(
                    prepEnd - prepStart
                ).count();

            double totalMs =
                std::chrono::duration<double, std::milli>(
                    totalEnd - totalStart
                ).count();

            std::cout
                << (imageIndex - 3) << ","
                << labels[prediction] << ","
                << scores[0] << ","
                << scores[1] << ","
                << scores[2] << ","
                << preprocessingMs << ","
                << gpuMs << ","
                << totalMs << "\n";
        }

        cudaCheck(cudaEventDestroy(gpuStart));
        cudaCheck(cudaEventDestroy(gpuEnd));
        cudaCheck(cudaStreamDestroy(stream));
        cudaCheck(cudaFree(deviceInput));
        cudaCheck(cudaFree(deviceOutput));

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
