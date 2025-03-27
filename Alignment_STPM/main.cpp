#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, "feature_extractor.onnx", session_options);

    cv::Mat img = cv::imread("fov_image.jpg");
    cv::resize(img, img, cv::Size(224, 224));
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // Normalize
    for (int i = 0; i < 3; i++)
        img.forEach<cv::Vec3f>([i](cv::Vec3f &pixel, const int * position) {
            pixel[i] = (pixel[i] - 0.5) / 0.5;
        });

    std::vector<float> input_tensor_values(img.total() * 3);
    std::memcpy(input_tensor_values.data(), img.data, input_tensor_values.size() * sizeof(float));

    // ONNX 추론 입력 설정
    std::array<int64_t, 4> input_shape{1, 3, 224, 224};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"features"};

    // 추론 실행
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names, &input_tensor, 1,
                                      output_names, 1);

    // Feature 벡터 추출
    float* features = output_tensors.front().GetTensorMutableData<float>();
    int dim = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape()[1];

    // Reference 벡터와 cosine similarity 계산 (예시)
    std::vector<float> reference(dim); // reference vector 로드해야 함
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += features[i] * reference[i];
        norm1 += features[i] * features[i];
        norm2 += reference[i] * reference[i];
    }
    float cosine_sim = dot / (sqrt(norm1) * sqrt(norm2));
    std::cout << "유사도 (cosine): " << cosine_sim << std::endl;

    // 정렬 위치 계산, offset 반영 등 후처리 구현 가능
    return 0;
}
