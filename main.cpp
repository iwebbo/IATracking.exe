#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace dnn;
using namespace std;

int main() {
    string model_path = "yolov8.onnx";  // Please to be sure ONNX is present on the same path folder

    // Load model
    Net net = readNetFromONNX(model_path);
    net.setPreferableBackend(DNN_BACKEND_CUDA);  //  CUDA
    net.setPreferableTarget(DNN_TARGET_CUDA_FP16);  // Précision FP16 

    cout << "Model succefully loaded by CUDA." << endl;

    return 0;
}
