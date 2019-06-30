#include <iostream>
#include <memory>
#include <map>
#include <stdexcept>
#include "crow_all.h"

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

typedef struct {
  std::map<std::string, std::string> header;
  std::string body;
} part;

static void
trim_string(std::string& s, const char* cutsel = " \t\v\r\n") {
  auto left = s.find_first_not_of(cutsel);
  if (left != std::string::npos) {
    auto right = s.find_last_not_of(cutsel);
    s = s.substr(left, right - left + 1);
  }
}

static std::vector<std::string>
split_string(std::string& s, const std::string& sep, int n = -1) {
  std::vector<std::string> result;
  auto pos = std::string::size_type(0);
  while (n != 0) {
    auto next = s.find(sep, pos);
    if (next == std::string::npos) {
      result.push_back(s.substr(pos));
      break;
    }
    result.push_back(s.substr(pos, next - pos));
    pos = next + sep.length();
    --n;
  }
  return result;
}

static std::map<std::string, std::string>
parse_header(std::string& lines) {
  std::map<std::string, std::string> result;
  std::string::size_type pos;
  for (auto& line : split_string(lines, "\r\n")) {
    auto token = split_string(line, ":", 2);
    if (token.size() != 2)
      break;
    trim_string(token[0]);
    std::transform(token[0].begin(), token[0].end(), token[0].begin(), ::tolower);
    trim_string(token[1]);
    result[token[0]] = token[1];
  }
  return result;
}

static std::vector<part>
parse_multipart(const crow::request& req, crow::response& res) {
  auto ct = req.get_header_value("content-type");
  auto pos = ct.find("boundary=");

  std::vector<part> result;
  if (pos != std::string::npos) {
    auto boundary = "\n--" + ct.substr(pos + 9);
    pos = boundary.find(";");
    if (pos != std::string::npos)
      boundary = boundary.substr(0, pos);
    pos = 0;
    auto& body = req.body;
    while (true) {
      auto next = body.find(boundary, pos);
      if (next == std::string::npos)
        break;
      auto data = body.substr(pos + boundary.size() + 2, next);
      auto eos = data.find("\r\n\r\n");
      if (eos == std::string::npos)
        break;
      auto lines = data.substr(0, eos);
      part p = {
        .header = parse_header(lines),
        .body = data = data.substr(eos + 4)
      };
      result.push_back(p);
      pos = next + boundary.size() + 1;
      if (body.at(pos) == '-' && body.at(pos + 1) == '-'
          && body.at(pos + 2) == '\n')
        break;
      else if (body.at(pos) != '\n')
        break;
    }
  }
  return result;
}

static crow::json::wvalue
detect_object(
    std::unique_ptr<tflite::Interpreter>& interpreter,
    std::string& data,
    int nresults,
    std::vector<std::string>& labels) {
  TfLiteStatus status;
  std::vector<uchar> buf;
  buf.assign(data.data(), data.data() + data.size());
  cv::Mat frame = cv::imdecode(cv::Mat(buf), 1);
  if (frame.empty()){
    std::cerr << "ERROR: cv::imdecode" << std::endl;
    return {};
  }
  int input = interpreter->inputs()[0];
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];
  int wanted_type = interpreter->tensor(input)->type;

  cv::Mat resized(wanted_height, wanted_width, frame.type());
  cv::resize(frame, resized, resized.size(), cv::INTER_LINEAR);

  if (wanted_type == kTfLiteFloat32) {
    int n = 0, nc = resized.channels(), ne = resized.elemSize();
    if (nc > wanted_channels) nc = wanted_channels;
    uint8_t *in8 = interpreter->typed_tensor<uint8_t>(input);
    for (int y = 0; y < resized.rows; ++y)
      for (int x = 0; x < resized.cols; ++x)
        for (int c = 0; c < nc; ++c)
          in8[n++] = (float)resized.data[(y * resized.cols + x) * ne + c] / 255.0;
  } else if (wanted_type == kTfLiteUInt8) {
    int n = 0, nc = resized.channels(), ne = resized.elemSize();
    if (nc > wanted_channels) nc = wanted_channels;
    uint8_t *in8 = interpreter->typed_tensor<uint8_t>(input);
    for (int y = 0; y < resized.rows; ++y)
      for (int x = 0; x < resized.cols; ++x)
        for (int c = 0; c < nc; ++c)
          in8[n++] = (uint8_t)resized.data[(y * resized.cols + x) * ne + c];
  }

  status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    std::cerr << "ERROR: interpreter->Invoke" << std::endl;
    return {};
  }

  int output = interpreter->outputs()[0];
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  auto output_size = output_dims->data[output_dims->size - 1];
  int output_type = interpreter->tensor(output)->type;

  std::vector<std::pair<float, int>> r;

  if (wanted_type == kTfLiteFloat32) {
    float *scores = interpreter->typed_output_tensor<float>(0);
    for (int i = 0; i < output_size; ++i) {
      float value = scores[i];
      if (value < 0.2)
        continue;
      r.push_back(std::pair<float, int>(value, i));
    }
  } else if (wanted_type == kTfLiteUInt8) {
    uint8_t *scores = interpreter->typed_output_tensor<uint8_t>(0);
    for (int i = 0; i < output_size; ++i) {
      float value = ((float)scores[i]) / 255.0;
      if (value < 0.2)
        continue;
      r.push_back(std::pair<float, int>(value, i));
    }
  }
  std::sort(r.begin(), r.end(),
    [](std::pair<float, int>& x, std::pair<float, int>& y) -> int {
      return x.first > y.first;
    }
  );

  crow::json::wvalue out;
  for (auto i = 0; i < nresults && i < r.size(); ++i) {
    out[i]["index"] = r[i].second;
    out[i]["probability"] = r[i].first;
    out[i]["label"] = labels[r[i].second];
  }
  return out;
}

int
main() {
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model;
  TfLiteStatus status;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  model = tflite::FlatBufferModel::BuildFromFile("mobilenet_quant_v1_224.tflite");
  if (!model) {
    std::cerr << "ERROR: failed to load model file." << std::endl;
    return -1;
  }
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "ERROR: failed to allocate the memory for tensors." << std::endl;
    return -2;
  }

  interpreter->SetNumThreads(4);
  interpreter->UseNNAPI(0);

  std::ifstream file("labels.txt");
  if (!file) {
    std::cerr << "ERROR: failed to read label file" << std::endl;
    return -3;
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(file, line))
    labels.push_back(line);
  while (labels.size() % 16)
    labels.emplace_back();

  crow::SimpleApp app;

  CROW_ROUTE(app, "/upload")
      .methods("POST"_method)
  ([&](const crow::request& req, crow::response& res) {
    auto result = parse_multipart(req, res);
    for (auto& x : result) {
      auto objects = detect_object(interpreter, x.body, 5, labels);
      if (objects.estimate_length()) {
        res.write(crow::json::dump(objects));
        break;
      }
    }
    res.end();
  });
  app.port(8888).run();
}
