// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>

#include "nlohmann/json.hpp"

#include "ming_tts_audio.h"
#include "ming_tts_generator.h"
#include "service.h"
#include "tts_service.h"

struct mg_connection;

namespace ming_tts {

struct SpeechRequest {
  std::string text;
  std::string prompt;
  std::string instructions;
  bool stream;
  bool wav;
  int max_new_tokens;
  float guidance_scale;
  float sigma;
  float temperature;
  std::vector<float> reference_samples;
  std::string reference_text;
  static SpeechRequest Parse(const nlohmann::json &request);
};

class MingSpeechService : public omni_server::Service {
public:
  explicit MingSpeechService(const std::filesystem::path &directory);
  void Register(mg_context *context, const std::string &model_name) override;
  std::map<std::string, int> RequestStates() const override;
  void CancelAll() override;

private:
  struct Job {
    mg_connection *connection;
    SpeechRequest request;
    bool headers_sent = false;
    std::string audio;
    int sample_rate = 0;
  };
  int HandleSpeech(mg_connection *connection);
  void Generate(Job &job, const std::function<void()> &check_cancelled);
  std::filesystem::path directory_;
  std::unique_ptr<MingFrontend> frontend_;
  std::unique_ptr<MingGenerator> generator_;
  std::unique_ptr<MingAudioDecoder> decoder_;
  std::unique_ptr<MingReferenceEncoder> reference_encoder_;
  tts_service::SynthesisWorker worker_;
};

} // namespace ming_tts
