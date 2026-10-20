// SPDX-License-Identifier: Apache-2.0
#include "common/tts_service.h"

#include <iostream>
#include <typeinfo>

#include "mlx/mlx.h"

namespace tts_service {

SynthesisWorker::SynthesisWorker(std::function<void()> load,
                                 std::function<void()> unload) {
  std::promise<void> loaded;
  auto ready = loaded.get_future();
  thread_ = std::thread(&SynthesisWorker::Run, this, std::move(load),
                        std::move(unload), std::move(loaded));
  try {
    ready.get();
  } catch (...) {
    thread_.join();
    throw;
  }
}

SynthesisWorker::~SynthesisWorker() {
  Stop();
  thread_.join();
}

void SynthesisWorker::Run(std::function<void()> load,
                          std::function<void()> unload,
                          std::promise<void> loaded) {
  try {
    load();
    loaded.set_value();
  } catch (...) {
    unload();
    loaded.set_exception(std::current_exception());
    return;
  }
  while (true) {
    std::shared_ptr<Task> job;
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [&] { return stopping_ || job_; });
      if (!job_) {
        break;
      } else {
        job = job_;
      }
    }
    std::exception_ptr error;
    try {
      const auto check_cancelled = [&] {
        if (job->cancelled.load()) {
          throw SynthesisCancelled();
        } else {
        }
      };
      check_cancelled();
      job->Synthesize(check_cancelled);
      check_cancelled();
    } catch (const SynthesisCancelled &) {
      std::cerr << "synthesis cancelled\n";
      error = std::current_exception();
    } catch (...) {
      error = std::current_exception();
    }
    try {
      mlx::core::synchronize();
      mlx::core::clear_cache();
    } catch (const std::exception &cleanup_error) {
      std::cerr << "synthesis cleanup failed: " << typeid(cleanup_error).name()
                << '\n';
      error = std::current_exception();
    }
    {
      std::lock_guard lock(mutex_);
      job_.reset();
    }
    job->Complete(error);
  }
  unload();
}

SynthesisWorker::Admission
SynthesisWorker::SubmitTask(const std::shared_ptr<Task> &job) {
  {
    std::lock_guard lock(mutex_);
    if (stopping_) {
      return Admission::Stopping;
    } else if (job_) {
      return Admission::Busy;
    } else {
      job_ = job;
    }
  }
  wake_.notify_one();
  return Admission::Accepted;
}

std::map<std::string, int> SynthesisWorker::RequestStates() const {
  std::lock_guard lock(mutex_);
  if (job_) {
    return {{"running", 1}};
  } else {
    return {};
  }
}

void SynthesisWorker::Stop() {
  {
    std::lock_guard lock(mutex_);
    stopping_ = true;
    if (job_) {
      job_->cancelled.store(true);
    } else {
    }
  }
  wake_.notify_all();
}

} // namespace tts_service
