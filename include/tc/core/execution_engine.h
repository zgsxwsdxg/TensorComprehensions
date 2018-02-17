/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <dlpack/dlpack.h>

#include "tc/core/tc_executor.h"
#include "tc/core/utils/dlpack.h"
#include "tc/lang/parser.h"

namespace tc {

class ExecutionEngine {
 public:

  struct ExecutorInfo {
   public:
    ExecutorInfo(
      std::string id,
      std::vector<const DLTensor*> inputsInfo,
      std::string options,
      lang::TreeRef tc,
      size_t handle) :
        identifier(id),
        inputsInfo(tc::dlutils::makeDLTensorVector(inputsInfo)),
        mappingOptions(options),
        // exec(tc, inputsInfo),
        objectLocalHandle(handle)
      {}

    virtual ~ExecutorInfo() {};
    virtual void clear() {
      CHECK(false) << "NYI: should be abstract, needs more refactoring";
    };

    std::string identifier;
    std::vector<tc::dlutils::DLTensorUPtr> inputsInfo;
    std::string mappingOptions;
    /// When run is called this is used to find the most recently compiled
    /// version.
    size_t objectLocalHandle;
    std::unique_ptr<TcExecutor> exec;
  };

  ExecutionEngine() {}
  virtual ~ExecutionEngine() {}

  /// Create the CudaExecutionEngine::tcNameMap_ using the language passed
  /// to it - should support many TC.
  void define(const std::string& language);

  /// Create the CudaExecutionEngine::tcNameMap_ from the parsed TC
  /// string - supports many TC.
  void define(const std::vector<lang::TreeRef>& treeRefs);

  /// Get the output Tensor info that can be used by the calling framework to
  /// allocate storage for the output.
  std::vector<const DLTensor*> inferOutputTensorInfo(
    const std::string& name,
    const std::vector<const DLTensor*>& inTensorPtrs);

  /// Returns a handle for the compiled kernel
  virtual size_t compile(
    const std::string& name,
    const std::vector<const DLTensor*>& inputs,
    const std::string& mappingOptions) = 0;

  virtual Duration run(
      size_t handle,
      const std::vector<const DLTensor*>& inputs,
      const std::vector<DLTensor*>& outputs,
      bool profile) = 0;

  virtual Duration run(
    size_t handle,
    const std::vector<const DLTensor*>& inputs,
    const std::vector<DLTensor*>& outputs,
    bool profile,
    std::function<bool(const ExecutorInfo*)> pruningFunction) = 0;

  /// This is the "low-latency" mode in which we just propagate raw pointers to
  /// data in the target address space.
  /// No tensor-related information can be checked so it is the user's
  /// responsibility to ensure that shapes and strides match. If the user
  /// doesn't then segfault will likely occur.
  virtual void uncheckedRun(
      size_t handle,
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) = 0;

  virtual void clear(size_t handle) = 0;

  size_t emplaceExecutor(std::unique_ptr<ExecutorInfo> p);

 protected:
  /// For thread-safety perform all cheap operations under lock
  std::mutex executorInfoMutex;

  // XXX:if ExecutorInfo is moved/copied (even when the vector's underlying
  // storage is extended) something inside isl segfaults,  unique_ptr is used as
  // a workaround
  std::vector<std::unique_ptr<ExecutorInfo>> executors_;
  std::map<std::string, lang::TreeRef> tcNameMap_;

  size_t uidCounter = 0;
};

} // namespace tc
