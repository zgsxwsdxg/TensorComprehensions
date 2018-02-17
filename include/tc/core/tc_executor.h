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

#include "tc/core/halide2pencil.h"
#include "tc/core/mapping_options.h"
#include "tc/core/utils/dlpack.h"
#include "tc/lang/parser.h"

namespace tc {

void checkSizesAndStridesAreCompliant(
    const DLTensor* actual,
    const DLTensor* expected,
    const lang::Param& dbg);

// templating to match both const and non-const DLTensor pointers
template <typename T>
void checkSizesAndStridesAreCompliant(
    const std::vector<T*>& dlTensors,
    const std::vector<dlutils::DLTensorUPtr>& tensorInfos,
    const lang::ListView<lang::Param>& dbgInfo) {
  if (tensorInfos.size() != dlTensors.size()) {
    throw lang::ErrorReport(dbgInfo)
        << "expected " << tensorInfos.size() << " values but found "
        << dlTensors.size();
  }
  for (size_t i = 0; i < tensorInfos.size(); ++i) {
    checkSizesAndStridesAreCompliant(
        dlTensors[i], tensorInfos[i].get(), dbgInfo[i]);
  }
}

class TcExecutor {
 public:

  struct TcExecutionInfo {
   public:
    std::string kernelName;
    std::vector<dlutils::DLTensorUPtr> inputsInfo;
    std::vector<dlutils::DLTensorUPtr> outputsInfo;
    std::vector<int> kernelParams;
    std::string kernelSpecializedName;
    std::unique_ptr<tc::MappingOptions> options;
    std::string cudaSource;
    Grid grid{{0, 0, 0}};
    Block block{{0, 0, 0}};
    std::shared_ptr<CudaRTCFunction> rtcFun;
  };

  TcExecutor(
    const std::string& TCDefinition,
    const std::vector<const DLTensor*>& inputsInfo);
  TcExecutor(
    lang::TreeRef TCDefinition,
    const std::vector<const DLTensor*>& inputsInfo);

  virtual void compile(const std::string& options) = 0;

  // Given a Tc and a list of input tensors that match the definition in the
  // Tc in positional order, this generates the output tensor infos issued
  // from forward inference.
  // The typical flow is to infer output sizes, allocate/resize them within
  // you favorite ML framework / tensor library and then call compile.
  std::vector<const DLTensor*> inferOutputTensorInfo();

  HalidePencilState getHalidePencilState(
      const std::vector<const DLTensor*>& inTensorPtrs);

  virtual Duration run(
      const std::vector<const DLTensor*>& inputs,
      const std::vector<DLTensor*>& outputs,
      bool profile = false) const = 0;

  virtual void uncheckedRun(
      const std::vector<const void*>& inputs,
      const std::vector<void*>& outputs) const = 0;

  virtual bool hasRTCFun() const = 0;

  const static size_t InvalidHandle = std::numeric_limits<size_t>::max();

 protected:
  void checkInputsCompliant(
      const std::vector<const DLTensor*>& inputsInfo) const;
  tc2halide::HalideComponents halideComponents_;
  TcExecutionInfo execInfo_;
  lang::TreeRef tcTree_;
  mutable isl::ctx ctx_;
};

} // namespace tc
