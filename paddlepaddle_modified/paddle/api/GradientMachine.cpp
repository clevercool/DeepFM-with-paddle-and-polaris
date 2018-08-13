/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "PaddleAPI.h"
#include "PaddleAPIPrivate.h"

#include "Internal.h"
#include "paddle/gserver/gradientmachines/NeuralNetwork.h"
//#define PRINTD

std::vector<int> GradientMachine::defaultParamTypes = {
    PARAMETER_VALUE, PARAMETER_GRADIENT, PARAMETER_MOMENTUM};

GradientMachine::GradientMachine() : m(new GradientMachinePrivate()) {
#ifdef PRINTD
  printf("----GradientMachine::GradientMachine()----\n");
#endif
  }

GradientMachine::~GradientMachine() { 
#ifdef PRINTD
  printf("----GradientMachine::GradientMachine() delete m----\n");  
#endif
  delete m; 
  }

GradientMachine* GradientMachine::createFromPaddleModelPtr(  
    const void* confPtr,
    GradientMatchineCreateMode mode,
    const std::vector<int>& types) {
#ifdef PRINTD
  printf("----GradientMachine::createFromPaddleModelPtr()----\n");
#endif
  auto& conf = *(const paddle::ModelConfig*)(confPtr);
  std::vector<ParameterType> realTypes;
  staticCastVector(&realTypes, types);
  auto machineRawPtr = paddle::GradientMachine::create(conf, mode, realTypes);
  auto machinePtr = std::shared_ptr<paddle::GradientMachine>(machineRawPtr);
  if (machinePtr != nullptr) {
    auto machine = new GradientMachine();
    machine->m->machine = machinePtr;
    return machine;
  } else {
    return nullptr;
  }
}

GradientMachine* GradientMachine::createByConfigProtoStr(
    const std::string& protoStr,
    GradientMatchineCreateMode mode,
    const std::vector<int>& types) {
#ifdef PRINTD
  printf("----GradientMachine::createByConfigProtoStr()----\n");
#endif
  paddle::ModelConfig conf;
  conf.ParseFromString(protoStr);
  if (conf.IsInitialized()) {
    return GradientMachine::createFromPaddleModelPtr(&conf, mode, types);
  } else {
    return nullptr;
  }
}

GradientMachine* GradientMachine::createByModelConfig(  
    ModelConfig* conf,
    GradientMatchineCreateMode mode,
    const std::vector<int>& types) {
#ifdef PRINTD
  printf("----GradientMachine::createByModelConfig()----\n");
#endif
  auto confPtr = &conf->m->conf->getModelConfig();
  return GradientMachine::createFromPaddleModelPtr(confPtr, mode, types);
}

void GradientMachine::start() { 
#ifdef PRINTD
  printf("----GradientMachine::start()----\n");
#endif
  m->machine->start(); 
  
  }

void GradientMachine::finish() {  
#ifdef PRINTD
  printf("----GradientMachine::finish()----\n");
#endif
   m->machine->finish(); }

void GradientMachine::onPassEnd() {  
#ifdef PRINTD
  printf("----GradientMachine::onPassEnd()----\n");
#endif
  m->machine->onPassEnd(); }

void GradientMachine::prefetch(const Arguments& inArgs) {  
#ifdef PRINTD
  printf("----GradientMachine::prefetch()----\n");
#endif
  auto& in =
      m->cast<std::vector<paddle::Argument>>(inArgs.getInternalArgumentsPtr());
  m->machine->prefetch(in);
}

void GradientMachine::forward(const Arguments& inArgs,
                              Arguments* outArgs,
                              PassType passType) {
#ifdef PRINTD                                
  printf("----GradientMachine::forward()----\n");
#endif
  auto& in =
      m->cast<std::vector<paddle::Argument>>(inArgs.getInternalArgumentsPtr());
  auto& out = m->cast<std::vector<paddle::Argument>>(
      outArgs->getInternalArgumentsPtr());
  paddle::PassType pt = (paddle::PassType)(passType);
  m->machine->forward(in, &out, pt);
}

UpdateCallback::~UpdateCallback() {}

void UpdateCallback::apply(Parameter* p) {
  // UNUSED(p);
}

class UpdateCallbackWrapper {
 public:
  explicit UpdateCallbackWrapper(const UpdateCallback& callback)
      : callback(const_cast<UpdateCallback&>(callback)) {}

  void operator()(paddle::Parameter* param) {
    auto p = Parameter::createFromRawPtr(&param);
    // @TODO Use Stack variable instead.
    callback.apply(p);
    delete p;
  }

 private:
  UpdateCallback& callback;
};

void GradientMachine::backward(const UpdateCallback& callback) {  
#ifdef PRINTD
  printf("----GradientMachine::backward()----\n");
#endif
  m->machine->backward(UpdateCallbackWrapper(callback));
}

void GradientMachine::forwardBackward(const Arguments& inArgs,
                                      Arguments* outArgs,
                                      PassType passType,
                                      const UpdateCallback& callback) {                                      
#ifdef PRINTD
  printf("----GradientMachine::forwardBackward()----\n");
#endif
  auto& in =
      m->cast<std::vector<paddle::Argument>>(inArgs.getInternalArgumentsPtr());
  auto& out = m->cast<std::vector<paddle::Argument>>(
      outArgs->getInternalArgumentsPtr());
  paddle::PassType pt = (paddle::PassType)(passType);
  m->machine->forwardBackward(in, &out, pt, UpdateCallbackWrapper(callback));
}

void GradientMachine::loadParameters(const std::string& path) {
#ifdef PRINTD
  printf("----GradientMachine::loadParameters()----\n");
#endif
  m->machine->loadParameters(path);
}

size_t GradientMachine::getParameterSize() const {
#ifdef PRINTD
  printf("----GradientMachine::getParameterSize()----\n");
#endif
  return m->machine->getParameters().size();
}

Parameter* GradientMachine::getParameter(size_t i) throw(RangeError) {
#ifdef PRINTD
  printf("----GradientMachine::getParameter()----\n");
#endif
  auto params = m->machine->getParameters();
  if (i < params.size()) {
    return Parameter::createFromSharedPtr(&m->machine->getParameters()[i]);
  } else {
    throw RangeError();
  }
}

size_t GradientMachine::getNonStaticParameterSize() const {
#ifdef PRINTD
  printf("----GradientMachine::getNonStaticParameterSize()----\n");
#endif
  return m->machine->getNonStaticParameters().size();
}

Parameter* GradientMachine::getNonStaticParameter(size_t i) throw(RangeError) {
#ifdef PRINTD
  printf("----GradientMachine::getNonStaticParameter()----\n");
#endif
  auto params = m->machine->getNonStaticParameters();
  if (i < params.size()) {
    return Parameter::createFromSharedPtr(
        &m->machine->getNonStaticParameters()[i]);
  } else {
    throw RangeError();
  }
}

void GradientMachine::randParameters() { 
#ifdef PRINTD
  printf("----GradientMachine::randParameters()----\n");
#endif
  m->machine->randParameters(); }

Arguments* GradientMachine::getLayerOutput(const std::string& layerName) const
    throw(UnsupportError) {
#ifdef PRINTD
  printf("----GradientMachine::getLayerOutput()----\n");
#endif
  auto nn = m->machine;
  if (nn) {
    auto arg = nn->getLayerOutput(layerName);
    return Arguments::createByPaddleArgument(&arg);
  } else {
    throw UnsupportError();
  }
}

SequenceGenerator* GradientMachine::asSequenceGenerator(
    const std::vector<std::string>& dict,
    size_t begin_id,
    size_t end_id,
    size_t max_length,
    size_t beam_size) {
#ifdef PRINTD
  printf("----GradientMachine::asSequenceGenerator()----\n");
#endif
  SequenceGenerator* r =
      SequenceGenerator::createByGradientMachineSharedPtr(&m->machine);
  r->setDict(dict);
  r->setBos(begin_id);
  r->setEos(end_id);
  r->setMaxLength(max_length);
  r->setBeamSize(beam_size);
  return r;
}

Evaluator* GradientMachine::makeEvaluator() {
#ifdef PRINTD
  printf("----GradientMachine::makeEvaluator()----\n");
#endif
  auto ev = new Evaluator();
  ev->m->rawPtr = m->machine->makeEvaluator();
  return ev;
}

void GradientMachine::eval(Evaluator* evaluator) {
#ifdef PRINTD
  printf("----GradientMachine::eval()----\n");
#endif
  m->machine->eval(evaluator->m->rawPtr);
}
