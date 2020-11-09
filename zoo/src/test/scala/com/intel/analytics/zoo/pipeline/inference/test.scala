package com.intel.analytics.zoo.pipeline.inference

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.common.{PythonInterpreter, PythonInterpreterTest}
import com.intel.analytics.zoo.core.TFNetNative
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.net.TorchModel
import org.apache.log4j.{Level, Logger}
import com.intel.analytics.zoo.common.NNContext.initNNContext
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FlatSpec

import scala.language.postfixOps

class test extends FlatSpec {
  val model = new InferenceModel(4)
  "PyTorch Model" should "be loaded and do predict" in {
    model.doLoadPyTorch("/Users/liyuangu/Desktop/Intel/torch_resnet50/SimpleTorchModel.pt")
    assert(model != null)
    val inputTensor = Tensor[Float](1, 2).rand()
    val exceptedResult = inputTensor.valueAt(1, 1) * 0.2f +
      inputTensor.valueAt(1, 2) * 0.5f + 0.3f
    val r = model.doPredict(inputTensor)
    assert(r == Tensor[Float](Array(exceptedResult), Array(1, 1)))
    
  }
}
