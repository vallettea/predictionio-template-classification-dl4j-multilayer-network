package org.template.classification

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.SparkContext
import grizzled.slf4j.Logger


import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;

case class AlgorithmParams(mult: Int) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {


  def train(sc: SparkContext, data: PreparedData): Model = {
        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10;

        val numInputs = 4;
        val outputNum = 3;
        val numSamples = 150;
        val batchSize = 150;
        val iterations = 100;
        val seed = 6L;
        val listenerFreq = iterations/5;

        println("Load data....");
        val iter = new IrisDataSetIterator(batchSize, numSamples);

        println("Build model....");
        val conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(1e-3)
                .l1(0.3).regularization(true).l2(1e-3)
                .constrainGradientToUnitNorm(true)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(2)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();

        val model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(listenerFreq));

        println("Train model....");
        while(iter.hasNext()) {
            val iris = iter.next();
            println(iris)
            iris.normalizeZeroMeanZeroUnitVariance();
            model.fit(iris);
        }
        iter.reset();

        // println("Evaluate weights....");
        // for(org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
        //     val w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
        //     println("Weights: " + w);
        // }


        println("Evaluate model....");
        val eval = new Evaluation(outputNum);
        val iterTest = new IrisDataSetIterator(numSamples, numSamples);
        val test = iterTest.next();
        test.normalizeZeroMeanZeroUnitVariance();
        val output = model.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        println(eval.stats());
        println("****************Example finished********************");

    new Model(model)
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val features = Array(
      query.sepal_length,
      query.sepal_width,
      query.petal_length,
      query.petal_width
    )
    val output = model.dbn.predict(Nd4j.create(features))
    val labelNames = Array("Iris-setosa", "Iris-versicolor", "Iris-virginica")
    new PredictedResult(labelNames(output(0)))
  }
}

class Model(val dbn: MultiLayerNetwork) extends Serializable {
  override def toString = s"dbn=${dbn}"
}

