package org.template.classification

import io.prediction.controller.PDataSource
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EmptyActualResult
import io.prediction.controller.Params
import io.prediction.data.storage.{PropertyMap, Storage}
import org.apache.spark.SparkContext
import grizzled.slf4j.Logger
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.apache.spark.rdd.RDD

case class DataSourceParams(appId: Int) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {

    // read from file
    val batchSize = 150
    val textFile = sc.textFile("data/iris.data")
    val columns = "sepal-length,sepal-width,petal-length,petal-width,species".split(",").zipWithIndex.map{ case (name,index) => name -> index }.toMap



    new TrainingData(textFile
      .glom()
      .map(batch => {

        val features: INDArray = Nd4j.zeros(batchSize, 4)
        val labels: INDArray = Nd4j.zeros(batchSize, 3)

        batch.zipWithIndex
        .foreach { case (line, row) =>
            val linesplit = line.split(",")

            val feature = Nd4j.create(
              Array(linesplit(columns("sepal-length")).toDouble,
                linesplit(columns("sepal-width")).toDouble,
                linesplit(columns("petal-length")).toDouble,
                linesplit(columns("petal-width")).toDouble
              )
            )
            features.putRow(row.toInt, feature)

            val label = Nd4j.create(
              linesplit(columns("species")) match {
                case "Iris-setosa" => Array(1.0, 0.0, 0.0)
                case "Iris-versicolor" => Array(0.0, 1.0, 0.0)
                case "Iris-virginica" => Array(0.0, 0.0, 1.0)
              }
            )
            labels.putRow(row.toInt, label)

        }
      val data = new DataSet(features, labels)
      data.normalizeZeroMeanZeroUnitVariance()
      data.shuffle()
      data
      })
    )
  }

}

class TrainingData(
  val data: RDD[DataSet]
) extends Serializable {
  override def toString = {
    s"events: dataSetIterator"
  }
}
