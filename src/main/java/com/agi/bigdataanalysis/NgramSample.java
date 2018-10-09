package com.agi.bigdataanalysis;

import java.util.Arrays;
import java.util.List;

/*import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;*/
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class NgramSample {

	public static void main(String[] args) {
		/*SparkConf sparkConf = new SparkConf()
			.setAppName("Ngram");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);*/
		SparkSession spark = SparkSession.builder()
			.appName("CollarberativeFilter").getOrCreate();

		List<Row> data = Arrays.asList(
		  RowFactory.create(0, Arrays.asList("Hi", "I", "heard", "about", "Spark")),
		  RowFactory.create(1, Arrays.asList("I", "wish", "Java", "could", "use", "case", "classes")),
		  RowFactory.create(2, Arrays.asList("Logistic", "regression", "models", "are", "neat"))
		);

		StructType schema = new StructType(new StructField[]{
		  new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
		  new StructField(
		    "words", DataTypes.createArrayType(DataTypes.StringType), false, Metadata.empty())
		});

		Dataset<Row> wordDataFrame = spark.createDataFrame(data, schema);

		NGram ngramTransformer = new NGram().setN(2).setInputCol("words").setOutputCol("ngrams");

		Dataset<Row> ngramDataFrame = ngramTransformer.transform(wordDataFrame);
		ngramDataFrame.select("ngrams").show(false);
		
		/**
		 * +------------------------------------------------------------------+
		   |ngrams                                                            |
		   +------------------------------------------------------------------+
		   |[Hi I, I heard, heard about, about Spark]                         |
		   |[I wish, wish Java, Java could, could use, use case, case classes]|
		   |[Logistic regression, regression models, models are, are neat]    |
		   +------------------------------------------------------------------+
		 */
	}

}
