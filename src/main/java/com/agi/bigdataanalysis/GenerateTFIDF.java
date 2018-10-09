package com.agi.bigdataanalysis;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.agi.bigdata.utils.FileParseUtils;
import com.agi.bigdata.utils.ReportGenerator;

public class GenerateTFIDF {

	static final Logger logger = LoggerFactory.getLogger("agiLogger");

	public void tfidf(SparkSession spark, String hadoopPath) {
		List<Row> corpusList = new ArrayList<Row>();
		Instant startTime = Instant.now();

		ArrayList<String> fileList = new ArrayList<String>();

		Configuration config = new Configuration();
		FileParseUtils fileParseUtils = new FileParseUtils();

		try {
			FileSystem fs = FileSystem.get(config);
			Path folderPath = new Path(hadoopPath);
			int documentIndex = 0;

			if (!fs.exists(folderPath) || !fs.isDirectory(folderPath)) {
				logger.info("Not a valid path");
				return;
			}
			RemoteIterator<LocatedFileStatus> iterator = fs.listFiles(folderPath, true);
			while (iterator.hasNext()) {
				Path filePath = iterator.next().getPath();
				String textContent = fileParseUtils.getFileContentAsString(filePath);
				if (textContent.isEmpty()) {
					continue;
				}
				logger.info(documentIndex++ + "\t" + filePath);
				corpusList.add(RowFactory.create(documentIndex++, textContent));
			}
		} catch (IOException ioEx) {
			logger.error("There was some exception while reading file contents ", ioEx);
		}

		StructType schema = new StructType(new StructField[] {
				new StructField("label", DataTypes.IntegerType, false, Metadata.empty()),
				new StructField("docwords", DataTypes.StringType, false, Metadata.empty())
		});

		Dataset<Row> sentenceData = spark.createDataFrame(corpusList, schema);

		logger.info("Tokenising the corpus");
		Tokenizer tokenizer = new Tokenizer().setInputCol("docwords").setOutputCol("words");
		Dataset<Row> wordsData = tokenizer.transform(sentenceData);

		logger.info("Filtering the corpus using default skip words");
		StopWordsRemover remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered");

		Dataset<Row> filteredData = remover.transform(wordsData);

		logger.info("Appyling custom remove filter");
		StopWordsRemover customFilterWords = new StopWordsRemover().setInputCol("filtered").setOutputCol("filteredAgain");

		// This could be used to filter custom stop words
		String[] removeUnwantedWords = { " ", "=", "se", "ï‚·" };
		customFilterWords.setStopWords(removeUnwantedWords);

		Dataset<Row> filteredDataAgain = customFilterWords.transform(filteredData);

		logger.info("Generating freequncy of words in each documents");
		// fit a CountVectorizerModel from the corpus
		CountVectorizerModel cvModel = new CountVectorizer().setInputCol("filteredAgain").setOutputCol("rawFeatures")
				.fit(filteredDataAgain);

		Dataset<Row> featurizedData = cvModel.transform(filteredDataAgain);
		String[] vocabularyWords = cvModel.vocabulary();

		List<Row> tfIdfResult = featurizedData.collectAsList();

		logger.info("Generating tf idf for the corpus");
		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
		IDFModel idfModel = idf.fit(featurizedData);

		Dataset<Row> rescaledData = idfModel.transform(featurizedData);

		logger.info("TF IDF generation completed");
		rescaledData.cache();

		List<Row> featuresRow = rescaledData.select("features").collectAsList();

		Instant endTime = Instant.now();
		String totalTimeInSeconds = String.valueOf(Duration.between(startTime, endTime).toString());
		logger.info("Time taken by the job is " + totalTimeInSeconds);

		ReportGenerator reportGenerator = new ReportGenerator();

		logger.info("Generating xml report");
		reportGenerator.generateXMLReportOfTFIDF(featuresRow, tfIdfResult,
				vocabularyWords, removeUnwantedWords, fileList,
				totalTimeInSeconds);

		logger.info("Generating html report");
		reportGenerator.generateHTMLFromXML();

		spark.stop();
	}

}
