package com.agi.bigdataanalysis;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.MinHashLSH;
import org.apache.spark.ml.feature.MinHashLSHModel;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.RegexTokenizer;
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
import com.agi.bigdata.utils.GetJobConfig;
import com.agi.bigdata.utils.ReportGenerator;

public class TextSimilarity {

	static final Logger logger = LoggerFactory.getLogger("agiLogger");
     
	public void textSimilarityJob(SparkSession spark, String query, String hadoopPath, double threshold, boolean runOnPartialSets) {
		int numberOfPartialSets = GetJobConfig.getTextSimilarityPartialSetValue();
		if (runOnPartialSets) {
			logger.info("Running on partial sets enabled, Partial set length is " + numberOfPartialSets);
		}
		/**
		 * This is for threshold explanation..
		 * https://www.quora.com/What-is-the-
		 * threshold-tuning-in-metric-evaluation-in-Apache-Spark-Mllib
		 */
		RegexTokenizer tokenizer = new RegexTokenizer().setPattern("").setInputCol("docwords").setOutputCol("tokens");

		NGram ngram = new NGram().setN(GetJobConfig.getNGramValue()).setInputCol("tokens")
				.setOutputCol("ngrams");

		HashingTF hTf = new HashingTF().setInputCol("ngrams").setOutputCol(
				"vectors");

		// Increase numHashTables in practice.
		MinHashLSH minLsh = new MinHashLSH().setNumHashTables(GetJobConfig.getNumHashTables())
				.setInputCol("vectors").setOutputCol("lsh");

		Pipeline pline = new Pipeline().setStages(new PipelineStage[] {
				tokenizer, ngram, hTf, minLsh
		});

		StructType schema = new StructType(new StructField[] {
				new StructField("label", DataTypes.IntegerType, false,
						Metadata.empty()),
				new StructField("docwords", DataTypes.StringType, false,
						Metadata.empty())
		});

		List<Row> corpusList1 = new ArrayList<Row>();
		List<Row> corpusList2 = new ArrayList<Row>();

		corpusList1.add(RowFactory.create(0, query));

		int index = 0;

		Map<String, Double> resultMap = new HashMap<String, Double>();

		ArrayList<String> fileList = new ArrayList<String>();
		FileParseUtils fileParseUtils = new FileParseUtils();

		Configuration config = new Configuration();
		List<Path> filePathList = new ArrayList<Path>();

		try {
			FileSystem fs = FileSystem.get(config);
			Path folderPath = new Path(hadoopPath);

			if (!fs.exists(folderPath) || !fs.isDirectory(folderPath)) {
				logger.info("Not a valid path");
				return;
			}
			RemoteIterator<LocatedFileStatus> iterator = fs.listFiles(folderPath, true);
			while (iterator.hasNext()) {
				filePathList.add(iterator.next().getPath());
			}
		} catch (IOException ioEx) {
			logger.error("There was some exception while reading file contents ", ioEx);
		}

		for (Path filePath : filePathList) {
			String textContent = fileParseUtils.getFileContentAsString(filePath);
			// Check for empty files and files and invalid parser files
			if (textContent.isEmpty()) {
				continue;
			}

			corpusList2.add(RowFactory.create(index++, textContent));
			fileList.add(filePath.toUri().toString());

			logger.info(index + "\t" + filePath);
			if (runOnPartialSets) {
				if (index % numberOfPartialSets == 0) {
					calculateTextSimilarity(spark, corpusList2, corpusList1, schema, pline, threshold, resultMap, fileList);
					corpusList2 = new ArrayList<Row>();
				}
			}
		}

		// Processing the remaining files
		if (runOnPartialSets) {
			if (filePathList.size() % numberOfPartialSets != 0) {
				calculateTextSimilarity(spark, corpusList2, corpusList1, schema, pline, threshold, resultMap, fileList);
			}
		} else {
			calculateTextSimilarity(spark, corpusList2, corpusList1, schema, pline, threshold, resultMap, fileList);
		}

		logger.info("Successfully matched " + resultMap.size() + " files");

		logger.info("Generating Text Similarity report");
		ReportGenerator reportGenerator = new ReportGenerator();
		try {
			reportGenerator.generateTextSimilarityReport(query, hadoopPath, threshold, resultMap);
		} catch (IOException ioEx) {
			logger.error("There was some exception while generating the report", ioEx);
		}
	}

	void calculateTextSimilarity(SparkSession spark, List<Row> corpusList2, List<Row> corpusList1, StructType schema,
			Pipeline pline, double threshold, Map<String, Double> resultMap, ArrayList<String> fileList) {
		Dataset<Row> db = spark.createDataFrame(corpusList2, schema).toDF("label", "docwords");
		Dataset<Row> queryDataset = spark.createDataFrame(corpusList1, schema).toDF("label", "docwords");

		PipelineModel model = pline.fit(db);
		Dataset<Row> dbHashed = model.transform(db);
		Dataset<Row> queryHashed = model.transform(queryDataset);

		int modelLength = model.stages().length;
		MinHashLSHModel lshModel = (MinHashLSHModel) model.stages()[modelLength - 1];

		List<Row> result = lshModel.approxSimilarityJoin(dbHashed, queryHashed, Double.valueOf(threshold))
				.select("datasetA", "distCol").collectAsList();

		for (Row row : result) {
			double percentage = 1 - row.getDouble(1);
			Row r1 = (Row) row.get(0);
			resultMap.put(fileList.get(Integer.parseInt(r1.get(0).toString())), percentage);
		}
	}

}
