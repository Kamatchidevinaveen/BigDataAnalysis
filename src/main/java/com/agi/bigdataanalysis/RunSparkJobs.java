package com.agi.bigdataanalysis;

import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RunSparkJobs {

	static final Logger logger = LoggerFactory.getLogger("agiLogger");

	public static void mains(String[] args) {
		if (args.length == 0) {
			logger.error("Invalid number of parameters !!");
			System.exit(0);
		}

		SparkSession spark = SparkSession.builder().appName(args[0]).getOrCreate();

		if (args[0].equals("TextSimilarity")) {
			if (args.length < 4 || args.length > 5) {
				logger.error("TextSimilarity job requires 4 arguments\n1. Query\n2. Hadoop folder path\n3. "
						+ "Threshold\n4. Run on partial sets <<optional>>");
				System.exit(0);
			}

			Boolean runOnPartialSets = false;
			if (args.length == 5) {
				runOnPartialSets = Boolean.parseBoolean(args[4]);
			}

			TextSimilarity textSimilarity = new TextSimilarity();
			textSimilarity.textSimilarityJob(spark, args[1], args[2], Double.valueOf(args[3]), runOnPartialSets);
		} else if (args[0].equals("GetTFIDF")) {
			if (args.length != 2) {
				logger.error("TF-IDF job requires 1 argument ie : hadoop folder path");
				System.exit(0);
			}
			GenerateTFIDF tfIdf = new GenerateTFIDF();
			tfIdf.tfidf(spark, args[1]);
		} else if (args[0].equals("TFIDFSimilarity")) {
			if (args.length < 3 || args.length > 4) {
				logger.error("TFIDFSimilarity requires 3 arguments\n1. Query\n2. Hadoop folder path\n3. Run on partial sets <<optional>>");
				System.exit(0);
			}
			DocumentSimilarity docSimilarity = new DocumentSimilarity();
			docSimilarity.getSampleDocSimilarity(spark.sparkContext(), args[1], args[2], Boolean.parseBoolean(args[3]));
		}
		else {
			logger.error("Invalid job name !!");
			System.exit(0);
		}
	}

}
