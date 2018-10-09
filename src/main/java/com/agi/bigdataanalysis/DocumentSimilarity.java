package com.agi.bigdataanalysis;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.spark.SparkContext;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.api.java.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.agi.bigdata.utils.FileParseUtils;
import com.agi.bigdata.utils.Functions;
import com.agi.bigdata.utils.GetJobConfig;
import com.agi.bigdata.utils.ReportGenerator;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DocumentSimilarity implements Serializable {

	private static final long serialVersionUID = 3853461736344510595L;
	private static final Logger logger = LoggerFactory.getLogger("agiLogger");

	public void getSampleDocSimilarity(SparkContext sparkContext, String query, String path, boolean runOnPartialSets) {
		int numberOfPartialSets = GetJobConfig.getTFIDFPartialSetValue();
		if (runOnPartialSets) {
			logger.info("Running on partial sets enabled, Partial set length is " + numberOfPartialSets);
		}
		JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sparkContext);

		List<String> files = new ArrayList<String>();
		List<String> fileNames = new ArrayList<String>();
		List<Path> filePaths = new ArrayList<Path>();

		Configuration config = new Configuration();
		FileParseUtils fileParseUtils = new FileParseUtils();

		try {
			FileSystem fs = FileSystem.get(config);
			Path folderPath = new Path(path);

			if (!fs.exists(folderPath) || !fs.isDirectory(folderPath)) {
				logger.info("Not a valid path");
				return;
			}
			RemoteIterator<LocatedFileStatus> iterator = fs.listFiles(folderPath, true);
			while (iterator.hasNext()) {
				filePaths.add(iterator.next().getPath());
			}
		} catch (IOException ioEx) {
			logger.error("There was some exception while reading file contents ", ioEx);
		}

		int counter = 0;
		List<List<String>> finalOutput = new ArrayList<List<String>>();
		for (Path filePath : filePaths) {
			String textContent = fileParseUtils.getFileContentAsString(filePath);
			if (textContent.isEmpty()) {
				continue;
			}
			files.add(textContent);
			fileNames.add(filePath.toUri().toString());

			logger.info(counter++ + "\t" + filePath);
			if (runOnPartialSets) {
				if (counter % numberOfPartialSets == 0) {
					files.add(query);
					fileNames.add("User-Query");
					finalOutput.addAll(performaPartialDocumentSimilarity(jsc, files, fileNames));
					fileNames = new ArrayList<String>();
					files = new ArrayList<String>();
				}
			}
		}

		files.add(query);
		fileNames.add("User-Query");
		// Processing the remaining files
		if (runOnPartialSets) {
			if (filePaths.size() % numberOfPartialSets != 0) {
				finalOutput.addAll(performaPartialDocumentSimilarity(jsc, files, fileNames));
			}
		} else {
			finalOutput = performaPartialDocumentSimilarity(jsc, files, fileNames);
		}

		ReportGenerator reportGenerator = new ReportGenerator();
		try {
			reportGenerator.generateTFIDFTextSimilarityReport(finalOutput, query, path);
		} catch (IOException ioEx) {
			logger.error("There was some exception while generating report", ioEx);
		}
	}

	public static RowMatrix transposeRM(JavaSparkContext jsc, RowMatrix mat) {
		List<Vector> newList = new ArrayList<Vector>();
		List<Vector> vs = mat.rows().toJavaRDD().collect();
		double[][] tmp = new double[(int) mat.numCols()][(int) mat.numRows()];

		for (int i = 0; i < vs.size(); i++) {
			double[] rr = vs.get(i).toArray();
			for (int j = 0; j < mat.numCols(); j++) {
				tmp[j][i] = rr[j];
			}
		}

		for (int i = 0; i < mat.numCols(); i++) {
			newList.add(Vectors.dense(tmp[i]));
		}

		JavaRDD<Vector> rows2 = jsc.parallelize(newList);
		RowMatrix newmat = new RowMatrix(rows2.rdd());
		return (newmat);
	}

	private List<List<String>> performaPartialDocumentSimilarity(JavaSparkContext jsc, List<String> files, List<String> fileNames) {
		Functions fc = new Functions();
		JavaRDD<List<String>> fileCounts = jsc.parallelize(files).map(fc.getFileFunction()).filter((list) -> {
			return list.size() != 0;
		});

		HashingTF hashingTF = new HashingTF();
		JavaRDD<Vector> tf = hashingTF.transform(fileCounts);
		tf.cache();

		IDFModel idfModel = new IDF().fit(tf);
		JavaRDD<Vector> idf = idfModel.transform(tf);
		RowMatrix rowMatrix = new RowMatrix(JavaRDD.toRDD(idf));
		CoordinateMatrix coordinateMatrix = transposeRM(jsc, rowMatrix).columnSimilarities();

		JavaRDD<MatrixEntry> entries = coordinateMatrix.entries().toJavaRDD();
		JavaRDD<String> output = entries.map(fc.getMatrixFunction(fileNames));

		return output.map(new Function<String, List<String>>() {
			private static final long serialVersionUID = 5289892532649425137L;

			@Override
			public List<String> call(String matchedResult) throws Exception {
				if (matchedResult.isEmpty()) {
					return null;
				}
				return Arrays.asList(matchedResult.split(" "));
			}
		}).filter(new Function<List<String>, Boolean>() {
			private static final long serialVersionUID = -1581789161177898926L;

			@Override
			public Boolean call(List<String> list) throws Exception {
				if (list == null || list.size() == 0) {
					return false;
				} else {
					return true;
				}
			}
		}).collect();
	}
}
