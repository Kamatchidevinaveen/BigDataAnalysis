package com.agi.bigdata.utils;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Functions implements Serializable {

	private static final long serialVersionUID = 9017828966857081370L;
	private static final Logger logger = LoggerFactory.getLogger("agiLogger");

	public Function<String, List<String>> getFileFunction() {
		return new Function<String, List<String>>() {

			private static final long serialVersionUID = -3733044437125897100L;

			@Override
			public List<String> call(String s) {
				return Arrays.asList(s.split("\\s+"));
			}
		};
	}

	public Function<MatrixEntry, String> getMatrixFunction(
			List<String> fileNameList) {
		return new Function<MatrixEntry, String>() {

			private static final long serialVersionUID = -7435346472945942111L;
			final Long queryIndex = new Long(fileNameList.size() - 1);

			@Override
			public String call(MatrixEntry e) {
				if (queryIndex.equals(new Long(e.j()))) {
					logger.info(fileNameList.get((int) e.i()) + "\t"
							+ fileNameList.get((int) e.j()) + "\t" + e.value());
					return String.format("%s,%s,%s", fileNameList.get((int) e.i()),
							fileNameList.get((int) e.j()), e.value());
				} else {
					return "";
				}
			}
		};
	}
}
