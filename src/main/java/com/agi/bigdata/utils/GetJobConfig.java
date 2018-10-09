package com.agi.bigdata.utils;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class GetJobConfig {

	private static final Log logger = LogFactory.getLog(GetJobConfig.class);

	public static int getNGramValue() {
		InputStream is = null;
		try {
			Properties prop = new Properties();
			is = GetJobConfig.class.getClassLoader().getResourceAsStream("job-config.properties");
			prop.load(is);
			return Integer.parseInt(prop.getProperty("text-similarity.ngram.value"));
		} catch (IOException ioe) {
			logger.error("There was some exception while reading properties file");
			// In case if there is any exception pass the default value
			return 3;
		} finally {
			try {
				if (is != null) {
					is.close();
				}
			} catch (IOException ioe) {
				logger.error("There was some exception while closing the file stream");
			}
		}
	}

	public static int getNumHashTables() {
		InputStream is = null;
		try {
			Properties prop = new Properties();
			is = GetJobConfig.class.getClassLoader().getResourceAsStream("job-config.properties");
			prop.load(is);
			return Integer.parseInt(prop.getProperty("text-similarity.hash-tables.value"));
		} catch (IOException ioe) {
			logger.error("There was some exception while reading properties file");
			// In case if there is any exception pass the default value
			return 10;
		} finally {
			try {
				if (is != null) {
					is.close();
				}
			} catch (IOException ioe) {
				logger.error("There was some exception while closing the file stream");
			}
		}
	}

	public static String getTextSimilarityReportOutputPath() {
		InputStream is = null;
		try {
			Properties prop = new Properties();
			is = GetJobConfig.class.getClassLoader().getResourceAsStream("job-config.properties");
			prop.load(is);
			return prop.getProperty("tex-similarity.output.filepath");
		} catch (IOException ioe) {
			logger.error("There was some exception while reading properties file");
			return "/user";
		} finally {
			try {
				if (is != null) {
					is.close();
				}
			} catch (IOException ioe) {
				logger.error("There was some exception while closing the file stream");
			}
		}
	}

	public static String getTFIDFXMLReportOutputPath() {
		InputStream is = null;
		try {
			Properties prop = new Properties();
			is = GetJobConfig.class.getClassLoader().getResourceAsStream("job-config.properties");
			prop.load(is);
			return prop.getProperty("tf-idf.ouput.xml.filepath");
		} catch (IOException ioe) {
			logger.error("There was some exception while reading properties file");
			return "/user";
		} finally {
			try {
				if (is != null) {
					is.close();
				}
			} catch (IOException ioe) {
				logger.error("There was some exception while closing the file stream");
			}
		}
	}

	public static String getTFIDFHTMLReportOutputPath() {
		InputStream is = null;
		try {
			Properties prop = new Properties();
			is = GetJobConfig.class.getClassLoader().getResourceAsStream("job-config.properties");
			prop.load(is);
			return prop.getProperty("tf-idf.ouput.html.filepath");
		} catch (IOException ioe) {
			logger.error("There was some exception while reading properties file");
			return "/user";
		} finally {
			try {
				if (is != null) {
					is.close();
				}
			} catch (IOException ioe) {
				logger.error("There was some exception while closing the file stream");
			}
		}
	}

	public static String getTFIDFSimilarityOutputPath() {
		InputStream is = null;
		try {
			Properties prop = new Properties();
			is = GetJobConfig.class.getClassLoader().getResourceAsStream("job-config.properties");
			prop.load(is);
			return prop.getProperty("tf-idf.similarity.ouput.html.filepath");
		} catch (IOException ioe) {
			logger.error("There was some exception while reading properties file");
			return "/user";
		} finally {
			try {
				if (is != null) {
					is.close();
				}
			} catch (IOException ioe) {
				logger.error("There was some exception while closing the file stream");
			}
		}
	}
	
	public static int getTextSimilarityPartialSetValue() {
		InputStream is = null;
		try {
			Properties prop = new Properties();
			is = GetJobConfig.class.getClassLoader().getResourceAsStream("job-config.properties");
			prop.load(is);
			return Integer.parseInt(prop.getProperty("text-similarity.partial-set"));
		} catch (IOException ioe) {
			logger.error("There was some exception while reading properties file");
			// In case if there is any exception pass the default value
			return 10;
		} finally {
			try {
				if (is != null) {
					is.close();
				}
			} catch (IOException ioe) {
				logger.error("There was some exception while closing the file stream");
			}
		}
	}
	
	public static int getTFIDFPartialSetValue() {
		InputStream is = null;
		try {
			Properties prop = new Properties();
			is = GetJobConfig.class.getClassLoader().getResourceAsStream("job-config.properties");
			prop.load(is);
			return Integer.parseInt(prop.getProperty("tfidf-similarity.partial-set"));
		} catch (IOException ioe) {
			logger.error("There was some exception while reading properties file");
			// In case if there is any exception pass the default value
			return 10;
		} finally {
			try {
				if (is != null) {
					is.close();
				}
			} catch (IOException ioe) {
				logger.error("There was some exception while closing the file stream");
			}
		}
	}

}
