package com.agi.bigdata.utils;

import java.io.ByteArrayInputStream;
import java.io.InputStream;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.input.PortableDataStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.agi.docparser.base.AbstractStreamFactory;
import com.agi.docparser.base.FileStreamParser;
import com.agi.docparser.base.StreamFactoryProducer;
import com.agi.docparser.utils.CommonFileExtensions;

public class FileParseUtils {
	private static final Logger logger = LoggerFactory.getLogger("agiLogger");

	/**
	 * Parses the file and returns text content of the file.
	 * If the file does not belong to any of the extensions mentioned in
	 * {@link com.agi.docparser.utils.CommonFileExtensions CommonFileExtensions}
	 * class
	 * then empty string is returned
	 * 
	 * @param stream
	 * @param filePath
	 * @return text content of the file
	 */
	public String getFileContentAsString(PortableDataStream stream, String filePath) {
		byte[] fileArrayStream = stream.toArray();
		InputStream is = new ByteArrayInputStream(fileArrayStream);
		AbstractStreamFactory aof = StreamFactoryProducer.getOfficeFactory(filePath);

		if (aof == null) {
			logger.warn("File parser does not support " + filePath + " file");
			return "";
		}

		FileStreamParser docFileParser = aof.getFileParser();
		String fileContent = "";
		try {
			fileContent = docFileParser.getFileContentAsStream(is);
		} catch (Exception ex) {
			logger.info("There was some exception while parsing the documents", ex);
		}

		return fileContent;
	}

	/**
	 * Parses the file and returns text content of the file.
	 * If the file does not belong to any of the extensions mentioned in
	 * {@link com.agi.docparser.utils.CommonFileExtensions CommonFileExtensions}
	 * class
	 * then empty string is returned
	 * 
	 * @param filePath
	 * @return text content of the file
	 */
	public String getFileContentAsString(Path filePath) {
		String fileContent = "";

		String filePathString = filePath.toUri().toString();
		if (filePathString.endsWith(CommonFileExtensions.TXT.toString()) || filePathString.endsWith(CommonFileExtensions.PDF.toString())
				|| filePathString.endsWith(CommonFileExtensions.PPT.toString()) || filePathString.endsWith(CommonFileExtensions.PPTX.toString())
				|| filePathString.endsWith(CommonFileExtensions.XLS.toString()) || filePathString.endsWith(CommonFileExtensions.XLSX.toString())
				|| filePathString.endsWith(CommonFileExtensions.DOC.toString()) || filePathString.endsWith(CommonFileExtensions.DOCX.toString())) {

			AbstractStreamFactory aof = StreamFactoryProducer.getOfficeFactory(filePath.toString());
			if (aof == null) {
				logger.warn("File parser does not support " + filePathString + " file");
				return "";
			}

			FileStreamParser docFileParser = aof.getFileParser();
			try {
				Configuration config = new Configuration();
				FileSystem fs = FileSystem.get(config);
				fileContent = docFileParser.getFileContentAsStream(fs.open(filePath).getWrappedStream());
			} catch (Exception ex) {
				logger.info("There was some exception while parsing the documents", ex);
			}
		}
		return fileContent;
	}
}
