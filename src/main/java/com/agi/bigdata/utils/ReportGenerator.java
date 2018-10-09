package com.agi.bigdata.utils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Result;
import javax.xml.transform.Source;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.sql.Row;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import com.agi.bigdataanalysis.GenerateTFIDF;

public class ReportGenerator {

	private static final Logger logger = LoggerFactory.getLogger("agiLogger");
	private Configuration conf = new Configuration();

	private static final String pattern = "[^" + "\u0009\r\n" + "\u0020-\uD7FF" + "\uE000-\uFFFD" + "\ud800\udc00-\udbff\udfff" + "]";
	private static Pattern invalidUnicode = Pattern.compile(pattern);

	public void generateXMLReportOfTFIDF(List<Row> featuresRow, List<Row> tfIdfResult, String[] vocabularyWords, String[] customStopWords,
			ArrayList<String> fileList, String totalProcessingTime) {
		DocumentBuilderFactory docFactory = DocumentBuilderFactory.newInstance();
		DocumentBuilder docBuilder = null;
		try {
			docBuilder = docFactory.newDocumentBuilder();
		} catch (ParserConfigurationException ex) {
			logger.error("There was some exception while creating xml document", ex);
			return;
		}

		// Root elements
		Document doc = docBuilder.newDocument();
		Element rootElement = doc.createElement("tf-idf-results");
		doc.appendChild(rootElement);

		Element documents = doc.createElement("documents");

		List<PopularWords> popularWords = new ArrayList<ReportGenerator.PopularWords>();

		for (int i = 0; i < featuresRow.size(); i++) {
			Element document = doc.createElement("document");

			Row row1 = featuresRow.get(i);
			SparseVector sv1 = (SparseVector) row1.get(row1.size() - 1);

			Row row2 = tfIdfResult.get(i);
			SparseVector sv2 = (SparseVector) row2.get(row2.size() - 1);

			int[] originalIndices = sv1.indices();
			for (int index : originalIndices) {
				Matcher matcher = invalidUnicode.matcher(vocabularyWords[index]);
				if (matcher.find()) {
					continue;
				}
				Element row = doc.createElement("row");
				row.setAttribute("word", vocabularyWords[index]);
				row.setAttribute("indexNumber", String.valueOf(index));
				row.setAttribute("tf", String.valueOf(sv2.apply(index)));
				row.setAttribute("tfidf", String.valueOf(sv1.apply(index)));
				document.appendChild(row);
				// Adding popular words to list
				popularWords.add(new PopularWords(vocabularyWords[index], sv1.apply(index)));
			}
			documents.appendChild(document);
		}

		Element summary = doc.createElement("summary");

		Element processedDocuments = doc.createElement("processedDocuments");
		processedDocuments.setTextContent(String.valueOf(fileList.size()));

		Element listOfDocuments = doc.createElement("listOfDocuments");
		listOfDocuments.setTextContent(fileList.toString());

		Element processingTimeInSeconds = doc.createElement("processingTimeInSeconds");
		processingTimeInSeconds.setTextContent(totalProcessingTime);

		HashMap<String, Integer> extensionCount = new HashMap<String, Integer>();
		for (String filePath : fileList) {
			String ext = filePath.split("\\.")[1];
			if (!extensionCount.containsKey(ext)) {
				extensionCount.put(ext, 1);
			} else {
				extensionCount.replace(ext, extensionCount.get(ext) + 1);
			}
		}

		Element documentStatistics = doc.createElement("documentStatistics");
		Set<String> keys = extensionCount.keySet();

		for (String key : keys) {
			Element documentType = doc.createElement("documentType");
			documentType.setAttribute("ext", key);
			documentType.setAttribute("numberOfDocumentProcessed", String.valueOf(extensionCount.get(key)));
			documentStatistics.appendChild(documentType);
		}

		summary.appendChild(processedDocuments);
		summary.appendChild(listOfDocuments);
		summary.appendChild(documentStatistics);
		summary.appendChild(processingTimeInSeconds);

		Collections.sort(popularWords);
		Element popularWordsTag = doc.createElement("popularWords");

		for (int i = 0; i < popularWords.size(); i++) {
			Element popularWordTag = doc.createElement("popularWord");
			popularWordTag.setAttribute("word", popularWords.get(i).getWord());
			popularWordTag.setAttribute("tfidf", String.valueOf(popularWords.get(i).getTfidf()));
			popularWordsTag.appendChild(popularWordTag);
		}

		summary.appendChild(popularWordsTag);
		rootElement.appendChild(documents);
		rootElement.appendChild(summary);

		Source xmlSource = new DOMSource(doc);
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
		Result outputTarget = new StreamResult(outputStream);

		try {
			TransformerFactory transformerFactory = TransformerFactory.newInstance();
			Transformer transformer = transformerFactory.newTransformer();
			transformer.setOutputProperty(OutputKeys.INDENT, "yes");
			transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
			transformer.transform(xmlSource, outputTarget);

			Path reportFilePath = new Path(GetJobConfig.getTFIDFXMLReportOutputPath());
			FileSystem fs = FileSystem.get(URI.create(GetJobConfig.getTFIDFXMLReportOutputPath()), conf);

			if (fs.exists(reportFilePath)) {
				fs.delete(reportFilePath, false);
			}
			OutputStream out = fs.create(reportFilePath);
			ByteArrayInputStream inStream = new ByteArrayInputStream(outputStream.toByteArray());
			org.apache.hadoop.io.IOUtils.copyBytes(inStream, out, 4096, true);
		} catch (TransformerException | IOException ex) {
			logger.error("Exception while writing Document stream to xml", ex);
		}
	}

	public void generateHTMLFromXML() {
		GenerateTFIDF.class.getClassLoader();
		InputStream is = GenerateTFIDF.class.getClassLoader().getResourceAsStream("HtmlTemplateReport.html");

		if (is == null) {
			System.out.println("input stream is null");
		}

		String htmlString = "";
		try {
			htmlString = IOUtils.toString(is);
		} catch (IOException ioEx) {
			System.out.println(ioEx);
		}

		StringBuilder body = new StringBuilder();
		String tableString = "<table border=\"1\" bordercolor=\"#000000\">" + "<tr><th>IndexNumber</th><th>Word</th><th>TF</th><th>TF-IDF</th></tr>";
		StringBuilder tableBody = new StringBuilder();
		DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
		FileSystem fs = null;

		try {
			DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
			Document doc;

			Path xmlReportFilePath = new Path(GetJobConfig.getTFIDFXMLReportOutputPath());
			fs = FileSystem.get(URI.create(GetJobConfig.getTFIDFXMLReportOutputPath()), conf);

			if (!fs.exists(xmlReportFilePath)) {
				logger.error("Text-Similarity-Report.xml does not exist in HDFS");
				return;
			}

			doc = dBuilder.parse(fs.open(xmlReportFilePath));
			doc.getDocumentElement().normalize();

			Element rootElement = (Element) doc.getElementsByTagName("tf-idf-results").item(0);

			Node documents = rootElement.getElementsByTagName("documents").item(0);
			Element docElements = (Element) documents;

			NodeList documentList = docElements.getElementsByTagName("document");

			for (int i = 0; i < documentList.getLength(); i++) {
				tableBody.append(tableString);
				Node documentNode = documentList.item(i);
				if (documentNode.getNodeType() == Node.ELEMENT_NODE) {
					Element documentElement = (Element) documentNode;
					NodeList rowList = documentElement.getElementsByTagName("row");
					for (int j = 0; j < rowList.getLength(); j++) {
						Node rowNode = rowList.item(j);
						if (rowNode.getNodeType() == Node.ELEMENT_NODE) {
							Element rowElement = (Element) rowNode;
							tableBody.append("<tr>");
							tableBody.append("<td>" + rowElement.getAttribute("indexNumber") + "</td>");
							tableBody.append("<td>" + rowElement.getAttribute("word") + "</td>");
							tableBody.append("<td>" + rowElement.getAttribute("tf") + "</td>");
							tableBody.append("<td>" + rowElement.getAttribute("tfidf") + "</td>");
							tableBody.append("</tr>");
						}
					}
				}
				tableBody.append("</table><hr />");
			}

			body.append(tableBody.toString());

			if (!htmlString.isEmpty()) {
				htmlString = htmlString.replace("$body", body.toString());
			}

			Element summary = (Element) rootElement.getElementsByTagName("summary").item(0);

			StringBuilder summaryBody = new StringBuilder();

			Element listOfDocuments = (Element) summary.getElementsByTagName("listOfDocuments").item(0);
			String list = listOfDocuments.getTextContent().replace("[", "").replace("]", "");
			summaryBody.append("<h4>List of all the document processed</h4>");
			summaryBody.append("<ul>");
			for (String filePath : list.split("\\,")) {
				summaryBody.append("<li>" + filePath.substring(filePath.lastIndexOf("/") + 1) + "</li>");
			}
			summaryBody.append("</ul>");

			Element processedDocuments = (Element) summary.getElementsByTagName("processedDocuments").item(0);
			summaryBody.append("<h4>Total number of document processed : " + processedDocuments.getTextContent() + "</h4>");

			Element processingTimeInSeconds = (Element) summary.getElementsByTagName("processingTimeInSeconds").item(0);
			summaryBody.append("<h4>Total processing time : " + processingTimeInSeconds.getTextContent() + "</h4>");

			Element documentStatistics = (Element) summary.getElementsByTagName("documentStatistics").item(0);

			NodeList documentTypes = documentStatistics.getChildNodes();
			StringBuilder tfidfSummary = new StringBuilder();

			for (int k = 0; k < documentTypes.getLength(); k++) {
				Node documentType = documentTypes.item(k);
				if (documentType.getNodeType() == Node.ELEMENT_NODE) {
					Element elementType = (Element) documentType;
					tfidfSummary.append("<tr>");
					tfidfSummary.append("<td>" + elementType.getAttribute("ext") + "</td>");
					tfidfSummary.append("<td>" + elementType.getAttribute("numberOfDocumentProcessed") + "</td>");
					tfidfSummary.append("</tr>");
				}
			}

			StringBuilder popularWords = new StringBuilder();
			Element popularWordsTag = (Element) summary.getElementsByTagName("popularWords").item(0);

			NodeList popularWordsNodes = popularWordsTag.getChildNodes();
			int wordCount = 0;
			for (int l = 0; l < popularWordsNodes.getLength(); l++) {
				Node wordNode = popularWordsNodes.item(l);
				if (wordNode.getNodeType() == Node.ELEMENT_NODE) {
					if (wordCount++ >= 5) {
						break;
					}
					Element wordElement = (Element) wordNode;
					popularWords.append("<tr>");
					popularWords.append("<td>" + wordElement.getAttribute("word") + "</td>");
					popularWords.append("<td>" + wordElement.getAttribute("tfidf") + "</td>");
					popularWords.append("</tr>");
				}
			}

			if (!htmlString.isEmpty()) {
				htmlString = htmlString.replace("$grossSummary", summaryBody.toString());
				htmlString = htmlString.replace("$summaryBody", tfidfSummary.toString());
				htmlString = htmlString.replace("$popularWordsBody", popularWords.toString());
			}

			InputStream htmlFileInputStream = new ByteArrayInputStream(htmlString.getBytes(StandardCharsets.UTF_8));
			Path htmlReportFilePath = new Path(GetJobConfig.getTFIDFHTMLReportOutputPath());

			if (fs.exists(htmlReportFilePath)) {
				fs.delete(htmlReportFilePath, false);
			}
			OutputStream out = fs.create(htmlReportFilePath);
			org.apache.hadoop.io.IOUtils.copyBytes(htmlFileInputStream, out, 4096, true);
		} catch (ParserConfigurationException | IOException | SAXException ex) {
			logger.error("Error while reading xml properties file ", ex);
		}
	}

	public void generateTextSimilarityReport(String query, String hadoopPath, double threshold, Map<String, Double> resultMap) throws IOException {
		InputStream is = GenerateTFIDF.class.getClassLoader().getResourceAsStream("TextSimilarityReport.html");

		if (is == null) {
			logger.error("input stream is null !! Could not locate TextSimilarityReport.html template");
		}

		String htmlString = "";
		try {
			htmlString = IOUtils.toString(is);
		} catch (IOException ioEx) {
			System.out.println(ioEx);
		}

		StringBuilder tableBody = new StringBuilder();

		Map<String, Double> sortedMap = resultMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
				.collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (oldValue, newValue) -> oldValue, LinkedHashMap::new));

		Set<String> files = sortedMap.keySet();
		for (String file : files) {
			tableBody.append("<tr><td>");
			tableBody.append(file);
			tableBody.append("</td></tr>");
			// appending the percentage here

			System.out.println(file + "\t" + resultMap.get(file));
			/*
			 * tableBody.append(resultMap.get(file) * 100);
			 * tableBody.append("</td><td>"); tableBody.append(1 -
			 * resultMap.get(file)); tableBody.append("</td></tr>");
			 */
		}

		htmlString = htmlString.replace("$query", query);
		htmlString = htmlString.replace("$hadoopPath", hadoopPath);
		htmlString = htmlString.replace("$threshold", String.valueOf(threshold));
		htmlString = htmlString.replace("$resultTable", tableBody.toString());

		InputStream htmlFileInputStream = new ByteArrayInputStream(htmlString.getBytes(StandardCharsets.UTF_8));
		Path reportFilePath = new Path(GetJobConfig.getTextSimilarityReportOutputPath());
		FileSystem fs = FileSystem.get(URI.create(GetJobConfig.getTextSimilarityReportOutputPath()), conf);

		if (fs.exists(reportFilePath)) {
			fs.delete(reportFilePath, false);
		}
		OutputStream out = fs.create(reportFilePath);
		org.apache.hadoop.io.IOUtils.copyBytes(htmlFileInputStream, out, 4096, true);
		logger.info("TextSimilarity report generated successfully");
	}

	class PopularWords implements Comparable<PopularWords> {
		private String word;
		private Double tfidf;

		public PopularWords(String word, Double tfidf) {
			this.tfidf = tfidf;
			this.word = word;
		}

		public Double getTfidf() {
			return tfidf;
		}

		public void setTfidf(Double tfidf) {
			this.tfidf = tfidf;
		}

		/*
		 * Here we are sorting in descending order
		 * (non-Javadoc)
		 * @see java.lang.Comparable#compareTo(java.lang.Object)
		 */
		@Override
		public int compareTo(PopularWords popularWords) {
			return popularWords.tfidf.compareTo(this.tfidf);
		}

		public String getWord() {
			return word;
		}

		public void setWord(String word) {
			this.word = word;
		}
	}

	public void generateTFIDFTextSimilarityReport(List<List<String>> output, String query, String path) throws IOException {
		List<TFIDFSimilarity> similarityList = new ArrayList<ReportGenerator.TFIDFSimilarity>();
		StringBuilder reportString = new StringBuilder();

		output.forEach((item) -> {
			if (item.size() == 0) {
				return;
			}
			similarityList.add(new TFIDFSimilarity(item.get(0).split("\\,")[0], new Double(item.get(0).split("\\,")[2])));
		});
		Collections.sort(similarityList);
		similarityList.forEach((matchedResult) -> {
			reportString.append("<tr><td>" + matchedResult.getPath() + "</td></tr>");
		});
		InputStream is = GenerateTFIDF.class.getClassLoader().getResourceAsStream("TFIDFSimilarity.html");

		if (is == null) {
			logger.error("Input stream is null !! Could not locate TFIDFSimilarity.html template");
		}

		String htmlString = "";
		try {
			htmlString = IOUtils.toString(is);
		} catch (IOException ioEx) {
			System.out.println(ioEx);
		}

		htmlString = htmlString.replace("$query", query);
		htmlString = htmlString.replace("$hadoopPath", path);
		htmlString = htmlString.replace("$resultTable", reportString.toString());

		InputStream htmlFileInputStream = new ByteArrayInputStream(htmlString.getBytes(StandardCharsets.UTF_8));
		Path reportFilePath = new Path(GetJobConfig.getTFIDFSimilarityOutputPath());
		FileSystem fs = FileSystem.get(URI.create(GetJobConfig.getTFIDFSimilarityOutputPath()), conf);

		if (fs.exists(reportFilePath)) {
			fs.delete(reportFilePath, false);
		}
		OutputStream out = fs.create(reportFilePath);
		org.apache.hadoop.io.IOUtils.copyBytes(htmlFileInputStream, out, 4096, true);
		logger.info("TFIDF TextSimilarity report generated successfully");
	}

	class TFIDFSimilarity implements Comparable<TFIDFSimilarity> {

		public TFIDFSimilarity(String path, Double similarityValue) {
			this.path = path;
			this.similarityValue = similarityValue;
		}

		public String getPath() {
			return path;
		}

		public void setPath(String path) {
			this.path = path;
		}

		public Double getSimilarityValue() {
			return similarityValue;
		}

		public void setSimilarityValue(Double similarityValue) {
			this.similarityValue = similarityValue;
		}

		private String path;
		private Double similarityValue;

		@Override
		public int compareTo(TFIDFSimilarity tfidfSimilarity) {
			return tfidfSimilarity.similarityValue.compareTo(this.similarityValue);
		}

	}
}
