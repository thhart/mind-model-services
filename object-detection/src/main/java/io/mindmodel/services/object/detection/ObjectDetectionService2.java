package io.mindmodel.services.object.detection;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import io.mindmodel.services.common.GraphRunner;
import io.mindmodel.services.common.GraphRunnerMemory;
import io.mindmodel.services.common.ProtoBufGraphDefinition;
import io.mindmodel.services.common.attic.GraphicsUtils;
import io.mindmodel.services.object.detection.domain.ObjectDetection;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.types.UInt8;

import org.springframework.core.io.DefaultResourceLoader;

/**
 * @author Christian Tzolov
 */
public class ObjectDetectionService2 implements AutoCloseable {

	public static List<String> FETCH_NAMES = Arrays.asList(
			ObjectDetectionOutputConverter.DETECTION_SCORES, ObjectDetectionOutputConverter.DETECTION_CLASSES,
			ObjectDetectionOutputConverter.DETECTION_BOXES, ObjectDetectionOutputConverter.NUM_DETECTIONS);

	public static List<String> FETCH_NAMES_WITH_MASKS = Arrays.asList(
			ObjectDetectionOutputConverter.DETECTION_SCORES, ObjectDetectionOutputConverter.DETECTION_CLASSES,
			ObjectDetectionOutputConverter.DETECTION_BOXES, ObjectDetectionOutputConverter.DETECTION_MASKS,
			ObjectDetectionOutputConverter.NUM_DETECTIONS);

	private final GraphRunner imageNormalization;
	private final GraphRunner objectDetection;
	private final ObjectDetectionOutputConverter outputConverter;


	public ObjectDetectionService2(String modelUri, ObjectDetectionOutputConverter outputConverter) {

		this.imageNormalization = new GraphRunner("raw_image", "normalized_image")
				.withGraphDefinition(tf -> {
					Placeholder<String> rawImage = tf.withName("raw_image").placeholder(String.class);
					Operand<UInt8> decodedImage = tf.dtypes.cast(
							tf.image.decodeJpeg(rawImage, DecodeJpeg.channels(3L)), UInt8.class);
					// Expand dimensions since the model expects images to have shape: [1, H, W, 3]
					tf.withName("normalized_image").expandDims(decodedImage, tf.constant(0));
				});

		this.objectDetection = new GraphRunner(Arrays.asList("image_tensor"), FETCH_NAMES)
				.withGraphDefinition(new ProtoBufGraphDefinition(
						new DefaultResourceLoader().getResource(modelUri), true));

		this.outputConverter = outputConverter;
	}

	public List<ObjectDetection> detect(byte[] image) {
		try (Tensor inputTensor = Tensor.create(image);
			 GraphRunnerMemory memorize = new GraphRunnerMemory()) {

			List<List<ObjectDetection>> out = this.imageNormalization.andThen(memorize)
					.andThen(this.objectDetection).andThen(memorize)
					.andThen(outputConverter)
					.apply(Collections.singletonMap("raw_image", inputTensor));

			return out.get(0);

		}
	}

	@Override
	public void close() {
		this.imageNormalization.close();
		this.objectDetection.close();
		//this.outputConverter.close();
	}

	public static void main(String[] args) throws IOException {
		String modelUri = "http://dl.bintray.com/big-data/generic/ssdlite_mobilenet_v2_coco_2018_05_09_frozen_inference_graph.pb";
		String labelUri = "http://dl.bintray.com/big-data/generic/mscoco_label_map.pbtxt";

		ObjectDetectionOutputConverter outputAdapter = new ObjectDetectionOutputConverter(
				new DefaultResourceLoader().getResource(labelUri), 0.4f, FETCH_NAMES);

		//byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/object-detection.jpg");
		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/wild-animals-15.jpg");

		try (ObjectDetectionService2 objectDetectionService2 =
					 new ObjectDetectionService2(modelUri, outputAdapter)) {

			List<ObjectDetection> boza = objectDetectionService2.detect(inputImage);

			System.out.println(boza);
		}
	}
}
