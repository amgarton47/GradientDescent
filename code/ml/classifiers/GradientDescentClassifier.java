package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * @author dkauchak, added on to by Aidan Garton
 *
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;

	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;

	protected int regularizationType, lossFunction;
	protected double lambda, eta;

	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight

	protected int iterations = 10;

	public GradientDescentClassifier() {
		regularizationType = NO_REGULARIZATION;
		lossFunction = EXPONENTIAL_LOSS;
		lambda = 0.01;
		eta = 0.01;
	}

	/**
	 * Set the surrogate loss function to use (i.e., objective function)
	 * 
	 * @param lossFunction
	 */
	public void setLoss(int lossFunction) {
		this.lossFunction = lossFunction;
	}

	/**
	 * Set the type of regularization to use during training (none, L1, or L2)
	 * 
	 * @param regularizationType
	 */
	public void setRegularization(int regularizationType) {
		this.regularizationType = regularizationType;
	}

	/**
	 * Set the regularization weight
	 * 
	 * @param lambda
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Set the learning rate
	 * 
	 * @param eta
	 */
	public void setEta(double eta) {
		this.eta = eta;
	}

	/**
	 * Get a weight vector over the set of features with each weight set to 0
	 * 
	 * @param features the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features) {
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		for (Integer f : features) {
			temp.put(f, 0.0);
		}

		return temp;
	}

	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features) {
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	/**
	 * Calculates the regulrization magnitude based on the specided regularization
	 * type
	 * 
	 * @param w the weight value being modified
	 * @return the regularization magnitude constant
	 */
	private double reg(double w) {
		switch (regularizationType) {
		case NO_REGULARIZATION:
			return 0;
		case L1_REGULARIZATION:
			return w <= 0 ? -1 : 1;
		case L2_REGULARIZATION:
			return w;
		default:
			return 0;
		}
	}

	/**
	 * Calculates the incremental loss for a given training example based on the
	 * specified surrogate loss function
	 * 
	 * @param e     the example to calculate loss for
	 * @param label the label of that example
	 * @return the loss for e
	 */
	private double loss(Example e, double label) {
		switch (lossFunction) {
		case EXPONENTIAL_LOSS:
			return Math.exp(-1 * label * getDistanceFromHyperplane(e, weights, b));
		case HINGE_LOSS:
			return getDistanceFromHyperplane(e, weights, b) * label < 1 ? 1 : 0;
		default:
			return Math.exp(-1 * label * getDistanceFromHyperplane(e, weights, b));
		}
	}

	/**
	 * trains the data set using a gradient descent approach by updating each weight
	 * for each example a certain number of iterations. The surrogate loss function
	 * can be altered as well as the regularization method.
	 */
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());

		ArrayList<Example> training = (ArrayList<Example>) data.getData().clone();

		for (int it = 0; it < iterations; it++) {
			Collections.shuffle(training);
			double totalLoss = 0;

			for (Example e : training) {
				double label = e.getLabel();
				String str = "";

				// update the weights
				// for( Integer featureIndex: weights.keySet() ){
				for (Integer featureIndex : e.getFeatureSet()) {
					double oldWeight = weights.get(featureIndex);
					double featureValue = e.getFeature(featureIndex);

					totalLoss += label * featureValue * loss(e, label);

					double newWeight = oldWeight
							+ eta * (label * featureValue * loss(e, label) - lambda * reg(oldWeight));

					weights.put(featureIndex, newWeight);
					str += newWeight + ", ";
				}

				// update b
				b += eta * (label * loss(e, label) - lambda * reg(b));

				// prints out weights after each update
//				System.out.println("it " + it + ": weights: " + str + " " + b);
			}
			// prints total loss per iteration
			System.out.println("(" + it + "," + totalLoss + ")");
		}

		// prints out the final weights after training is done
		for (Integer featureIndex : weights.keySet()) {
			System.out.println(weights.get(featureIndex));
		}
	}

	@Override
	public double classify(Example example) {
		return getPrediction(example);
	}

	@Override
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}

	/**
	 * Get the prediction from the current set of weights on this example
	 * 
	 * @param e the example to predict
	 * @return
	 */
	protected double getPrediction(Example e) {
		return getPrediction(e, weights, b);
	}

	/**
	 * Get the prediction from the on this example from using weights w and inputB
	 * 
	 * @param e      example to predict
	 * @param w      the set of weights to use
	 * @param inputB the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = getDistanceFromHyperplane(e, w, inputB);

		if (sum > 0) {
			return 1.0;
		} else if (sum < 0) {
			return -1.0;
		} else {
			return 0;
		}
	}

	protected static double getDistanceFromHyperplane(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = inputB;

		// for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for (Integer featureIndex : e.getFeatureSet()) {
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}

		return sum;
	}

	public String toString() {
		StringBuffer buffer = new StringBuffer();

		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);

		for (Integer index : temp) {
			buffer.append(index + ":" + weights.get(index) + " ");
		}

		return buffer.substring(0, buffer.length() - 1);
	}

	public static void main(String[] args) {
		DataSet data = new DataSet("/Users/aidangarton/Desktop/Java/assign4-starter/data/titanic-train.csv",
				DataSet.CSVFILE);
		DataSet data1 = new DataSet("code/ml/data/data.csv", DataSet.CSVFILE);

		double lamb = 0.01, et = 0.07;
		int numTrain = 10;
		for (int i = 0; i < 10; i++) {
			GradientDescentClassifier gdc = new GradientDescentClassifier();

			gdc.setEta(lamb);
			gdc.setLambda(et);
//			gdc.setLoss(GradientDescentClassifier.HINGE_LOSS);
			gdc.setRegularization(L2_REGULARIZATION);
			gdc.train(data);

			double c = 0, t = 0;
			for (Example e : data.getData()) {
				t++;
				if (e.getLabel() == gdc.classify(e)) {
					c++;
				}
			}
			System.out.println("(" + numTrain + "," + c / t + ")");

			numTrain += 5;
		}

	}
}
