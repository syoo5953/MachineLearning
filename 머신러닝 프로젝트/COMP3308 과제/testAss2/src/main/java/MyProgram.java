// author: Internet's own boy

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.TreeMap;


public class MyProgram 
{

	public static void main(String[] args) 
	{
		BufferedReader trainingSetReader = null;
		BufferedReader testSetReader = null;
		String trainingSetPath = args[0];
		String testSetPath = args[1];
		String classifier = args[2];
		ArrayList<Instance> trainingSet = new ArrayList<>();
		ArrayList<Double[]> testSet = new ArrayList<>();
		
		try 
		{
			trainingSetReader = new BufferedReader(new FileReader(trainingSetPath));
			String line;
			String[] tokenizedLine;
			Double[] attributeValues;
			
			String instanceClass;
			
			/* read in training data set */
			while((line = trainingSetReader.readLine()) != null) 
			{
				if(line.trim().length() > 0) 
				{
					tokenizedLine = line.split(",");
					instanceClass = tokenizedLine[tokenizedLine.length - 1];
					attributeValues = new Double[tokenizedLine.length - 1];
					for(int i = 0; i < tokenizedLine.length - 1; i++) 
					{
						attributeValues[i] = Double.parseDouble(tokenizedLine[i]);
					}
					trainingSet.add(new Instance(attributeValues, instanceClass));
				}
			}

			testSetReader = new BufferedReader(new FileReader(testSetPath));
			
			/* read in test data set */
			while((line = testSetReader.readLine()) != null) 
			{
				if(line.trim().length() > 0) 
				{
					tokenizedLine = line.split(",");
					attributeValues = new Double[tokenizedLine.length];
					for(int i = 0; i < tokenizedLine.length; i++) 
					{
						attributeValues[i] = Double.parseDouble(tokenizedLine[i]);
					}
					testSet.add(attributeValues);
				}
			}
			
			/* run the classification algorithm given as an argument */
			if(classifier.equals("NB")) 
			{
				NaiveBayes nb = new NaiveBayes();
				nb.train(trainingSet);
				for(Double[] testInstance : testSet) 
				{
					System.out.println(nb.classify(testInstance));
				}
			}
			else
			{
				int k = Character.getNumericValue(classifier.charAt(0));
				KNN knn = new KNN(trainingSet, k);
				for(Double[] testInstance : testSet) 
				{
					System.out.println(knn.classify(testInstance));
				}
			}
			
		} 
		catch (FileNotFoundException e) 
		{
			e.printStackTrace();
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		} 
		finally {
			if(trainingSetReader != null) 
			{
				try 
				{
					trainingSetReader.close();
				}
				catch (IOException e) 
				{
					e.printStackTrace();
				}
			}
			if(testSetReader != null) 
			{
				try 
				{
					testSetReader.close();
				} catch (IOException e) 
				{
					e.printStackTrace();
				}
			}
		}
		
		/* if a fourth argument is passed in (filename), run 10-fold Stratified 
		 * Cross Validation
		 */
		if(args.length > 3) 
		{
			CrossValidation cv = new CrossValidation();
			cv.stratification(args[3]);
			cv.validation();
		}
	}//end of main method

	public static class CrossValidation
	{

		private final int S_FOLDS = 10;

		// change for different K in MyProgram.KNN
		private final int K = 5;
		private ArrayList<ArrayList<Instance>> folds;
		private String filename;

		public CrossValidation()
		{
			folds = new ArrayList<>();
			for(int i = 0; i < S_FOLDS; i++)
			{
				folds.add(new ArrayList<Instance>());
			}
		}

		/**
		 * Stratifies the data into S folds
		 * @param filename the file containing the data
		 */
		public void stratification(String filename)
		{
			BufferedReader reader = null;
			BufferedWriter writer = null;
			HashMap<String, ArrayList<Instance>> dataSet = new HashMap<>();
			String[] tokenizedString = filename.split("\\.");
			String outFilename = tokenizedString[0] + "-folds." + tokenizedString[1];
			this.filename = outFilename;

			try
			{
				reader = new BufferedReader(new FileReader(filename));
				String line;
				String[] tokenizedLine;
				Double[] attributeValues;

				String instanceClass;

				/* read in the data set */
				while((line = reader.readLine()) != null)
				{
					if(line.trim().length() > 0)
					{
						tokenizedLine = line.split(",");
						instanceClass = tokenizedLine[tokenizedLine.length - 1];
						attributeValues = new Double[tokenizedLine.length - 1];
						for(int i = 0; i < tokenizedLine.length - 1; i++)
						{
							attributeValues[i] = Double.parseDouble(tokenizedLine[i]);
						}
						if(!dataSet.containsKey(instanceClass))
						{
							dataSet.put(instanceClass, new ArrayList<Instance>());
						}
						dataSet.get(instanceClass).add(new Instance(attributeValues, instanceClass));
					}
				}

				/* create the S folds */
				for(String key : dataSet.keySet())
				{
					// for each class
					/* shuffle the instances */
					Collections.shuffle(dataSet.get(key));
					int numClassInstances = dataSet.get(key).size();

					 // for each fold
					for(int i = 0; i < folds.size(); i++)
					{
						// for each instance of that class
						for(int j = (numClassInstances/S_FOLDS) - 1; j >= 0; j--)
						{
							folds.get(i).add(dataSet.get(key).get(j));
							dataSet.get(key).remove(j);
						}
					}
				}
				/* add remaining instances evenly to folds */
				for(String key : dataSet.keySet())
				{
					// for each class
					for(int i = 0; i < folds.size(); i++)
					{ // for each fold
						if(dataSet.get(key).size() <= 0)
						{
							break;
						}
						folds.get(i).add(dataSet.get(key).get(0));
						dataSet.get(key).remove(0);
					}
				}

				writer = new BufferedWriter(new FileWriter(outFilename));

				/* write the S folds to file */
				for(int i = 0; i < folds.size(); i++)
				{
					writer.write("fold" + (i + 1));
					for(int j = 0; j < folds.get(i).size(); j++)
					{
						writer.write("\n" + folds.get(i).get(j).getAttributesAsString());
					}
					if(i != folds.size() - 1)
					{
						writer.write("\n\n");
					}
				}
			}
			catch (FileNotFoundException e)
			{
				e.printStackTrace();
			}
			catch (IOException e)
			{
				e.printStackTrace();
			}
			finally
			{
				if(reader != null)
				{
					try
					{
						reader.close();
					}
					catch (IOException e)
					{
						e.printStackTrace();
					}
				}
				if(writer != null)
				{
					try
					{
						writer.close();
					}
					catch (IOException e)
					{
						e.printStackTrace();
					}
				}
			}
		}

		/**
		 * Runs Cross-Validation on the S folds
		 */
		public void validation()
		{
			double nbAccuracy = 0.0;
			double knnAccuracy = 0.0;

			System.out.println("\nRunning Cross-Validation on file: " + filename);

			/* Run 10 fold cross validation for Naive Bayes and MyProgram.KNN */
			for(int i = 0; i < S_FOLDS; i++)
			{
				ArrayList<Instance> trainingSet = new ArrayList<>();
				for(int j = 0; j < folds.size(); j++)
				{
					if(j != i)
					 {
						trainingSet.addAll(folds.get(j));
					}
				}
				NaiveBayes nb = new NaiveBayes();
				nb.train(trainingSet);
				KNN knn = new KNN(trainingSet, K);
				double nbRunAccuracy = 0.0;
				double knnRunAccuracy = 0.0;

				// for each instance in the test fold
				for(int k = 0; k < folds.get(i).size(); k++)
				{
					String result = nb.classify(folds.get(i).get(k).getAttributes());
					String actualClass = folds.get(i).get(k).getInstanceClass();
					if(result.equals(actualClass))
					{
						nbRunAccuracy += 1.0;
					}
					result = knn.classify(folds.get(i).get(k).getAttributes());
					if(result.equals(actualClass))
					{
						knnRunAccuracy += 1.0;
					}
				}
				nbRunAccuracy /= folds.get(i).size();
				knnRunAccuracy /= folds.get(i).size();
				nbAccuracy += nbRunAccuracy;
				knnAccuracy += knnRunAccuracy;
				System.out.println("Run " + (i + 1) + " leaving fold " + (i + 1) +" out - " + "Accuracy of Naive Bayes: " + nbRunAccuracy + "%");
				System.out.println("Run " + (i + 1) + " leaving fold " + (i + 1) +" out - " + "Accuracy of " + K + "NN: " + knnRunAccuracy + "%");
			}
			nbAccuracy /= S_FOLDS;
			knnAccuracy /= S_FOLDS;
			System.out.println("Accuracy of Naive Bayes: " + nbAccuracy + "%");
			System.out.println("Accuracy of " + K + "NN: " + knnAccuracy + "%");
		}

	}//end of class

	public static class NaiveBayes
	{

		private HashMap<String, Double[]> attributeMean;
		private HashMap<String, Double[]> attributeStdDev;
		private HashMap<String, Double> classPriorProb;
		private HashMap<String, Integer> numClasses;

		/**
		 * Constructs the Naive Bayes classifier
		 */
		public NaiveBayes()
		{
			attributeMean = new HashMap<>();
			attributeStdDev = new HashMap<>();
			numClasses = new HashMap<>();
			classPriorProb = new HashMap<>();
		}

		/**
		 * Normal Probability Density Function
		 * @param x the value to calculate for
		 * @param u mean
		 * @param o standard deviation
		 * @return returns the Normal Probability Density Function for x
		 */
		public double pdf(double x, double u, double o)
		{
			double fraction = (1.0)/(o * Math.sqrt(2*Math.PI));
			double exponent = -((Math.pow((x - u), 2)/(2*Math.pow(o, 2))));
			double e = Math.pow(Math.E, exponent);
			double pdf = fraction * e;
			return pdf;
		}

		/**
		 * Method to train the classifier
		 * @param trainingSet set of arrays(instances) the classifier trains on
		 */
		public void train(ArrayList<Instance> trainingSet)
		{

			/* calculate mean for each attribute */
			for(Instance instance : trainingSet)
			{
				// for each instance of the trainingSet

				String instanceClass = instance.getInstanceClass();

				/* If there is no attribute mean for a class in the map, initialise it with
				 * the values array of that instance and then skip to the next instance.
				 * While traversing the instances keep count of the number of each class.
				 */
				if(!attributeMean.containsKey(instanceClass))
				{
					attributeMean.put(instanceClass, instance.getAttributes());
					numClasses.put(instanceClass, 1);
					continue;
				}

				int update = numClasses.get(instanceClass) + 1;
				numClasses.put(instanceClass, update);

				for(int i = 0; i < instance.getNumAttributes(); i++)
				{
					attributeMean.get(instanceClass)[i] += instance.getAttributeValue(i);
				}
			}

			// for each class
			for(String key : attributeMean.keySet())
			{
				 // for each attribute
				for(int i = 0; i < attributeMean.get(key).length; i++)
				{
					attributeMean.get(key)[i] /= numClasses.get(key);
				}
			}

			/* calculate standard deviation for each attribute */
			for(Instance instance : trainingSet)
			{
				 // for each instance of the trainingSet
				String instanceClass = instance.getInstanceClass();

				/* If there is no attribute standard deviation for a class in the map,
				 * initialise it with that instances squared difference from the mean
				 * for each attribute and then skip to the next instance.
				 */
				if(!attributeStdDev.containsKey(instanceClass))
				{
					Double[] squaredDifferences = new Double[instance.getNumAttributes()];
					for(int i = 0; i < squaredDifferences.length; i++)
					{
						double x = instance.getAttributeValue(i);
						double mean = attributeMean.get(instanceClass)[i];
						squaredDifferences[i] = Math.pow((x - mean), 2);
					}
					attributeStdDev.put(instanceClass, squaredDifferences);
					continue;
				}

				// for each attribute
				for(int i = 0; i < instance.getNumAttributes(); i++)
				{
					double x = instance.getAttributeValue(i);
					double mean = attributeMean.get(instanceClass)[i];
					attributeStdDev.get(instanceClass)[i] += Math.pow((x - mean), 2);
				}
			}

			// for each class
			for(String key : attributeStdDev.keySet())
			{
				// for each attribute
				for(int i = 0; i < attributeStdDev.get(key).length; i++)
				{
					attributeStdDev.get(key)[i] = Math.sqrt(attributeStdDev.get(key)[i]*(1.0/((numClasses.get(key) - 1))));
				}
			}

			/* calculate prior probability for each class */
			int totalInstances = trainingSet.size();

			// for each class
			for(String key : numClasses.keySet())
			{
				double priorProb = (double) (numClasses.get(key))/totalInstances;
				classPriorProb.put(key, priorProb);
			}
		}

		/**
		 * Classifies the given instance using Bayes Theorem: P(H|E) = P(E|H)P(H)
		 * @param testInstance an array containing the attribute values
		 * @return returns the classification for the given instance
		 */
		public String classify(Double[] testInstance)
		{
			HashMap<String, Double> probabilities = new HashMap<>();
			String classification = "";
			Double highestProb = -Double.MAX_VALUE;

			/* calculate the probability for each class */
			for(String key : attributeMean.keySet())
			{
				// for each class
				Double probability = classPriorProb.get(key);

				// for each attribute
				for(int i = 0; i < testInstance.length; i++)
				{
					double x = testInstance[i];
					double u = attributeMean.get(key)[i];
					double o = attributeStdDev.get(key)[i];
					double pdf = pdf(x, u, o);
					probability *= pdf;
				}
				probabilities.put(key, probability);
			}

			/* check which class is most likely */
			for(String key : probabilities.keySet())
			{
			// for each class
				if(probabilities.get(key) > highestProb) {
					highestProb = probabilities.get(key);
					classification = key;
				}
			}
			return classification;
		}

	}//end of class

	public static class KNN
	{

		private ArrayList<Instance> trainingSet;
		private final int K;

		/**
		 * Constructs the kNN classifier
		 * @param trainingSet set of arrays(instances) the classifier trains on
		 * @param k number of neighbours
		 */
		public KNN(ArrayList<Instance> trainingSet, int k)
		{
			this.trainingSet = trainingSet;
			this.K = k;
		}

		/**
		 * Euclidean distance measure
		 * @param a vector of attribute values
		 * @param b vector of attribute values
		 * @return returns the Eucledian distance between vector a and vector b
		 */
		private double eucledianDistance(Double[] a, Double[] b)
		{
			double distance = 0.0;
			for(int i = 0; i < a.length; i++) {
				distance += Math.pow((a[i] - b[i]), 2);
			}
			return Math.sqrt(distance);
		}

		/**
		 * Classifies the given instance using MyProgram.KNN classifier
		 * @param testInstance an array containing the attribute values
		 * @return returns the classification for the given instance
		 */
		public String classify(Double[] testInstance)
		{
			TreeMap<Double, String> eucledianDistances= new TreeMap<>();
			HashMap<String, Integer> classCount = new HashMap<>();
			String classification = "";

			/* calculate the testInstances distance from each value of the trainingSet */
			for(Instance instance : trainingSet)
			{
				String instanceClass = instance.getInstanceClass();
				double distance = eucledianDistance(instance.getAttributes(), testInstance);
				eucledianDistances.put(distance, instanceClass);
			}

			/* get the K closest instances */
			int i = 1;
			for(Double distance : eucledianDistances.keySet())
			{
				if(i > K) {
					break;
				}
				String distanceClass = eucledianDistances.get(distance);
				if(!classCount.containsKey(distanceClass))
				{
					classCount.put(distanceClass, 1);
					i++;
					continue;
				}
				classCount.put(distanceClass, (classCount.get(distanceClass) + 1));
				i++;
			}


			/* work out the majority class - this is the classification */
			int trackMajority = 0;

			// for each class
			for(String instanceClass : classCount.keySet())
			{
				if(classCount.get(instanceClass) == trackMajority)
				{
					classification = "yes";
				}
				else if(classCount.get(instanceClass) > trackMajority)
				{
					classification = instanceClass;
					trackMajority = classCount.get(instanceClass);
				}
			}
			return classification;
		}

	}//end of class

	/**
	 * Class to store a single instance of a data set
	 */
	public static class Instance
	{

		private final Double[] attributeValues;
		private final String instanceClass;

		public Instance(Double[] attributeValues, String instanceClass)
		{
			this.attributeValues = attributeValues;
			this.instanceClass = instanceClass;
		}

		public Double[] getAttributes()
		{
			return attributeValues.clone();
		}

		public double getAttributeValue(int index)
		{
			return attributeValues[index];
		}

		public int getNumAttributes()
		{
			return attributeValues.length;
		}

		public String getInstanceClass()
		{
			return instanceClass;
		}

		public String getAttributesAsString()
		{
			String line = "";
			for(int i = 0; i < attributeValues.length; i++)
			{
				line += attributeValues[i];
				line += ",";
			}
			line += instanceClass;
			return line;
		}

	}//end of class
}//end of class