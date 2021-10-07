import java.io.*;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.TreeMap;

public class MyClassifier {

    private static final int MAX_NUMBEROFFOLDS = 10;

    public static void main(String[] args) {

        if (getData(args)[0].matches("stratify")) { // CMArguments: stratify 5NN pima.csv/pima-CFS.csv

            final ArrayList<ArrayList<HashMap<String, Double[]>>> cross = new ArrayList<>();
            for (int i = 0; i < MAX_NUMBEROFFOLDS; i++) {
                cross.add(new ArrayList<>());

            }
            int number = 1;
            String filename = getData(args)[2];
            BufferedReader reader;
            BufferedWriter writer;
            final HashMap<String, ArrayList<HashMap<String, Double[]>>> tmpMap = new HashMap<>();
            final String[] tokenizedString = filename.split("\\.");
            final String outFilename = getData(tokenizedString)[0] + "-folds." +
                    getData(tokenizedString)[1];

            try {
                reader = new BufferedReader(new FileReader(filename));
                String line;
                String[] data;
                Double[] values;

                String YesOrNo;

                while ((line = reader.readLine()) != null) {
                    data = line.trim().split(",");

                    YesOrNo = getData(data)[getData(data).length - 1]; // yes, no

                    values = new Double[getData(data).length - 1];


                    for (int i1 = 0; i1 < getData(data).length - 1; i1++) {
                        values[i1] = Double.parseDouble(getData(data)[i1]);
                    }
                    if (!tmpMap.containsKey(YesOrNo)) {
                        tmpMap.put(YesOrNo, new ArrayList<>());
                    }

                    HashMap<String, Double[]> map = new HashMap<>();
                    map.put(YesOrNo, values);
                    tmpMap.get(YesOrNo).add(map);
                }
                tmpMap.keySet().forEach(key -> {
                    Collections.shuffle(tmpMap.get(key));
                    final int eachClassSize = tmpMap.get(key).size();
                    cross.forEach(fold -> {
                        for (int j1 = (eachClassSize / MAX_NUMBEROFFOLDS) - 1; j1 >= 0; j1--) {
                            fold.add(tmpMap.get(key).get(j1));
                            tmpMap.get(key).remove(j1);
                        }
                    });
                });

                tmpMap.keySet().forEach(key -> {
                    for (ArrayList<HashMap<String, Double[]>> fold1 : cross) {
                        if (tmpMap.get(key).size() <= 0) {
                            break;
                        }
                        fold1.add(tmpMap.get(key).get(0));


                        tmpMap.get(key).remove(0);
                    }
                });


                writer = new BufferedWriter(new FileWriter(outFilename));

                for (int i1 = 0; i1 < cross.size(); i1++) {
                    writer.write("fold" + (i1 + 1));
                    for (int j1 = 0; j1 < cross.get(i1).size(); j1++) {
                        HashMap<String, Double[]> node = cross.get(i1).get(j1);
                        final StringBuilder line1 = new StringBuilder();

                        for(String str : node.keySet())
                        {
                            for (double attributeValue : node.get(str)) {
                                line1.append(attributeValue);
                                line1.append(",");
                            }
                            line1.append(str);
                            writer.write("\n" + line1.toString());
                        }

                    }
                    if (i1 != cross.size() - 1) {
                        writer.write("\n\n");
                    }
                }


                reader.close();
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (getData(args)[1].substring(1).matches("NN")) {
                number = Character.getNumericValue(getData(args)[1].charAt(0));
            }

            double NB_Result = 0.0;
            double KNN_Result = 0.0;
            int r = 0;
            while (r < MAX_NUMBEROFFOLDS) {

                final ArrayList<HashMap<String, Double[]>> tempTraining = new ArrayList<>();

                int p = 0;
                while(p < cross.size())
                {
                    if (p != r) {
                        tempTraining.addAll(cross.get(p));
                    }
                    p++;
                }

                final NaiveBayes nb = new NaiveBayes();
                nb.classify(tempTraining);

                kNearestNeighbours knn;

                double NB_acc = 0.0;
                double KNN_acc = 0.0;

                int l = 0;
                while (l < cross.get(r).size()) {

                    for (String str : cross.get(r).get(l).keySet()) {
                        Double[] test = cross.get(r).get(l).get(str);
                        String result = probNB(nb, test);
                        if (result.equals(str)) {
                            NB_acc += 1.0;
                        }
                        knn = new kNearestNeighbours();
                        result = knn.classify(tempTraining, number, cross.get(r).get(l).get(str));
                        if (result.equals(str)) {
                            KNN_acc += 1.0;
                        }
                        l++;
                    }
                }
                NB_acc /= cross.get(r).size();
                KNN_acc /= cross.get(r).size();
                NB_Result += NB_acc;
                KNN_Result += KNN_acc;

                System.out.println("Fold " + (r + 1) + " accuracy for " + (number*100.00) + "NN: " +
                        KNN_acc + "%");

                System.out.println("Fold " + (r + 1) + " accuracy for Naive Bayes: " + (NB_acc*100.00) +
                        "%");

                r++;
            }
            NB_Result /= MAX_NUMBEROFFOLDS;
            KNN_Result /= MAX_NUMBEROFFOLDS;

            System.out.println(number + "NN Accuracy: " + (KNN_Result*100.00) +
                    "%");

            System.out.println(
                    "NB Accuracy: " + (NB_Result*100) + "%");

        } else {
            processClassifying(getData(args));
        }
    }

    private static String probNB(NaiveBayes nb, Double[] instance) {
        final HashMap<String, Double> probs = new HashMap<>();
        String data_cl = "";
        Double max = -Double.MAX_VALUE;

        nb.mean.keySet().forEach(key -> {
            Double prob = nb.priorProb.get(key);

            for (int i1 = 0; i1 < instance.length; i1++) {
                Double tempProb = prob;
                final double x = instance[i1];
                final double y = nb.mean.get(key)[i1];
                final double z = nb.stdDev.get(key)[i1];


                final double valueFrac = (1.0) / (z * Math.sqrt(2 * Math.PI));
                final double exp = -((Math.pow((x - y), 2) / (2 * Math.pow(z, 2))));
                final double mapthPow = Math.pow(Math.E, exp);
                final double result = valueFrac * mapthPow;
                tempProb *= result;
                prob = tempProb;
            }
            probs.put(key, prob);
        });

        for (String key : probs.keySet()) {
            if (probs.get(key) > max) {
                max = probs.get(key);
                data_cl = key;
            }
        }
        return data_cl;
    }

    private static void processClassifying(String[] args) {
        ArrayList<HashMap<String, Double[]>> training = new ArrayList<>();
        BufferedReader readTrain;
        BufferedReader readTest;

        try {
            readTrain = new BufferedReader(new FileReader(getData(args)[0]));
            String line;
            String[] data;
            Double[] attribute;

            String instClass;

            while ((line = readTrain.readLine()) != null) {
                data = line.trim().split(",");
                instClass = getData(data)[getAnInt(getData(data))];
                attribute = new Double[getData(data).length - 1];
                int i = 0;
                while(i < getData(data).length - 1) {
                    attribute[i] = Double.parseDouble(getData(data)[i]);
                    //System.out.println(attributeValues[i]);
                    i++;
                }
                HashMap<String, Double[]> map = new HashMap<>();
                map.put(instClass, attribute);
                training.add(map);

            }

            readTest = new BufferedReader(new FileReader(getData(args)[1]));
            BufferedReader readTest1 = new BufferedReader(new FileReader(getData(args)[1]));
            BufferedReader readTest2 = new BufferedReader(new FileReader(getData(args)[1]));
            int dataLength = getSize(readTest1) - 1;
            int valueLength = getSize1(readTest2);
            Double[][] newVal = new Double[dataLength][valueLength];
            int o = 0;

            while ((line = readTest.readLine()) != null) {
                data = line.trim().split(",");

                for (int i = 0; i < getData(data).length; i++) {
                    newVal[o][i] = Double.parseDouble(getData(data)[i]);
                }
                o++;
            }
            if ("NB".matches(getData(args)[2])) {
                final NaiveBayes nb = new NaiveBayes();
                nb.classify(training);
                for(int y = 0; y < dataLength; y++)
                {
                    System.out.println(probNB(nb, newVal[y]));
                }
            } else {
                final int k = Character.getNumericValue(getData(args)[2].charAt(0));
                kNearestNeighbours knn = new kNearestNeighbours();

                for(int y = 0; y < dataLength; y++)
                {
                    System.out.println(knn.classify(training, k, newVal[y]));
                }

            }

            readTrain.close();
            readTest.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static int getSize1(BufferedReader readTest2) throws IOException {
        int length = 1;
        String line;
        String[] data;

        while ((line = readTest2.readLine()) != null) {
            data = line.trim().split(",");
            length = data.length;
        }

        return length;
    }

    private static int getSize(BufferedReader readTest) throws IOException {
        int length = 1;

        while (readTest.readLine() != null) {
            length++;
        }

        return length;
    }

    private static String[] getData(String[] data) {
        return data;
    }

    private static int getAnInt(String[] data) {
        return getData(data).length - 1;
    }

    public static class NaiveBayes {

        private final HashMap<String, Double[]> mean;
        private final HashMap<String, Double> priorProb;
        private final HashMap<String, Double[]> stdDev;
        private final Integer[][] classNumber;


        public NaiveBayes() {
            mean = new HashMap<>();
            priorProb = new HashMap<>();
            stdDev = new HashMap<>();
            classNumber = new Integer[2][1];
        }

        public void classify(ArrayList<HashMap<String, Double[]>> trainingSet) {
            for (HashMap<String, Double[]> node : trainingSet) {

                for (String str : node.keySet()) {

                    if (!mean.containsKey(str)) {
                        mean.put(str, node.get(str));
                        if(str.toLowerCase().matches("yes")) {
                            classNumber[0][0] = 1;
                        } else if(str.toLowerCase().matches("no")) {
                            classNumber[1][0] = 1;
                        } else {
                            System.out.println("the class instance is not yes or no");
                            System.exit(0);
                        }
                        continue;
                    }
                    if(str.toLowerCase().matches("yes")) {
                        classNumber[0][0] += 1;
                    } else if(str.toLowerCase().matches("no")) {
                        classNumber[1][0] += 1;
                    }

                    for (int i = 0; i < node.get(str).length; i++) {
                        mean.get(str)[i] += node.get(str)[i];
                    }
                }
            }

            mean.keySet().forEach(key -> {

                for (int i = 0; i < mean.get(key).length; i++) {
                    if(key.toLowerCase().matches("yes"))
                    {
                        mean.get(key)[i] /= classNumber[0][0];
                    } else if (key.toLowerCase().matches("no"))
                    {
                        mean.get(key)[i] /= classNumber[1][0];
                    }
                }
            });

            for (HashMap<String, Double[]> node : trainingSet) {

                for (String str : node.keySet()) {

                    if (!stdDev.containsKey(str)) {
                        final Double[] diff = new Double[node.get(str).length];
                        for (int i = 0; i < diff.length; i++) {
                            diff[i] = Math
                                    .pow(((double) node.get(str)[i] - mean.get(str)[i]), 2);
                        }
                        stdDev.put(str, diff);
                        continue;
                    }

                    for (int i = 0; i < node.get(str).length; i++) {
                        final double x = node.get(str)[i];
                        final double mean = this.mean.get(str)[i];
                        stdDev.get(str)[i] += Math.pow((x - mean), 2);
                    }
                }
            }

            stdDev.keySet().forEach(key -> {
                for (int i = 0; i < stdDev.get(key).length; i++) {

                    if(key.toLowerCase().matches("yes"))
                    {
                        stdDev.get(key)[i] = Math
                                .sqrt(stdDev.get(key)[i] * (1.0 / ((classNumber[0][0] - 1))));
                    } else if (key.toLowerCase().matches("no"))
                    {
                        stdDev.get(key)[i] = Math
                                .sqrt(stdDev.get(key)[i] * (1.0 / ((classNumber[1][0] - 1))));
                    }
                }
            });
            final int total = trainingSet.size();

            String[] YN = {"yes", "no"};

            for(String key : YN) {
                if(key.toLowerCase().matches("yes"))
                {
                    this.priorProb.put(key, (double) (classNumber[0][0]) / total);
                } else if (key.toLowerCase().matches("no"))
                {
                    this.priorProb.put(key, (double) (classNumber[1][0]) / total);
                }
            }
        }
    }

    public static class kNearestNeighbours {

        public String classify(ArrayList<HashMap<String, Double[]>> setting, int K, Double[] testInstance) {
            final TreeMap<Double, String> distance = new TreeMap<>();
            final HashMap<String, Integer> numYesorNo = new HashMap<>();
            final Integer[][] numClass = new Integer[2][1];
            numClass[0][0] = null;
            numClass[1][0] = null;
            String sysOut = "";

            setting.forEach(node -> {
                for (String str : node.keySet()) {
                    double S = 0.0;
                    for (int i = 0; i < node.get(str).length; i++) {
                        S += ((node.get(str)[i] - testInstance[i]) * (node.get(str)[i] - testInstance[i]));
                    }
                    final double euclidDistance = Math.abs(Math.sqrt(S));
                    distance.put(euclidDistance, str);
                }
            });

            int i = 1;
            for (Double dist1 : distance.keySet()) {
                if (i > K) {
                    break;
                }
                if (!numYesorNo.containsKey(distance.get(dist1))) {

                    numYesorNo.put(distance.get(dist1), 1);
                    i++;
                    continue;
                }
                numYesorNo.put(distance.get(dist1), (numYesorNo.get(distance.get(dist1)) + 1));
                i++;
            }

            int keepTracking = 0;

            for (String YN : numYesorNo.keySet()) {
                if (numYesorNo.get(YN) == keepTracking) {
                    sysOut = "yes";


                } else if (numYesorNo.get(YN) > keepTracking) {
                    sysOut = YN;
                    keepTracking = numYesorNo.get(YN);
                }
            }
            return sysOut;
        }
    }
}

