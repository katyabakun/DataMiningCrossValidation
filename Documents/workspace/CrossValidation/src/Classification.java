import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Matrix;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class Classification {
     static String filePath = "/Users/katerina/Desktop/MATLAB/Lista9_213049/213049_4_1.arff";
     static Instances data = null;
     
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		loadData(filePath);
		data.setClassIndex( 6 );
	
		int testCount = 2;
		int foldCount = 6;
	    // classifiers
	    
		//Zero-R
		//System.out.println("ZeroR: \n");
		double [] zeroR=cross_valid(new ZeroR(),data,foldCount,testCount);
		
		//NaiveBayes
		System.out.println("NaiveBayes: \n");
		double[] naiveB=cross_valid(new NaiveBayes(),data,foldCount,testCount);
		
		//JRip
		System.out.println("JRip: \n");
		double[] jrip=cross_valid(new JRip(),data,foldCount,testCount);
		
		//J48
        //System.out.println("J48: \n");
		double[] j48=cross_valid(new J48(),data,foldCount,testCount);
		
		//SMO
        //System.out.println("SMO: \n");
		//double[] smo=cross_valid(new SMO(),data,foldCount,testCount);
		
		//MultilayerPerceptron
       // System.out.println("MultilayerPerceptron: \n");
		//double[] multPerc=cross_valid(new MultilayerPerceptron(),data,foldCount,testCount);
		
		System.out.format("Porownanie ze wzgledu na gmean i AUC dla NaiveBayes i JRip dla parametrow foldCount i testCount\n");
		System.out.format("Gmean: %.3f oraz Auc dla NaiveBayes %.3f\n",naiveB[0],naiveB[1]);
		System.out.format("Gmean: %.3f oraz Auc dla JRip %.3f\n",jrip[0],jrip[1]);
	}
	public static void loadData(String filePath){
        DataSource source = null;
		
		try{
		source = new DataSource(filePath);
		data = source.getDataSet();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	public static double[] cross_valid(Classifier classifier,Instances test_data,int foldCount,int testCount) throws Exception{
		//ConfusionMatrix m = new ConfusionMatrix();
		double gmean = 0;
		double auc=0; 
		double [][]Matrix= new double[2][2];
		for(int i=0;i<testCount;++i){
			int seed = i+1;
		Random random = new Random(seed);
		Instances randData=new Instances(test_data);
		randData.randomize(random);
		 Evaluation eval = new Evaluation(randData);
		for (int n = 0; n < foldCount; n++) {
			   Instances train = randData.trainCV(foldCount, n);
			   Instances test = randData.testCV(foldCount, n);
			 
			   // further processing, classification, etc.
			   Classifier classifierCopy = classifier;
		        classifierCopy.buildClassifier(train);
		        
		        eval.evaluateModel(classifierCopy, test);
			 }

		double [][]confMatrix= new double[2][2];
		confMatrix = eval.confusionMatrix();
		Matrix = add(Matrix,confMatrix);
		
		System.out.format("Accuracy: %.3f\n", (eval.numTruePositives(0)+eval.numTrueNegatives(0))/(eval.numFalseNegatives(0)+eval.numFalsePositives(0)+eval.numTrueNegatives(0)+eval.numTruePositives(0)));
		System.out.format("TPrate: %.3f\n", eval.weightedTruePositiveRate());
		System.out.format("TNrate: %.3f\n", eval.weightedTrueNegativeRate());
		System.out.format("AUC: %.3f\n", (1+eval.weightedTruePositiveRate() - eval.weightedTrueNegativeRate())/2);
		System.out.format("GMean: %.3f\n", Math.sqrt(eval.weightedTruePositiveRate() * eval.weightedTrueNegativeRate()));
		gmean=Math.sqrt(eval.weightedTruePositiveRate() * eval.weightedTrueNegativeRate());
		auc=(1+eval.weightedTruePositiveRate() - eval.weightedTrueNegativeRate())/2;

		}
		
		
		for (String[] s : getStrings(Matrix)){
	        System.out.println(Arrays.toString(s));
	    }
		double[]array = new double[2];
		array[0]=gmean;
		array[1]=auc;
		return array;
	}
	public static double[][] add(double[][]tab1,double[][]tab2){
		double[][]tab=new double[tab1[0].length][tab1.length];
		for(int i =0;i<tab[0].length;++i){
			for(int j=0;j<tab.length;++j){
				tab[i][j]=tab1[i][j]+tab2[i][j];
			}
		}
		return tab;		
	}
	public static double[][]average(double[][]tab,int count){
		
		for(int i =0;i<tab[0].length;++i){
			for(int j=0;j<tab.length;++j){
				tab[i][j]/=count;
			}
		}
		return tab;
	}
	public static String[][] getStrings(double [][]tab){

		String[][] output = new String[tab.length][];
	    int i = 0;
	    for (double[] d : tab){
	        output[i++] = Arrays.toString(d).replace("[", "").replace("]", "").split(",");
	    }
	    return output;
	}
	public static void saveas_arff() throws IOException{
		try{
		ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("/Users/katerina/Desktop/MATLAB/Lista9_213049/213049L4_2.arff"));
        saver.writeBatch();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}

}
