import java.io.File;
import java.io.IOException;

import weka.core.*;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class Credit {
	static Instances data = null;
	static Instances datanew = null;

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		DataSource source = null;
		
		try{
		source = new DataSource("/Users/katerina/Desktop/MATLAB/Lista8_213049/213049L2_2.arff");
		data = source.getDataSet();
		zad1();
		zad2();
		zad3();
		}
		catch(Exception e){
			 System.out.println("Error: " + e.getMessage());
			e.printStackTrace();
		}
		 // setting class attribute if the data format does not provide this information
		 // For example, the XRFF format saves the class attribute information as well
	    if (data.classIndex() == -1)
		   data.setClassIndex(data.numAttributes() - 1);
	
	    }
	
	public static void zad1(){
		int numInstances = data.numInstances(); 
		for(int i = numInstances-1;i>=0;--i){
			Instance currInstance = data.instance(i);
			Attribute statPoz = data.attribute("status-pozyczki");
			Attribute kwotaKredytu = data.attribute("kwoty-kredytu");
			if (currInstance.stringValue(statPoz).equals("odmowa")||currInstance.value(kwotaKredytu)>900){
				data.delete(i);
			}
		}
		System.out.println(data);
	}
	public static void 	zad2() throws Exception{
		try{
		String[] options = new String[2];
        options[0] = "-R";                                    
        options[1] = "1";                                     
        Remove remove = new Remove();                         
        remove.setOptions(options);                          
        remove.setInputFormat(data);                          // inform filter about dataset **AFTER** setting options
        data = Filter.useFilter(data, remove);   // apply filter
	     System.out.println(datanew);
	     System.out.println(data);
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	public static void zad3() throws IOException{
		try{
		ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("/Users/katerina/Desktop/MATLAB/Lista8_213049/213049L3_2.arff"));
        saver.writeBatch();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
}
