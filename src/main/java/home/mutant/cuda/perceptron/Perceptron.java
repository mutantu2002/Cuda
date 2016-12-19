package home.mutant.cuda.perceptron;

import java.io.IOException;

import home.mutant.cuda.model.Kernel;
import home.mutant.cuda.model.MemoryFloat;
import home.mutant.cuda.model.Program;
import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageDouble;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;

public class Perceptron {
	float weights[];
	float deltas[];
	float images[];
	float outputs[];
	double maxAccuracy = 0;
	float learningRate = 10;
	
	int noThreads = 32*25;
	int noImages=60000;
	int imageSize;
	int trainLabel = 8;
	
	Program program;
	
	MemoryFloat memWeights ;
	MemoryFloat memWeightsDeltas;
	MemoryFloat memImages ;
	MemoryFloat memOutputs;
	Kernel deltasBatch;
	Kernel deltasOne;
	int offsetInputImages;
	
	public Perceptron(int size) throws IOException{
		imageSize = size;
		int inputsPerBatch = (int)Math.ceil(((double)noImages)/noThreads);
		weights = new float[size+1];
		deltas = new float[(size+1)*noThreads];
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImagesNormalized();
		images = new float[noImages*imageSize];
		for (int i=0;i<noImages;i++){
			System.arraycopy(MnistDatabase.trainImages.get(i).getDataFloat(), 0, images, i*imageSize, imageSize);
		}
		
		outputs = new float[noImages];
		for (int i=0;i<noImages;i++){
			outputs[i] = MnistDatabase.trainLabels.get(i)==trainLabel?1:0;
		}
		
		program = new Program("src/main/resources/Perceptron.ptx",null);	
		
		memWeights = new MemoryFloat(program);
		memWeights.add(weights);
		
		memWeightsDeltas = new MemoryFloat(program);
		memWeightsDeltas.add(deltas);
		
		memImages = new MemoryFloat(program);
		memImages.add(images);
		
		memOutputs = new MemoryFloat(program);
		memOutputs.add(outputs);
		
		deltasBatch = new Kernel(program, "deltasBatch");
		deltasBatch.addArgument(memImages);
		deltasBatch.addArgument(memOutputs);
		deltasBatch.addArgument(memWeights);
		deltasBatch.addArgument(memWeightsDeltas);
		deltasBatch.addArgument(inputsPerBatch);
		deltasBatch.addArgument(imageSize);
		
		deltasOne = new Kernel(program, "deltasOne");
		deltasOne.addArgument(memImages);
		deltasOne.addArgument(memOutputs);
		deltasOne.addArgument(memWeights);
		deltasOne.addArgument(memWeightsDeltas);
		deltasOne.addArgument(offsetInputImages);
		deltasOne.addArgument(imageSize);
		
	}
	
	/**
	 * modify the weights after calculating gradient for all entire training set
	 * @param noIterations
	 */
	public void runAllTrainingSet(int noIterations) {
		for(int it=0;it<noIterations;it++){
			deltasBatch.run(noThreads/32, 32);
			memWeightsDeltas.copyDtoH();
			modifyWeightsFromDelta();
			memWeights.copyHtoD();
			learningRate/=1.001;
			test();
			//System.out.println(learningRate);
		}
	}

	/**
	 * modify the weights after calculating gradient for a subset (batch) of the training set
	 * @param noIterations
	 */
	public void runBatch(int noIterations) {
		learningRate/=noThreads;
		int noStart=(int)Math.ceil((double)noImages/noThreads);
		for(int it=0;it<noIterations;it++){
			for(int batch=0;batch<noStart;batch++){
				deltasOne.setArgument(batch*noThreads, 4);
				deltasOne.run(noThreads/32, 32);
				memWeightsDeltas.copyDtoH();
				modifyWeightsFromDelta();
				memWeights.copyHtoD();
			}
			//learningRate/=1.006;
		}
	}
	
	public void modifyWeightsFromDelta(){
		for (int i = 0; i < weights.length; i++) {
			float delta=0;
			for(int noBatch=0;noBatch<noThreads;noBatch++){
				delta+=deltas[(imageSize+1)*noBatch+i];
			}
			weights[i]+=delta*learningRate;
		}
	}
	
	public Image getImage(){
		Image image = new ImageDouble(weights.length-1);
		double max = -1*Double.MAX_VALUE;
		double min = Double.MAX_VALUE;
		for (int i = 0; i < weights.length-1; i++) {
			if (weights[i]>max)max=weights[i];
			if (weights[i]<min)min=weights[i];
		}
		max=255/(max-min);
		for (int i = 0; i < weights.length-1; i++) {
			image.getDataDouble()[i]=(float) ((weights[i]-min)*max);
		}
		return image;
	}
	public void test(){
		int count=0;
		int total=0;
		for (int i=0;i<10000;i++){
			if(MnistDatabase.testLabels.get(i)==trainLabel){
				if (output(MnistDatabase.testImages.get(i).getDataFloat()))count++;
			}else{
				if (!output(MnistDatabase.testImages.get(i).getDataFloat()))count++;
			}
			total++;
		}
		double accuracy = (double)count/total;
		if(accuracy>maxAccuracy)
		{
			maxAccuracy=accuracy;
			System.out.println(accuracy);
		}
	}
	private boolean output(float[] dataFloat) {
		float sum=0;
		for (int i = 0; i < weights.length-1; i++) {
			sum+=weights[i]*dataFloat[i];
		}
		sum+=weights[weights.length-1];
		return sum>0?true:false;
	}
	
	private void release() {
		memWeights.release();
		memImages.release();
		memWeightsDeltas.release();
		memOutputs.release();
		program.release();
	}
	
	public static void main(String[] args) throws Exception {
		int noIterations=3000;
		Perceptron  p = new Perceptron(784);
		long t0=System.currentTimeMillis();
		p.runAllTrainingSet(noIterations);
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImage(p.getImage());
		System.out.println(1000.*noIterations/t+" it/sec; "+t/1000.+" seconds");
		p.test();
		p.release();
	}

}
