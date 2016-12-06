package home.mutant.cuda.perceptron;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import home.mutant.cuda.model.Kernel;
import home.mutant.cuda.model.MemoryFloat;
import home.mutant.cuda.model.Program;
import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;

public class TwoLayersPerceptron {
	public static final int NO_HIDDEN_NEURONS = 15;
	float weights[];
	float deltas[];
	float images[];
	float outputs[];
	
	float learningRate = 1;
	
	int noThreads = 32*125;
	int noImages=60000;
	int imageSize;
	int trainLabel = 0;
	
	Program program;
	
	MemoryFloat memWeights ;
	MemoryFloat memWeightsDeltas;
	MemoryFloat memImages ;
	MemoryFloat memOutputs;
	Kernel deltasBatch;

	int offsetInputImages;
	
	ResultFrame frame = new ResultFrame(600, 600);
	
	public TwoLayersPerceptron(int size) throws IOException{
		imageSize = size;
		int inputsPerBatch = (int)Math.ceil(((double)noImages)/noThreads);
		weights = new float[(size+1)*NO_HIDDEN_NEURONS+NO_HIDDEN_NEURONS+1];
		for (int i = 0; i < weights.length; i++) {
			weights[i]=(float) (1-2*Math.random());
		}
		deltas = new float[((size+1)*NO_HIDDEN_NEURONS+NO_HIDDEN_NEURONS+1)*noThreads];
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
		
		program = new Program("src/main/resources/TwoLayersPerceptron.ptx",null);	
		
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
			//showLastWeights();
		}
	}

	private void showLastWeights() {
		for (int hidden = 0; hidden < NO_HIDDEN_NEURONS; hidden++) {
			System.out.println(weights[(imageSize+1)*NO_HIDDEN_NEURONS+hidden]);
		}
	}
	
	public void modifyWeightsFromDelta(){
		for (int i = 0; i < weights.length; i++) {
			float delta=0;
			for(int noBatch=0;noBatch<noThreads;noBatch++){
				delta+=deltas[((imageSize+1)*NO_HIDDEN_NEURONS+NO_HIDDEN_NEURONS+1)*noBatch+i];
			}
			weights[i]+=delta*learningRate;
		}
	}
	
	public List<Image> getImages(){
		List<Image> images = new ArrayList<>();
		for (int hidden = 0; hidden < NO_HIDDEN_NEURONS; hidden++) {
			images.add(getImage(hidden*(imageSize+1), hidden*(imageSize+1)+imageSize));
		}
		return images;
	}
	
	private Image getImage(int i1, int i2){
		Image image = new ImageFloat(i2-i1);
		double max = -1*Double.MAX_VALUE;
		double min = Double.MAX_VALUE;
		for (int i = i1; i < i2; i++) {
			if (weights[i]>max)max=weights[i];
			if (weights[i]<min)min=weights[i];
		}
		max=255/(max-min);
		for (int i = i1; i < i2; i++) {
			image.getDataFloat()[i-i1]=(float) ((weights[i]-min)*max);
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
		System.out.println((double)count/total);
		frame.showImages(getImages());
	}
	private boolean output(float[] dataFloat) {
		float sum=0;
		float[] activation = new float[NO_HIDDEN_NEURONS];
		for (int hidden = 0; hidden < NO_HIDDEN_NEURONS; hidden++) {
			sum=0;
			for (int i = 0; i < imageSize; i++) {
				sum+=weights[(imageSize+1)*hidden+i]*dataFloat[i];
			}
			sum+=weights[(imageSize+1)*hidden+imageSize];
			if(sum>0)activation[hidden]=1;
			else activation[hidden]=0;
			//activation[hidden]=sum/(1+Math.abs(sum));
		}
		sum=0;
		for (int hidden = 0; hidden < NO_HIDDEN_NEURONS; hidden++) {
			sum+=weights[(imageSize+1)*NO_HIDDEN_NEURONS+hidden]*activation[hidden];
		}
		sum+=weights[(imageSize+1)*NO_HIDDEN_NEURONS+NO_HIDDEN_NEURONS];
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
		int noIterations=1000;
		TwoLayersPerceptron  p = new TwoLayersPerceptron(784);
		long t0=System.currentTimeMillis();
		p.runAllTrainingSet(noIterations);
		long t=System.currentTimeMillis()-t0;
		System.out.println(1000.*noIterations/t+" it/sec; "+t/1000.+" seconds");
		p.release();
	}
}
