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
	
	float learningRate = 1;
	
	int noBatches = 1000;
	int noImages=60000;
	int imageSize;
	int trainLabel = 0;
	
	Program program;
	
	MemoryFloat memWeights ;
	MemoryFloat memWeightsDeltas;
	MemoryFloat memImages ;
	MemoryFloat memOutputs;
	Kernel deltasBatch;
	
	public Perceptron(int size) throws IOException{
		weights = new float[size+1];
		deltas = new float[(size+1)*noBatches];
		imageSize = size;
		
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();
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
		deltasBatch.addArgument(noImages/noBatches);
		deltasBatch.addArgument(imageSize);
	}
	private void run() {
		for(int it=0;it<200;it++){
			deltasBatch.run(noBatches, noBatches);
			memWeightsDeltas.copyDtoH();
			modifyWeightsFromDelta();
		}
	}

	public void modifyWeightsFromDelta(){
		for (int i = 0; i < weights.length; i++) {
			float delta=0;
			for(int noBatch=0;noBatch<noBatches;noBatch++){
				delta+=deltas[(imageSize+1)*noBatch+i]/(noImages/noBatches);
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
	
	public static void main(String[] args) throws Exception {
		
		int noIterations=200;
		Perceptron  p = new Perceptron(784);
		long t0=System.currentTimeMillis();
		p.run();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImage(p.getImage());
		System.out.println(1000.*noIterations/t+" it/sec");
	}
}
