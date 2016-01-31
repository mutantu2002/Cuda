package home.mutant.cuda.runners.steps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import home.mutant.cuda.Kernel;
import home.mutant.cuda.MemoryFloat;
import home.mutant.cuda.MemoryInt;
import home.mutant.cuda.Program;
import home.mutant.dl.models.Image;
import home.mutant.dl.models.ImageFloat;
import home.mutant.dl.utils.kmeans.Kmeans;

public class ClusterImages {
	List<Image> images;
	List<Integer> labels;
	List<Image> clusters = new ArrayList<>();
	List<Integer> clusterLabels = new ArrayList<>();
	
	float[] allImages;
	float[] clustersCenters;
	int[] clustersUpdates;
	
	int imageSize;
	int noIterations;
	int noClusters;
	
	Program program;
	
	MemoryFloat memClusters ;
	MemoryFloat memImages;
	MemoryInt memUpdates;
	Kernel updateCenters;
	
	public ClusterImages(List<Image> images, List<Integer> labels,int noClusters,int noIterations) {
		super();
		this.images = images;
		this.labels = labels;
		this.noClusters = noClusters;
		this.noIterations = noIterations;
		imageSize = images.get(0).getDataFloat().length;
		
		allImages= new float[imageSize*images.size()];
		clustersCenters = new float[imageSize*noClusters];
		clustersUpdates = new int[images.size()];
		
		randomizeCentersFromImages(clustersCenters);
	}
	
	public List<Image> getClusters() {
		return clusters;
	}
	public List<Integer> getClusterLabels() {
		return clusterLabels;
	}
	public void cluster(){
		prepareOpenCl();
		for (int iteration=0;iteration<noIterations;iteration++){
			
			updateCenters.run(images.size(), 1000);
			program.finish();
			memUpdates.copyDtoH();
			reduceCenters();
			memClusters.copyHtoD();
		}
		
		memClusters.copyDtoH();
		updateClustersLabels();
		contructImageClusters();
	}
	public void test(List<Image> testImages, List<Integer> testLabels){
		memImages.release();
		memUpdates.release();
		allImages= new float[imageSize*testImages.size()];
		clustersUpdates = new int[testImages.size()];
		for (int i=0;i<testImages.size();i++){
			System.arraycopy(testImages.get(i).getDataFloat(), 0, allImages, i*(imageSize), imageSize);
		}
		memImages = new MemoryFloat(program);
		memImages.add(allImages);
		
		memUpdates = new MemoryInt(program);
		memUpdates.add(clustersUpdates);
		updateCenters.addArgument(memImages);
		updateCenters.addArgument(memUpdates);
		
		updateCenters.run(testImages.size(), 1000);
		program.finish();
		memUpdates.copyDtoH();
		int count=0;
		for (int i = 0; i < clustersUpdates.length; i++) {
			count+=clusterLabels.get(clustersUpdates[i])==testLabels.get(i)?1:0;
		}
		System.out.println(100.*count/testImages.size()+"%");
	}
	private void updateClustersLabels(){
		List<HashMap<Integer, Integer>> clusterHash = new ArrayList<>();
		for (int i=0;i<noClusters;i++) {
			clusterHash.add(new HashMap<>());
		}
		for (int i=0;i<images.size();i++){
			int currentClusterIndex = clustersUpdates[i];
			int currentLabel = labels.get(i);
			HashMap<Integer, Integer> currentMembers =  clusterHash.get(currentClusterIndex);
			if (currentMembers.get(currentLabel)==null){
				currentMembers.put(currentLabel, 0);
			}
			currentMembers.put(currentLabel, currentMembers.get(currentLabel)+1);
		}
		for (int i=0;i<noClusters;i++) {
			clusterLabels.add(Kmeans.getMaxKeyHash(clusterHash.get(i)));
		}
	}
	private void reduceCenters() {
		int[] clustersMembers = new int[noClusters];
		Arrays.fill(clustersCenters, 0);
		for (int i=0;i<images.size();i++){
			int toUpdate = clustersUpdates[i];
			clustersMembers[toUpdate]++;
			for (int j=0;j<imageSize;j++){
				clustersCenters[toUpdate*imageSize+j]+=allImages[i*imageSize+j];
			}
		}
		for (int i=0;i<noClusters;i++){
			if (clustersMembers[i]==0) continue;
			for (int j=0;j<imageSize;j++){
				clustersCenters[i*imageSize+j]/=clustersMembers[i];
			}
		}
	}
	private void prepareOpenCl(){
		Map<String, Object> params = new HashMap<>();
		params.put("imageSize", imageSize);
		program = new Program("src/main/resources/Kmeans2.ptx",params);		
		
		memClusters = new MemoryFloat(program);
		memClusters.add(clustersCenters);
		
		for (int i=0;i<images.size();i++){
			System.arraycopy(images.get(i).getDataFloat(), 0, allImages, i*(imageSize), imageSize);
		}
		memImages = new MemoryFloat(program);
		memImages.add(allImages);
		
		memUpdates = new MemoryInt(program);
		memUpdates.add(clustersUpdates);

		updateCenters = new Kernel(program, "updateCenters");
		updateCenters.addArgument(memClusters);
		updateCenters.addArgument(memImages);
		updateCenters.addArgument(memUpdates);
		updateCenters.addArgument(noClusters);
		
		memImages.copyHtoD();
		
	}
	public void releaseOpenCl(){
		memClusters.release();
		memImages.release();
		memUpdates.release();
		program.release();
	}
	private void randomizeCentersFromImages(float[] clustersCenters) {
		for (int i=0;i<noClusters;i++){
			System.arraycopy(images.get((int) (Math.random()*images.size())).getDataFloat(), 0, clustersCenters, i*(imageSize), imageSize);
		}
	}
	private void contructImageClusters(){
		for (int i=0;i<noClusters;i++) {
			Image image = new ImageFloat(imageSize);
			System.arraycopy(memClusters.getSrc(), i*imageSize, image.getDataFloat(), 0, imageSize);
			clusters.add(image);
		}
	}
}
