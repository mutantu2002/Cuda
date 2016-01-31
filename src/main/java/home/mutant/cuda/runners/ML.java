package home.mutant.cuda.runners;
import home.mutant.cuda.runners.steps.ClusterImages;
import home.mutant.dl.ui.ResultFrame;
import home.mutant.dl.utils.MnistDatabase;
import home.mutant.dl.utils.MnistDatabase.TYPE;

public class ML {

	public static void main(String[] args) throws Exception {
		MnistDatabase.IMAGE_TYPE = TYPE.FLOAT;
		MnistDatabase.loadImages();

		long t0=System.currentTimeMillis();
		int noIterations=10;
		ClusterImages  ci = new ClusterImages(MnistDatabase.trainImages, MnistDatabase.trainLabels, 100, noIterations);
		ci.cluster();
		ci.releaseOpenCl();
		long t=System.currentTimeMillis()-t0;
		ResultFrame frame = new ResultFrame(1600, 800);
		frame.showImages(ci.getClusters());
		System.out.println(1000.*noIterations/t+" it/sec");
	}
}
