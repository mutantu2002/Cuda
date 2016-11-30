package home.mutant.cuda.model;


import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import static jcuda.driver.JCudaDriver.*;

import jcuda.Pointer;

public class MemoryDouble {
	Program program;
	CUdeviceptr memObj;
	double[] src;
	
	public MemoryDouble(Program program) {
		super();
		this.program = program;
	}

	public void add(double[] src){
		this.src = src;
		memObj = new CUdeviceptr();
        cuMemAlloc(memObj, src.length * Sizeof.DOUBLE);
        copyHtoD();
	}
	
	public int copyDtoH()
	{
        return cuMemcpyDtoH(Pointer.to(src), memObj,src.length * Sizeof.DOUBLE);	
	}
	public int copyHtoD()
	{
		return cuMemcpyHtoD(memObj, Pointer.to(src),src.length * Sizeof.DOUBLE);
	}
	public double[] getSrc() {
		return src;
	}
	public CUdeviceptr gemMemObject() {
		return memObj;
	}
	public void release()
	{
		cuMemFree(memObj);;
	}
}
