package home.mutant.cuda.model;

import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import static jcuda.driver.JCudaDriver.*;

import jcuda.Pointer;


public class MemoryFloat {
	Program program;
	CUdeviceptr memObj;
	float[] src;
	
	public MemoryFloat(Program program) {
		super();
		this.program = program;
	}

	public void add(float[] src){
		this.src = src;
		memObj = new CUdeviceptr();
        cuMemAlloc(memObj, src.length * Sizeof.FLOAT);
        copyHtoD();
	}
	
	public int copyDtoH()
	{
        return cuMemcpyDtoH(Pointer.to(src), memObj,src.length * Sizeof.FLOAT);	
	}
	public int copyHtoD()
	{
		return cuMemcpyHtoD(memObj, Pointer.to(src),src.length * Sizeof.FLOAT);
	}
	public float[] getSrc() {
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
