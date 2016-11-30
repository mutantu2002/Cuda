package home.mutant.cuda.model;

import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import static jcuda.driver.JCudaDriver.*;

import jcuda.Pointer;


public class MemoryInt {
	Program program;
	CUdeviceptr memObj;
	int[] src;
	
	public MemoryInt(Program program) {
		super();
		this.program = program;
	}

	public void add(int[] src){
		this.src = src;
		memObj = new CUdeviceptr();
        cuMemAlloc(memObj, src.length * Sizeof.INT);
        copyHtoD();
	}
	
	public int copyDtoH()
	{
        return cuMemcpyDtoH(Pointer.to(src), memObj,src.length * Sizeof.INT);	
	}
	public int copyHtoD()
	{
		return cuMemcpyHtoD(memObj, Pointer.to(src),src.length * Sizeof.INT);
	}
	public int[] getSrc() {
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