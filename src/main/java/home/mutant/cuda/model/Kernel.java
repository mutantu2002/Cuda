package home.mutant.cuda.model;

import static jcuda.driver.JCudaDriver.*;

import java.util.ArrayList;
import java.util.List;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class Kernel {
	Program program;
	CUfunction function;
	List<Pointer> arguments = new ArrayList<>();
	
	public Kernel(Program program, String functionName) {
		super();
		this.program = program;
		function = new CUfunction();
		cuModuleGetFunction(function, program.getModule(), functionName);
	}
	
	public void addArgument(MemoryDouble memory){
		arguments.add(Pointer.to(memory.gemMemObject()));
	}
	
	public void addArgument(MemoryFloat memory){
		arguments.add(Pointer.to(memory.gemMemObject()));
	}
	
	public void addArgument(MemoryInt memory){
		arguments.add(Pointer.to(memory.gemMemObject()));
	}	
	
	public void addArgument(int value){
		arguments.add(Pointer.to(new int[]{value}));
	}
	
	public void setArgument(int value, int index){
		arguments.set(index,Pointer.to(new int[]{value}));
	}	
	public void setArguments(MemoryDouble ... memories){
		for (MemoryDouble memory:memories) {
			addArgument(memory);
		}
	}
	public void setArguments(MemoryFloat ... memories){
		for (MemoryFloat memory:memories) {
			addArgument(memory);
		}
	}
	public int run(int gridDimX, int blockDimX)
	{
		int res= cuLaunchKernel(function,
				gridDimX,  1, 1,      // Grid dimension
				blockDimX, 1, 1,      // Block dimension
				0, null,               // Shared memory size and stream
				Pointer.to(arguments.toArray(new Pointer[0])), null // Kernel- and extra parameters
		);
		cuCtxSynchronize();
		return res;
	}
}
