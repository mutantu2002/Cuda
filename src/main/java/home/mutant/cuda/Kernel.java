package home.mutant.cuda;

import static org.jocl.CL.*;

import java.util.ArrayList;
import java.util.List;

import org.jocl.Sizeof;
import org.jocl.cl_kernel;

import jcuda.Pointer;
import jcuda.driver.*;
import static jcuda.driver.JCudaDriver.*;

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
	public int run(int globalworkSize, int localWorksize)
	{
		return cuLaunchKernel(function,
				(int)Math.ceil((double)globalworkSize / localWorksize),  1, 1,      // Grid dimension
				localWorksize, 1, 1,      // Block dimension
				0, null,               // Shared memory size and stream
				Pointer.to(arguments.toArray(new Pointer[0])), null // Kernel- and extra parameters
		);
	}
}
