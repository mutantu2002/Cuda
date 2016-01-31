package home.mutant.cuda;


import static jcuda.driver.JCudaDriver.*;
import java.util.Map;
import jcuda.driver.*;

public class Program 
{
	CUcontext context;
	CUdevice device;
	CUmodule module;
	
	public Program(String source)
	{
		this(source,null);
	}
	public Program(String ptxFileName, Map<String, Object> params)
	{
		JCudaDriver.setExceptionsEnabled(true);
		
        cuInit(0);
        device = new CUdevice();
        cuDeviceGet(device, 0);
        context = new CUcontext();
        cuCtxCreate(context, 0, device);
        
        module = new CUmodule();
        cuModuleLoad(module, ptxFileName);
	}
	
	public void release()
	{
        cuModuleUnload(module);
        cuCtxDestroy(context);
	}
	public int finish()
	{
		return cuCtxSynchronize();
	}

	public CUcontext getContext() {
		return context;
	}
	public CUmodule getModule() {
		return module;
	}
	public void setModule(CUmodule module) {
		this.module = module;
	}
}
