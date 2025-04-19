# Project Guide :rocket:

GPU programming usually involves using specific APIs or frameworks to write code so as to 
be able to utilize the powerful parallel computing capabilities of GPUs. One of the most commonly 
used GPU programming models is NVIDIA's CUDA (Compute Unified Device Architecture). Through CUDA, 
developers can use C++ language extensions to write programs that can be executed on NVIDIA GPUs.
And there are some core concepts in CUDA programming that are worth learning:
___Host___: Refers to a computer system running a CPU, which is responsible for executing regular 
			program code and managing communication with the Device.  
___Device___: Refers to the GPU connected to this computer, which is specifically used for performing 
			highly parallelized computing tasks.  
___Kernel Function___: In CUDA, a kernel function refers to a function that is concurrently executed 
			on the GPU and is identified by the global declarator. When a kernel function is called, 
			in fact, one or more thread blocks are launched to execute the function.  
___Thread hierarchy___: The CUDA programming model is based on a hierarchical thread organization structure, 
including grid, block, and thread. ___Grid___: Composed of multiple Blocks. Each Grid corresponds to one 
kernel function call. ___Block___: Contains multiple Threads. All threads share the shared memory within 
the same Block. Block is the basic unit of scheduling. ___Thread___: The most basic execution unit that 
can independently execute instructions in the kernel function.

## 1.CUDA Tutorials

[NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?spm=5176.28103460.0.0.29b01db8ILLOEG)£ºThis is one of the most authoritative resources for learning CUDA programming, 
covering all topics from basic to advanced.  
