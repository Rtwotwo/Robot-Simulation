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
<p align="center">
<src img="./assets/gpu_cpu.jpg", alt="GPU devotes more transistors to data processing" >
</p>

## 1.CUDA Tutorials

___Official documents and tutorials___  
[NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/): This is one of the most 
authoritative resources for learning CUDA programming, covering all topics from basic to advanced.  
[NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples?spm=5176.28103460.0.0.29b01db8ILLOEG): A large number of 
sample codes provided officially are very suitable for beginners to learn CUDA programming through practice.  
___CUDA tutorials and projects on GitHub___  
[CUDA by Example](https://github.com/jeffra/cuda-by-example?spm=5176.28103460.0.0.29b01db8ILLOEG): The source code of 
this book provides many practical examples and is suitable for beginners to quickly get started.  
[CUDA Tutorials](https://github.com/parallel-forall/code-samples?spm=5176.28103460.0.0.29b01db8ILLOEG): NVIDIA's official 
collection of code examples covers various application scenarios.  
___Other online resources___  
[Udacity's Intro to Parallel Programming Course](https://www.udacity.com/course/): This is a free online course focusing 
on CUDA programming. It teaches the basics of CUDA through video lectures and homework exercises.  
[Stack Overflow](https://stackoverflow.com/questions/tagged/cuda?spm=5176.28103460.0.0.29b01db8ILLOEG): For problems 
encountered, you can search or ask questions on Stack Overflow. The community is very active and you can get timely help.  

