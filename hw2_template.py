import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
from map_parallel import sArray

def gpu_total_memory():
	'''
	Query the GPU's properties via Numba to obtain the total memory of the device.
	'''
	context = cuda.current_context()
	memory = context.get_memory_info()
	return(memory[1])

def gpu_compute_capability():
	'''
	Query the GPU's properties via Numba to obtain the compute capability of the device.
	'''
	device = cuda.get_current_device()
	compute = device.compute_capability
	
	return(compute)

def gpu_name():
	'''
	Query the GPU's properties via Numba to obtain the name of the device.
	'''
	device = cuda.get_current_device()
	name = device.name
	return(name)

def max_float64s():
	'''
	Compute the maximum number of 64-bit floats that can be stored on the GPU
	'''
	context = cuda.current_context()
	memory = context.get_memory_info()
	size = 64
	number = math.floor(memory[1]*8 / size)
	return(number)

def map_64():
	'''
	Execute the map app modified to use 64-bit floats
	'''
	N = 110501888
	x = np.linspace(0,1,N,dtype = np.float64)
	y = sArray(x)

	plt.plot(x,y)
	plt.show()

@cuda.jit(device = True)
def f(x, r):
	'''
	Execute 1 iteration of the logistic map
	'''
	return r*x*(1 - x)

@cuda.jit()
def logistic_map_kernel(ss, r, x, transient, steady):
	'''
	Kernel for parallel iteration of logistic map

	Arguments:
		ss: 2D numpy device array to store steady state iterates for each value of r
		r: 1D  numpy device array of parameter values
		x: float initial value
		transient: int number of iterations before storing results
		steady: int number of iterations to store
	'''

	i = cuda.grid(1)
	n = r.size
	y = np.zeros(steady, dtype = np.float32)	
	if i < n:
		# function body
		x_old = x
		for j in range(transient):
			x_new = f(x_old,r[i])
			x_old = x_new
		for j in range(steady):
			y[j] = x_old
			x_new = f(x_old,r[i])
			x_old = x_new
		
		ss[i] = y[i]

def parallel_logistic_map(r, x, transient, steady):
	'''
	Parallel iteration of the logistic map

	Arguments:
		r: 1D numpy array of float64 parameter values
		x: float initial value
		transient: int number of iterations before storing results
		steady: int number of iterations to store
	Return:
		2D numpy array of steady iterates for each entry in r
	'''
	n = r.size
	d_r = cuda.to_device(r)
	d_ss = cuda.device_array([r.size, steady], dtype = r.dtype)

	TPB = 32
	gridDim = (n+TPB-1)//TPB
	blockDim = TPB
	logistic_map_kernel[gridDim,blockDim](d_ss, d_r, x, transient, steady)

	return d_ss.copy_to_host

#@cuda.jit(device = True)
def iteration_count(cx, cy, dist, itrs):
	'''
	Computed number of Mandelbrot iterations

	Arguments:
		cx, cy: float64 parameter values
		dist: float64 escape threshold
		itrs: int iteration count limit
	'''
	pass

#@cuda.jit
def mandelbrot_kernel(out, cx, cy, dist, itrs):
	'''
	Kernel for parallel computation of Mandelbrot iteration counts

	Arguments:
		out: 2D numpy device array for storing computed iteration counts
		cx, cy: 1D numpy device arrays of parameter values
		dist: float64 escape threshold
		itrs: int iteration count limit
	'''
	pass

def parallel_mandelbrot(cx, cy, dist, itrs):
	'''
	Parallel computation of Mandelbrot iteration counts

	Arguments:
		cx, cy: 1D numpy arrays of parameter values
		dist: float64 escape threshold
		itrs: int iteration count limit
	Return:
		2D numpy array of iteration counts
	'''
	pass

if __name__ == "__main__":
	
	#Problem 1
	print("GPU memory in GB: ", gpu_total_memory()/1024**3)
	print("Compute capability (Major, Minor): ", gpu_compute_capability())
	print("GPU Model Name: ", gpu_name())
	print("Max float64 count: ", max_float64s())

	#PASTE YOUR OUTPUT HERE#
	#GPU memory in GB:  2.0
	#Compute capability (Major, Minor):  (3, 0)
	#GPU Model Name:  b'GeForce GTX 760M'
	#Max float64 count:  268435456
	
	#Problem 2
	#The largest array allowed is 110501888
	map_64()

	#PASTE YOUR ERROR MESSAGES HERE#
	#numba.cuda.cudadrv.driver.CudaAPIError: [2] Call to cuMemAlloc results in CUDA_ERROR_OUT_OF_MEMORY
	
	#Problem 3

	#Problem 4
