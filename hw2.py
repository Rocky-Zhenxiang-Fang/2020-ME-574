import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
from map_parallel import sArray
import time

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

def f_CPU(x, r):
	'''
	Execute 1 iteration of the logistic map on CPU
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

	if i < n:
		# function body
		x_old = x
		for j in range(transient):
			x_new = f(x_old,r[i])
			x_old = x_new
		for j in range(steady):
			x_new = f(x_old,r[i])
			x_old = x_new
			ss[j][i] = x_old
		
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
	d_ss = cuda.device_array([steady, n], dtype = r.dtype)

	TPB = 32
	gridDim = (n+TPB-1)//TPB
	blockDim = TPB
	logistic_map_kernel[gridDim,blockDim](d_ss, d_r, x, transient, steady)

	return d_ss.copy_to_host()

def serial_logistic_map(r,x,transient,steady):
	ss = np.zeros((steady, r.size), dtype = r.dtype)
	n = r.size 

	for i in range(n):
		# This for loop changes the parameter r
		x_old = x
		for j in range(transient):
			#This for loop moves the system a step foward
			x_new = f_CPU(x_old,r[i])
			x_old = x_new
		for j in range(steady):
			#This for loop moves the system a step foward
			x_new = f_CPU(x_old,r[i])
			x_old = x_new
			ss[j][i] = x_old
		
	return ss

@cuda.jit(device = True)
def iteration_count(cx, cy, dist, itrs):
	'''
	Computed number of Mandelbrot iterations

	Arguments:
		cx, cy: float64 parameter values
		dist: float64 escape threshold
		itrs: int iteration count limit
	'''
	x_old = 0
	y_old = 0
	radius = (x_old**2+y_old**2)**0.5
	for i in range(itrs):
		radius = (x_old**2+y_old**2)**0.5
		if radius < dist:
			x_new = x_old**2-y_old**2+cx
			y_new = 2*x_old*y_old+cy
			x_old = x_new
			y_old = y_new
		else:
			break
	return i


@cuda.jit
def mandelbrot_kernel(out, cx, cy, dist, itrs):
	'''
	Kernel for parallel computation of Mandelbrot iteration counts

	Arguments:
		out: 2D numpy device array for storing computed iteration counts
		cx, cy: 1D numpy device arrays of parameter values
		dist: float64 escape threshold
		itrs: int iteration count limit
	'''
	i,j = cuda.grid(2)

	if i < cx.size and j < cy.size:
		# function body
		out[j][i] = iteration_count(cx[i],cy[j],dist,itrs)



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
	d_cx = cuda.to_device(cx)
	d_cy = cuda.to_device(cy)
	d_out = cuda.device_array([cx.size, cy.size], dtype = cx.dtype)
	
	n_x = cx.size
	n_y = cy.size
	TPB_x = 32
	TPB_y = 32
	gridDim = ((n_x+TPB_x-1) // TPB_x, (n_y+TPB_y-1) // TPB_y)
	blockDim = (TPB_x,TPB_y)
	mandelbrot_kernel[gridDim, blockDim](d_out,d_cx, d_cy,dist,itrs)

	return d_out.copy_to_host()

def escape(cx, cy, dist,itrs, x0=0, y0=0):
	'''
	Compute the number of iterations of the logistic map, 
	f(x+j*y)=(x+j*y)**2 + cx +j*cy with initial values x0 and y0 
	with default values of 0, to escape from a cirle centered at the origin.

	inputs:
		cx - float: the real component of the parameter value
		cy - float: the imag component of the parameter value
		dist: radius of the circle
		itrs: int max number of iterations to compute
		x0: initial value ofRR x; default value 0
		y0: initial value of y; default value 0
	returns:
		an int scalar interation count
	'''
	x_old = x0
	y_old = y0
	radius = (x_old**2+y_old**2)**0.5
	for i in range(itrs):
		if radius < dist:
			x_new = x_old**2-y_old**2+cx
			y_new = 2*x_old*y_old+cy
			x_old = x_new
			y_old = y_new
			radius = (x_old**2+y_old**2)**0.5
		else:
			break
	return i

def serial_mandelbrot(cx,cy,dist,itrs):

	"""
	Compute escape iteration counts for an array of parameter values
	input:
		cx - array: 1d array of real part of parameter
		cy - array: 1d array of imaginary part of parameter
		dist - float: radius of circle for escape
	output:
		a 2d array of iteration count for each parameter value (indexed pair of values cx, cy)	
	"""
	y_ = np.zeros((len(cx),len(cy)))
	for i in range(len(cx)):
		for j in range(len(cy)):
			y_[j][i] = escape(cx[i],cy[j],dist,itrs)
            
	return y_

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

	#The largest array allowed is 134217728 by hand caculation. However, in real world, it can only afford 110501888
	#map_64()

	#PASTE YOUR ERROR MESSAGES HERE#
	#numba.cuda.cudadrv.driver.CudaAPIError: [2] Call to cuMemAlloc results in CUDA_ERROR_OUT_OF_MEMORY
	
	#Problem 3
	print("3a) The for loops and their explaination are listed in the function")
	print("3b) The first loop that changes r[i] assign a new parameter r in each simulation. Each iteration are independent of one another in this for loop.The second and third for loop moves the simulation a step foward. Each iteration will affect the next iteration.")

	r = np.linspace(0,4,num = 1000)
	x0 = 0.5
	trans = 992
	steady = 8

	start = time.time()
	ss = parallel_logistic_map(r,x0,trans,steady)
	end = time.time()
	ss_time = end-start
	print("The time of parallel computation is " + str(ss_time) +"sec")

	start = time.time()
	ss2 = serial_logistic_map(r,x0,trans,steady)
	end = time.time()
	ss2_time = end-start
	print("The time of serial computation is " + str(ss2_time) +"sec")
	print("The acceleration factor is " + str(ss2_time / ss_time))

	### plots for testing
	#plt.figure()
	#for i in range(steady):
	#	plt.scatter(r,ss[i,:],s = 0.1)
	#plt.show()

	#plt.figure()
	#for i in range(steady):
	#	plt.scatter(r,ss2[i,:],s = 0.1)
	#plt.show()
	###
	
	
	#Problem 4
	print("4a)In serial code, for loops can bbe used to assign differenc cx or cy value, or to move the simulation. For loops that are used to assign different cx and cy are independent to one another.")
	print("4b)I verified the result by comparing two plots below")
	print("4c)My finest resolution of the 2d grid of cx and cy is 186*186, anything bigger than this will result an error: Call to cuMemcpyDtoH results in CUDA_ERROR_LAUNCH_FAILED")
	print("4c)My largest square block of TPBs that can run on my gpu is 32*32, anything bigger than this will result an error: Call to cuLaunchKernel results in CUDA_ERROR_INVALID_VALUE")
	cx = np.linspace(-2.5,2.5,100)
	cy = np.linspace(-2.5,2.5,100)
	dist = 2.5
	itrs = 256

	y_ = parallel_mandelbrot(cx,cy,dist,itrs)
	y_2 = serial_mandelbrot(cx,cy,dist,itrs)

	plt.pcolormesh(y_)
	plt.show()
	plt.pcolormesh(y_2)
	plt.show()