Emilio Lopez, elopez1
Sujay Karve, sskarve
===================================================

We first refactored the code to combine cpu_fftx and cpu_ifftx into one function, passing in a scaling factor (1 in the fftx case and size_y in the ifftx case). We did something similar for the functions running the FT in the y-direction. To improve speed we used OpenMp for loops with static schedule and no chunk size on the outer for loop for each of these methods. This way, each thread would take a chunk of rows or columns of the image matrix to run the respective FT or inverse FT on, in parallel. Static scheduling with no chunk size means that each thread will take size_x/NUM_THREADS or size_y/NUM_THREADS of the rows or columns (with some taking one more or one less that the other threads if that fraction is not whole). These rows or columns are consecutive, so the program can take advantage of memory locality of the image data.

The only variables that are shared in the parallel region are the real_image and imag_image matrices, since the threads need to modify these. Everything else is either private or firstprivate if it was initialized to a value we need. By minimizing how much data we share, we minimize the overhead of fetching and updating shared data.

Lastly, we also used a parallel for region in the cpu_filter function to parallelize passing the low-pass filter over the values of the image matrices.
