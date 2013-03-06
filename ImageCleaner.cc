#include "ImageCleaner.h"
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <omp.h>

#define PI	3.14159265
#define TWO_PI  2 * PI
#define NEGATIVE_TWO_PI  -1 * TWO_PI

void cpu_fftx(float *real_image, float *imag_image, int size_x, int size_y, int scaling_factor)
{
  // Create some space for storing temporary values
  float *realOutBuffer;// = new float[size_x];
  float *imagOutBuffer;// = new float[size_x];
  // Local values
  float fft_real, fft_imag, term, term_coefficient, term_coefficient_times_size_y, real_image_value, imag_image_value;
  unsigned int x, y, n, x_offset, x_offset_plus_n, x_offset_plus_y;
  #pragma omp parallel default(none) shared(real_image, imag_image) firstprivate(size_x, size_y, scaling_factor) private(realOutBuffer, imagOutBuffer, fft_real, fft_imag, term, term_coefficient, term_coefficient_times_size_y, real_image_value, imag_image_value, x, y, n, x_offset, x_offset_plus_n, x_offset_plus_y)
  {
  realOutBuffer = new float[size_x];
  imagOutBuffer = new float[size_x];
  #pragma omp for schedule(static)
  for(x = 0; x < size_x; x++)
  {
    x_offset = x*size_x; // For serial speedup
    term_coefficient_times_size_y = NEGATIVE_TWO_PI / size_y;
    for(y = 0; y < size_y; y++)
    {
      // Compute the value for this index
      realOutBuffer[y] = 0.0f;
      imagOutBuffer[y] = 0.0f;
      term_coefficient = NEGATIVE_TWO_PI * y / size_y; // For serial speedup
      // Compute the frequencies for this index
      for(n = 0; n < size_y; n++)
      {
        x_offset_plus_n = x_offset + n;
        real_image_value = real_image[x_offset_plus_n];
        imag_image_value = imag_image[x_offset_plus_n];
	term = term_coefficient * n;
	fft_real = cos(term);
	fft_imag = sin(term);
	realOutBuffer[y] += (real_image_value * fft_real) - (imag_image_value * fft_imag);
	imagOutBuffer[y] += (imag_image_value * fft_real) + (real_image_value * fft_imag);
      }
    }
    // Write the buffer back to were the original values were
    for(y = 0; y < size_y; y++)
    {
      x_offset_plus_y = x_offset + y;
      real_image[x_offset_plus_y] = realOutBuffer[y] / scaling_factor;
      imag_image[x_offset_plus_y] = imagOutBuffer[y] / scaling_factor;
    }
  }
  delete [] realOutBuffer;
  delete [] imagOutBuffer;
  }
  // Reclaim some memory
  //delete [] realOutBuffer;
  //delete [] imagOutBuffer;
}

void cpu_ffty(float *real_image, float *imag_image, int size_x, int size_y)
{
  // Allocate some space for temporary values
  float *realOutBuffer = new float[size_y];
  float *imagOutBuffer = new float[size_y];
  float *fft_real = new float[size_x];
  float *fft_imag = new float[size_x];
  for(unsigned int y = 0; y < size_y; y++)
  {
    for(unsigned int x = 0; x < size_x; x++)
    {
      // Compute the frequencies for this index
      for(unsigned int n = 0; n < size_y; n++)
      {
	float term = -2 * PI * x * n / size_x;
	fft_real[n] = cos(term);
	fft_imag[n] = sin(term);
      }

      // Compute the value for this index
      realOutBuffer[x] = 0.0f;
      imagOutBuffer[x] = 0.0f;
      for(unsigned int n = 0; n < size_x; n++)
      {
	realOutBuffer[x] += (real_image[n*size_x + y] * fft_real[n]) - (imag_image[n*size_x + y] * fft_imag[n]);
	imagOutBuffer[x] += (imag_image[n*size_x + y] * fft_real[n]) + (real_image[n*size_x + y] * fft_imag[n]);
      }
    }
    // Write the buffer back to were the original values were
    for(unsigned int x = 0; x < size_x; x++)
    {
      real_image[x*size_x + y] = realOutBuffer[x];
      imag_image[x*size_x + y] = imagOutBuffer[x];
    }
  }
  // Reclaim some memory
  delete [] realOutBuffer;
  delete [] imagOutBuffer;
  delete [] fft_real;
  delete [] fft_imag;
}

// This is the same as the thing about it, but it includes a scaling factor
void cpu_iffty(float *real_image, float *imag_image, int size_x, int size_y)
{
  // Create some space for storing temporary values
  float *realOutBuffer = new float[size_y];
  float *imagOutBuffer = new float[size_y];
  float *fft_real = new float[size_x];
  float *fft_imag = new float[size_x];
  for(unsigned int y = 0; y < size_y; y++)
  {
    for(unsigned int x = 0; x < size_x; x++)
    {
      // Compute the frequencies for this index
      for(unsigned int n = 0; n < size_y; n++)
      {
	// Note that the negative sign goes away for the term
	float term = 2 * PI * x * n / size_x;
	fft_real[n] = cos(term);
	fft_imag[n] = sin(term);
      }

      // Compute the value for this index
      realOutBuffer[x] = 0.0f;
      imagOutBuffer[x] = 0.0f;
      for(unsigned int n = 0; n < size_x; n++)
      {
	realOutBuffer[x] += (real_image[n*size_x + y] * fft_real[n]) - (imag_image[n*size_x + y] * fft_imag[n]);
	imagOutBuffer[x] += (imag_image[n*size_x + y] * fft_real[n]) + (real_image[n*size_x + y] * fft_imag[n]);
      }

      // Incorporate the scaling factor here
      realOutBuffer[x] /= size_x;
      imagOutBuffer[x] /= size_x;
    }
    // Write the buffer back to were the original values were
    for(unsigned int x = 0; x < size_x; x++)
    {
      real_image[x*size_x + y] = realOutBuffer[x];
      imag_image[x*size_x + y] = imagOutBuffer[x];
    }
  }
  // Reclaim some memory
  delete [] realOutBuffer;
  delete [] imagOutBuffer;
  delete [] fft_real;
  delete [] fft_imag;
}

void cpu_filter(float *real_image, float *imag_image, int size_x, int size_y)
{
  int eightX = size_x/8;
  int eight7X = size_x - eightX;
  int eightY = size_y/8;
  int eight7Y = size_y - eightY;
  for(unsigned int x = 0; x < size_x; x++)
  {
    for(unsigned int y = 0; y < size_y; y++)
    {
      if(!(x < eightX && y < eightY) &&
	 !(x < eightX && y >= eight7Y) &&
	 !(x >= eight7X && y < eightY) &&
	 !(x >= eight7X && y >= eight7Y))
      {
	// Zero out these values
	real_image[y*size_x + x] = 0;
	imag_image[y*size_x + x] = 0;
      }
    }
  }
}

float imageCleaner(float *real_image, float *imag_image, int size_x, int size_y)
{
  // These are used for timing
  struct timeval tv1, tv2;
  struct timezone tz1, tz2;

  // Start timing
  gettimeofday(&tv1,&tz1);

  // Perform fft with respect to the x direction
  cpu_fftx(real_image, imag_image, size_x, size_y, 1);
  // Perform fft with respect to the y direction
  cpu_ffty(real_image, imag_image, size_x, size_y);

  // Filter the transformed image
  cpu_filter(real_image, imag_image, size_x, size_y);

  // Perform an inverse fft with respect to the x direction
  cpu_fftx(real_image, imag_image, size_x, size_y, size_y);
  // Perform an inverse fft with respect to the y direction
  cpu_iffty(real_image, imag_image, size_x, size_y);

  // End timing
  gettimeofday(&tv2,&tz2);

  // Compute the time difference in micro-seconds
  float execution = ((tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec));
  // Convert to milli-seconds
  execution /= 1000;
  // Print some output
  printf("OPTIMIZED IMPLEMENTATION STATISTICS:\n");
  printf("  Optimized Kernel Execution Time: %f ms\n\n", execution);
  return execution;
}
