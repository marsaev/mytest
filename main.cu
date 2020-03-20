#include <stdio.h>
#include <cuda.h>
#include "cuComplex.h"

#define WARP_SIZE 32

// Kernel that executes on the CUDA device
template <int N>
__global__ void test(float *in, float *mul, float *out)
{
  int lane_id = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;

  __shared__ float smem [WARP_SIZE*2];
  volatile float* my_smem = &smem[WARP_SIZE*warp_id];
  __shared__ float smul[N];

  my_smem[lane_id] = in[lane_id];

  if (lane_id < N)
	smul[lane_id] = mul[lane_id];
  
  my_smem[lane_id] = smul[lane_id%N] * my_smem[lane_id];
  out[lane_id] = my_smem[lane_id];
}



// main routine that executes on the host
int main(void)
{
 
  const int N = 8;  // Number of elements in arrays
  size_t elems = 256;
  size_t size = elems * sizeof(float);
  
  float* in_h = (float*)calloc(elems, sizeof(float));        // Allocate array on host
  float* mul_h = (float*)calloc(N, sizeof(float));
  float* out_h = (float*)calloc(elems, sizeof(float));        // Allocate array on host

  for (int i = 0; i < elems; i++)
  {
    in_h[i] = (float)i;
  }
  for (int i = 0; i < N; i++)
  {
    mul_h[i] = 1.f;
  }
  
  float *in_d, *out_d, *mul_d;

  cudaMalloc((void **) &in_d, size);   // Allocate array on device
  cudaMalloc((void **) &out_d, size);   // Allocate array on device
  cudaMalloc((void **) &mul_d, N*sizeof(float));   // Allocate array on device
  cudaMemcpy(in_d, in_h, size, cudaMemcpyDefault);
  cudaMemcpy(out_d, out_h, size, cudaMemcpyDefault);
  cudaMemcpy(mul_d, mul_h, N*sizeof(float), cudaMemcpyDefault);
  cudaDeviceSynchronize();

  {    
    test<N> <<< 1, WARP_SIZE >>> (in_d, mul_d, out_d);
    cudaDeviceSynchronize();
    cudaMemcpy(out_h, out_d, WARP_SIZE*sizeof(float), cudaMemcpyDefault);
    cudaDeviceSynchronize();
  
    float expected = 0.0, got = 0.0, test = 0.0;
    for (int j = 0; j < 32; j++)
    {
      expected += in_h[j]*mul_h[j%N];
      got += out_h[j];
    }
   
    // Print results
    printf("sz: %d, Expected: %f, Got: %f\n", 32, expected, got);
  }  

  // Cleanup
  free(in_h);
  free(mul_h);
  free(out_h);
  cudaFree(in_d);
  cudaFree(out_d);
  cudaFree(mul_d);

  return 0;
}
