#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <winsock.h>
#include <time.h>

#define Grid2DBlock2D

void initialFloat(float* ip, int size) {
	for (int i = 0; i < size; i++) {
		ip[i] = i;
	}
}

void sumMatrixOnHost(float* A, float* B, float* C, const int nx, const int ny) 
{
	float* ia = A;
	float* ib = B;
	float* ic = C;
	for (int iy = 0; iy < ny; iy++) {
		for (int ix = 0; ix < nx; ix++) {
			ic[ix] = ia[ix] + ib[ix];
		}
		ia += nx;
		ib += nx;
		ic += nx;
	}
}

__global__ void sumMatrixOnGPU2D(float* MatA, float* MatB, float* MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy * nx + ix;
	if (ix < nx && iy < ny) {
		MatC[idx] = MatA[idx] + MatB[idx];
	}
}

__global__ void sumMatrixOnGPUMix(float* MatA, float* MatB, float* MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = blockDim.y;
	unsigned int idx = iy * nx + ix;
	if (ix < nx && iy < ny) {
		MatC[idx] = MatA[idx] + MatB[idx];
	}
}

int main()
{
	int nx = 1 << 14;
	int ny = 1 << 14;
	int nxy = nx * ny;
	int nBytes = nx * ny * sizeof(float);
	printf("matrix nx:%d ny:%d\n", nx, ny);
	float* h_a, * h_b, * hostRef, * gpuRef;
	h_a = new float[nxy];
	h_b = new float[nxy];
	hostRef = new float[nxy];
	gpuRef = new float[nxy];

	initialFloat(h_a, nxy);
	initialFloat(h_b, nxy);

	sumMatrixOnHost(h_a, h_b, hostRef, nx, ny);

	//malloc device global memory
	float* d_MatA;
	float* d_MatB;
	float* d_MatC;
	cudaMalloc((void**)&d_MatA, nBytes);
	cudaMalloc((void**)&d_MatB, nBytes);
	cudaMalloc((void**)&d_MatC, nBytes);

	//transfer data from host to device
	cudaMemcpy((void*)d_MatA, (void*)h_a, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_MatB, (void*)h_b, nBytes, cudaMemcpyHostToDevice);


	cudaEvent_t start, stop;
	float elapsedTime = 0.f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventSynchronize(start);
	cudaEventRecord(start, 0);
#ifdef Grid2DBlock2D
	//invoke kernel at host side
	int dimx = 32;
	int dimy = 32;
	dim3 threadsPerBlock(dimx, dimy);
	dim3 numBlocks((nx + dimx - 1) / threadsPerBlock.x, (ny + dimy - 1) / threadsPerBlock.y);
	sumMatrixOnGPU2D << < numBlocks, threadsPerBlock >> > (d_MatA, d_MatB, d_MatC, nx, ny);

#else
	int dimx = 1024;
	dim3 threadsPerBlock(dimx);
	dim3 numBlocks((nx + dimx - 1) / threadsPerBlock.x, ny);
	sumMatrixOnGPUMix<<<numBlocks, threadsPerBlock >>>(d_MatA, d_MatB, d_MatC, nx, ny);
#endif

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("sumMatrixOnGPU<<<(%d %d),(%d %d)>>> time speed:%f ms\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y,  elapsedTime);

	cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);

	delete[]h_a;
	delete[]h_b;
	delete[]hostRef;
	delete[]gpuRef;

	return 0;
}

