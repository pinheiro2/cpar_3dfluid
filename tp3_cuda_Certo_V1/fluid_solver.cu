#include "fluid_solver.h"
#include <cmath>
#include <stdio.h>
#include <cuda_runtime.h>

#define IX(i, j, k) ((k) + (M + 2) * (j) + (M + 2) * (N + 2) * (i))
#define SWAP(x0, x)  \
  {                  \
    float *tmp = x0; \
    x0 = x;          \
    x = tmp;         \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define LINEARSOLVERTIMES 20

__global__ void add_source_kernel(int size, float *x, float *s, float dt)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // Compute the global thread ID
  if (idx < size)                                  // Ensure threads do not access out-of-bounds memory
  {
    x[idx] += dt * s[idx];
  }
}

void add_source(int M, int N, int O, float *d_x, float *d_s, float dt)
{
  int size = (M + 2) * (N + 2) * (O + 2);

  // Define the number of threads per block and the number of blocks
  int threads_per_block = 256;
  int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  // Launch the kernel
  add_source_kernel<<<blocks_per_grid, threads_per_block>>>(size, d_x, d_s, dt);

  // Synchronize to ensure the kernel has completed before returning
  cudaDeviceSynchronize();
}
__global__ void set_bnd_kernel(int M, int N, int O, int b, float *x)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int size = (M + 2) * (N + 2);

  if (idx < size)
  {
    int i = idx / (N + 2);
    int j = idx % (N + 2);

    if (i >= 1 && i <= M && j >= 1 && j <= N)
    {
      x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
      x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
    }
  }
}

void set_bnd(int M, int N, int O, int b, float *x)
{
  int threadsPerBlock = 256;
  int blocksPerGrid = ((M + 2) * (N + 2) + threadsPerBlock - 1) / threadsPerBlock;

  set_bnd_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, b, x);

  cudaDeviceSynchronize();
}

__global__ void red_dot_kernel(int M, int N, int O, float *x, float *x0, float a, float c, float *max_changes)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i > 0 && i <= M && j > 0 && j <= N && k > 0 && k <= O)
  {
    if ((i + j) % 2 == 0) // Red point check
    {
      float old_x = x[IX(i, j, k)];
      x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                        a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
                       c;

      float change = fabs(x[IX(i, j, k)] - old_x);
      int idx = IX(i, j, k);

      max_changes[idx] = change; // Write the change to the array
    }
  }
}
__global__ void black_dot_kernel(int M, int N, int O, float *x, float *x0, float a, float c, float *max_changes)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i > 0 && i <= M && j > 0 && j <= N && k > 0 && k <= O)
  {
    if ((i + j + 1) % 2 == 0) // Black point check
    {
      float old_x = x[IX(i, j, k)];
      x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                        a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
                       c;

      float change = fabs(x[IX(i, j, k)] - old_x);

      int idx = IX(i, j, k);

      max_changes[idx] = change; // Write the change to the array
    }
  }
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c)
{

  const float tol = 1e-7;        // Convergence tolerance
  const int max_iterations = 20; // Maximum allowed iterations
  int l = 0;                     // Iteration counter

  float *d_max_changes;
  float *max_changes = new float[M * N * O]; // Temporary array to store changes

  // Allocate memory on the device
  cudaMalloc((void **)&d_max_changes, sizeof(float) * M * N * O);

  // Initialize device memory to zero
  cudaMemset(d_max_changes, 0.0, sizeof(float) * M * N * O);

  // Set smaller grid and block dimensions for debugging purposes
  dim3 threadsPerBlock(16, 16, 4);
  dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

  // Loop for the iterations

  while (l < max_iterations)
  {
    float max_c = 0.0f;

    // Launch the red dot kernel
    red_dot_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, x, x0, a, c, d_max_changes);
    cudaDeviceSynchronize();

    // Copy the max_changes array back from device to host
    cudaMemcpy(max_changes, d_max_changes, sizeof(float) * M * N * O, cudaMemcpyDeviceToHost);

    // Perform reduction on the host to find the maximum change
    for (int i = 0; i < M * N * O; i++)
    {
      // printf("change: %f", max_changes[i]);
      max_c = fmaxf(max_c, max_changes[i]);
    }

    // Launch the black dot kernel
    black_dot_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, x, x0, a, c, d_max_changes);
    cudaDeviceSynchronize();

    // Copy the max_changes array back from device to host
    cudaMemcpy(max_changes, d_max_changes, sizeof(float) * M * N * O, cudaMemcpyDeviceToHost);

    // Perform reduction on the host to find the maximum change
    for (int i = 0; i < M * N * O; i++)
    {
      // printf("change: %f", max_changes[i]);
      max_c = fmaxf(max_c, max_changes[i]);
    }

    // Update boundary conditions
    set_bnd(M, N, O, b, x);

    // Convergence check
    if (max_c <= tol)
      break;

    l++;
  }

  // Free memory on the device
  cudaFree(d_max_changes);
  delete[] max_changes;
}

void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt)
{
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;

  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}
__global__ void advect_kernel(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // Grid and block indexing
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O) // Ensure within bounds
  {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    float x = i - dtX * u[IX(i, j, k)];
    float y = j - dtY * v[IX(i, j, k)];
    float z = k - dtZ * w[IX(i, j, k)];

    x = fmaxf(0.5f, fminf(x, M + 0.5f));
    y = fmaxf(0.5f, fminf(y, N + 0.5f));
    z = fmaxf(0.5f, fminf(z, O + 0.5f));

    int i0 = (int)x, i1 = i0 + 1;
    int j0 = (int)y, j1 = j0 + 1;
    int k0 = (int)z, k1 = k0 + 1;

    float s1 = x - i0, s0 = 1 - s1;
    float t1 = y - j0, t0 = 1 - t1;
    float u1 = z - k0, u0 = 1 - u1;

    d[IX(i, j, k)] =
        s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) +
              t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
        s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) +
              t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
  }
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt)
{
  // Define thread and block dimensions
  dim3 threadsPerBlock(8, 8, 8); // Adjust as necessary for hardware
  dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

  // Launch the kernel
  advect_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d, d0, u, v, w, dt);

  // Synchronize device
  cudaDeviceSynchronize();

  // Apply boundary conditions
  set_bnd(M, N, O, b, d);
}

__global__ void compute_div_and_reset_p_kernel(int M, int N, int O, float *u, float *v, float *w, float *div, float *p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O)
  {
    int idx = IX(i, j, k);
    div[idx] = -0.5f * (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) /
               MAX(M, MAX(N, O));
    p[idx] = 0;
  }
}

__global__ void update_velocity_kernel(int M, int N, int O, float *u, float *v, float *w, float *p)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  if (i <= M && j <= N && k <= O)
  {
    int idx = IX(i, j, k);
    u[idx] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
    v[idx] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
    w[idx] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
  }
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div)
{
  // Define grid and block dimensions
  dim3 threadsPerBlock(16, 16, 4);
  dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

  // Step 1: Compute divergence and reset pressure
  compute_div_and_reset_p_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, u, v, w, div, p);
  cudaDeviceSynchronize();

  // Step 2: Apply boundary conditions on div and p
  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);

  // Step 3: Solve the linear system for pressure using CUDA lin_solve
  lin_solve(M, N, O, 0, p, div, 1, 6);

  // Step 4: Update velocity based on pressure gradient
  update_velocity_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, u, v, w, p);
  cudaDeviceSynchronize();

  // Step 5: Apply boundary conditions on velocity fields
  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt)
{
  // Allocate CUDA memory
  float *d_x, *d_x0, *d_u, *d_v, *d_w;
  size_t size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);
  cudaMalloc(&d_x, size);
  cudaMalloc(&d_x0, size);
  cudaMalloc(&d_u, size);
  cudaMalloc(&d_v, size);
  cudaMalloc(&d_w, size);

  // Copy inputs from host to device
  cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x0, x0, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice);

  // Perform the steps using device memory
  add_source(M, N, O, d_x, d_x0, dt);
  SWAP(d_x0, d_x);
  diffuse(M, N, O, 0, d_x, d_x0, diff, dt);
  SWAP(d_x0, d_x);
  advect(M, N, O, 0, d_x, d_x0, d_u, d_v, d_w, dt);

  // Copy results back to host
  cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

  // Free CUDA memory
  cudaFree(d_x);
  cudaFree(d_x0);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
}

void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt)
{
  // Allocate CUDA memory
  float *d_u, *d_v, *d_w, *d_u0, *d_v0, *d_w0;
  size_t size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);
  cudaMalloc(&d_u, size);
  cudaMalloc(&d_v, size);
  cudaMalloc(&d_w, size);
  cudaMalloc(&d_u0, size);
  cudaMalloc(&d_v0, size);
  cudaMalloc(&d_w0, size);

  // Copy inputs from host to device
  cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u0, u0, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v0, v0, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_w0, w0, size, cudaMemcpyHostToDevice);

  // Perform the steps using device memory
  add_source(M, N, O, d_u, d_u0, dt);
  add_source(M, N, O, d_v, d_v0, dt);
  add_source(M, N, O, d_w, d_w0, dt);
  SWAP(d_u0, d_u);
  diffuse(M, N, O, 1, d_u, d_u0, visc, dt);
  SWAP(d_v0, d_v);
  diffuse(M, N, O, 2, d_v, d_v0, visc, dt);
  SWAP(d_w0, d_w);
  diffuse(M, N, O, 3, d_w, d_w0, visc, dt);
  project(M, N, O, d_u, d_v, d_w, d_u0, d_v0);
  SWAP(d_u0, d_u);
  SWAP(d_v0, d_v);
  SWAP(d_w0, d_w);
  advect(M, N, O, 1, d_u, d_u0, d_u0, d_v0, d_w0, dt);
  advect(M, N, O, 2, d_v, d_v0, d_u0, d_v0, d_w0, dt);
  advect(M, N, O, 3, d_w, d_w0, d_u0, d_v0, d_w0, dt);
  project(M, N, O, d_u, d_v, d_w, d_u0, d_v0);

  // Copy results back to host
  cudaMemcpy(u, d_u, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(v, d_v, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(w, d_w, size, cudaMemcpyDeviceToHost);

  // Free CUDA memory
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_u0);
  cudaFree(d_v0);
  cudaFree(d_w0);
}
