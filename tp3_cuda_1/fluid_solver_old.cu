#include "fluid_solver.h"
#include <cmath>
#include <omp.h>
#include <stdio.h>

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

void add_source(int M, int N, int O, float *x, float *s, float dt)
{
  int size = (M + 2) * (N + 2) * (O + 2);
  float *d_x, *d_s;

  // Define the number of threads per block and the number of blocks
  int threads_per_block = 256;
  int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

  cudaMalloc((void **)&d_x, size * sizeof(float));
  cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void **)&d_s, size * sizeof(float));
  cudaMemcpy(d_s, s, size * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel
  add_source_kernel<<<blocks_per_grid, threads_per_block>>>(size, d_x, d_s, dt);

  // Synchronize to ensure the kernel has completed before returning
  cudaDeviceSynchronize();

  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error in set_bnd: %s\n", cudaGetErrorString(err));
  }

  // Copy the result back to host memory
  cudaMemcpy(x, d_x, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(s, d_s, size * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_x);
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
  // Calculate the size of the grid
  int size = (M + 2) * (N + 2) * (O + 2);
  float *d_x;

  cudaMalloc((void **)&d_x, size * sizeof(float));
  cudaMemcpy(d_x, x, size * sizeof(float), cudaMemcpyHostToDevice);

  // Define thread and block dimensions
  int threadsPerBlock = 256;
  int blocksPerGrid = ((M + 2) * (N + 2) + threadsPerBlock - 1) / threadsPerBlock;

  // Launch the CUDA kernel
  set_bnd_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, b, d_x);

  // Synchronize the device to ensure all operations are completed
  cudaDeviceSynchronize();

  // Check for kernel errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error in set_bnd: %s\n", cudaGetErrorString(err));
  }

  // Copy the result back to host memory
  cudaMemcpy(x, d_x, size * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_x);
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

void lin_solve_cuda(int M, int N, int O, int b, float *h_x, float *h_x0, float a, float c)
{
  size_t size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

  // Allocate device arrays
  float *d_x, *d_x0, *d_max_changes;
  cudaMalloc(&d_x, size);
  cudaMalloc(&d_x0, size);
  cudaMalloc(&d_max_changes, M * N * O * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x0, h_x0, size, cudaMemcpyHostToDevice);

  // Initialize device memory for max_changes
  cudaMemset(d_max_changes, 0.0, M * N * O * sizeof(float));

  // Set grid and block dimensions
  dim3 threadsPerBlock(16, 16, 4);
  dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

  // Convergence parameters
  const float tol = 1e-7;        // Convergence tolerance
  const int max_iterations = 20; // Maximum allowed iterations
  int l = 0;                     // Iteration counter
  float max_c = 0.0f;
  float *max_changes = new float[M * N * O]; // Temporary array to store changes

  // Iterative loop
  while (l < max_iterations)
  {
    max_c = 0.0f;

    // Launch the red dot kernel
    red_dot_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, d_x, d_x0, a, c, d_max_changes);
    cudaDeviceSynchronize();

    // Launch the black dot kernel
    black_dot_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, d_x, d_x0, a, c, d_max_changes);
    cudaDeviceSynchronize();

    // Copy max_changes back to host and reduce
    cudaMemcpy(max_changes, d_max_changes, M * N * O * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N * O; i++)
    {
      max_c = fmaxf(max_c, max_changes[i]);
    }

    // Update boundary conditions
    cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
    set_bnd(M, N, O, b, h_x);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    // Convergence check
    if (max_c <= tol)
      break;

    l++;
  }

  // Copy final result back to host
  cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_x0, d_x0, size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_x);
  cudaFree(d_x0);
  cudaFree(d_max_changes);
  delete[] max_changes;
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c)
{

  float old_x, change;
  const float tol = 1e-7;        // Convergence tolerance
  const int max_iterations = 20; // Maximum allowed iterations
  int l = 0;                     // Iteration counter
  float max_c = 0.0f;            // Maximum change for convergence
  float inv_c = 1.0f / c;

  while (l < max_iterations)
  {
    max_c = 0.0f;

    // Red points update
    for (int i = 1; i <= M; i++)
    {
      for (int j = 1; j <= N; j++)
      {
        for (int k = 1 + (i + j) % 2; k <= O; k += 2)
        {
          old_x = x[IX(i, j, k)];
          x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                            a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                 x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                 x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) *
                           inv_c;
          change = fabs(x[IX(i, j, k)] - old_x);
          max_c = MAX(max_c, change);
        }
      }
    }

    // Ensure all threads complete red points update

    // Black points update
    for (int i = 1; i <= M; i++)
    {
      for (int j = 1; j <= N; j++)
      {
        for (int k = 1 + (i + j + 1) % 2; k <= O; k += 2)
        {
          old_x = x[IX(i, j, k)];
          x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                            a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                 x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                 x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) *
                           inv_c;
          change = fabs(x[IX(i, j, k)] - old_x);
          max_c = MAX(max_c, change);
        }
      }
    }

    // Ensure all threads complete black points update

    // Update boundary conditions
    set_bnd(M, N, O, b, x);

    // Convergence check
    if (max_c <= tol)
      break;

    l++;
  }
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
void advect(int M, int N, int O, int b, float *h_d, float *h_d0, float *h_u, float *h_v, float *h_w, float dt)
{
  size_t size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

  // Allocate device memory
  float *d_d, *d_d0, *d_u, *d_v, *d_w;
  cudaMalloc(&d_d, size);
  cudaMalloc(&d_d0, size);
  cudaMalloc(&d_u, size);
  cudaMalloc(&d_v, size);
  cudaMalloc(&d_w, size);

  // Copy data from host to device
  cudaMemcpy(d_d, h_d, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_d0, h_d0, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, h_w, size, cudaMemcpyHostToDevice);

  // Define thread and block dimensions
  dim3 threadsPerBlock(8, 8, 8); // Adjust as necessary for hardware
  dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

  // Launch the kernel
  advect_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_d, d_d0, d_u, d_v, d_w, dt);

  // Synchronize device
  cudaDeviceSynchronize();

  // Copy results back to host
  cudaMemcpy(h_d, d_d, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_d0, d_d0, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_v, d_v, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_w, d_w, size, cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_d);
  cudaFree(d_d0);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);

  // Apply boundary conditions on host array
  set_bnd(M, N, O, b, h_d);
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

void project(int M, int N, int O, float *h_u, float *h_v, float *h_w, float *h_p, float *h_div)
{
  size_t size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

  // Allocate device memory
  float *d_u, *d_v, *d_w, *d_p, *d_div;
  cudaMalloc(&d_u, size);
  cudaMalloc(&d_v, size);
  cudaMalloc(&d_w, size);
  cudaMalloc(&d_p, size);
  cudaMalloc(&d_div, size);

  // Copy data from host to device
  cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, h_w, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_p, h_p, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_div, h_div, size, cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  dim3 threadsPerBlock(16, 16, 4); // Adjust based on hardware capabilities
  dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

  // Step 1: Compute divergence and reset pressure
  compute_div_and_reset_p_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, d_u, d_v, d_w, d_div, d_p);
  cudaDeviceSynchronize();

  // Copy results back to host for debugging
  cudaMemcpy(h_div, d_div, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);

  // Step 2: Apply boundary conditions on div and p
  set_bnd(M, N, O, 0, h_div);
  set_bnd(M, N, O, 0, h_p);

  // Step 3: Solve the linear system for pressure using CUDA lin_solve
  lin_solve(M, N, O, 0, h_p, h_div, 1, 6);

  cudaMemcpy(d_p, h_p, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_div, h_div, size, cudaMemcpyHostToDevice);

  // Step 4: Update velocity based on pressure gradient
  update_velocity_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, d_u, d_v, d_w, d_p);
  cudaDeviceSynchronize();

  // Copy updated velocity fields back to host
  cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_v, d_v, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_w, d_w, size, cudaMemcpyDeviceToHost);

  // Step 5: Apply boundary conditions on velocity fields
  set_bnd(M, N, O, 1, h_u);
  set_bnd(M, N, O, 2, h_v);
  set_bnd(M, N, O, 3, h_w);

  // Free device memory
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_p);
  cudaFree(d_div);
}

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt)
{
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt)
{
  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);
  SWAP(u0, u);
  diffuse(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse(M, N, O, 3, w, w0, visc, dt);
  project(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);
}