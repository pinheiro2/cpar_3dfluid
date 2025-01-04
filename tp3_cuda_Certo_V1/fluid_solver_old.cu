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

void add_source(int M, int N, int O, float *x, float *s, float dt)
{
  int size = (M + 2) * (N + 2) * (O + 2);
#pragma omp parallel for
  for (int i = 0; i < size; i++)
  {
    x[i] += dt * s[i];
  }
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
void set_bnd_old(int M, int N, int O, int b, float *x)
{
  int size = (M + 2) * (N + 2);
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  set_bnd_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, b, x);
  cudaDeviceSynchronize();
}

void set_bnd(int M, int N, int O, int b, float *x)
{
  // Calculate the size of the grid
  int size = (M + 2) * (N + 2) * (O + 2);
  float *d_x;

  // Allocate device memory
  cudaMalloc((void **)&d_x, size * sizeof(float));

  // Copy data from host to device
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

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt)
{
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  for (int i = 1; i <= M; i++)
  {
    for (int j = 1; j <= N; j++)
    {
      for (int k = 1; k <= O; k++)
      {
        float x = i - dtX * u[IX(i, j, k)];
        float y = j - dtY * v[IX(i, j, k)];
        float z = k - dtZ * w[IX(i, j, k)];

        x = MAX(0.5f, MIN(x, M + 0.5f));
        y = MAX(0.5f, MIN(y, N + 0.5f));
        z = MAX(0.5f, MIN(z, O + 0.5f));

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
  }
  set_bnd(M, N, O, b, d);
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p,
             float *div)
{
  for (int i = 1; i <= M; i++)
  {
    for (int j = 1; j <= N; j++)
    {
      for (int k = 1; k <= O; k++)
      {
        div[IX(i, j, k)] =
            -0.5f *
            (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
             v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) /
            MAX(M, MAX(N, O));
        p[IX(i, j, k)] = 0;
      }
    }
  }

  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  for (int i = 1; i <= M; i++)
  {
    for (int j = 1; j <= N; j++)
    {
      for (int k = 1; k <= O; k++)
      {
        u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
      }
    }
  }
  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
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

  // for (int i = 0; i < M + 2; i++)
  // {
  //   for (int j = 0; j < N + 2; j++)
  //   {
  //     for (int k = 0; k < O + 2; k++)
  //     {
  //       int idx = IX(i, j, k);
  //       printf("Initial u[%d, %d, %d] = %f\n", i, j, k, u[idx]);
  //       printf("Initial v[%d, %d, %d] = %f\n", i, j, k, v[idx]);
  //       printf("Initial w[%d, %d, %d] = %f\n", i, j, k, w[idx]);
  //     }
  //   }
  // }

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