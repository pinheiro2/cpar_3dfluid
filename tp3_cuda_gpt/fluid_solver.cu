#include "fluid_solver.h"
#include <cmath>
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

__global__ void add_source_kernel(int M, int N, int O, float *x, const float *s, float dt)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int size = (M + 2) * (N + 2) * (O + 2);
  if (idx < size)
  {
    x[idx] += dt * s[idx];
  }
}

void add_source(int M, int N, int O, float *x, float *s, float dt)
{
  int size = (M + 2) * (N + 2) * (O + 2);
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  add_source_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, x, s, dt);
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
  int size = (M + 2) * (N + 2);
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  set_bnd_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, b, x);
  cudaDeviceSynchronize();
}

__global__ void lin_solve_kernel(int M, int N, int O, int b, float *x, const float *x0, float a, float c)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int size = (M + 2) * (N + 2) * (O + 2);

  if (idx < size)
  {
    int k = idx % (O + 2);
    int j = (idx / (O + 2)) % (N + 2);
    int i = idx / ((O + 2) * (N + 2));

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O)
    {
      x[IX(i, j, k)] = (x0[IX(i, j, k)] + a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                               x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                               x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /
                       c;
    }
  }
}

void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c)
{
  int size = (M + 2) * (N + 2) * (O + 2);
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  for (int l = 0; l < 20; l++)
  {
    lin_solve_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, b, x, x0, a, c);
    cudaDeviceSynchronize();
    set_bnd(M, N, O, b, x);
  }
}

__global__ void advect_kernel(int M, int N, int O, int b, float *d, const float *d0, const float *u, const float *v, const float *w, float dt, float dtX, float dtY, float dtZ)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int size = (M + 2) * (N + 2) * (O + 2);

  if (idx < size)
  {
    int k = idx % (O + 2);
    int j = (idx / (O + 2)) % (N + 2);
    int i = idx / ((O + 2) * (N + 2));

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O)
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

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt)
{
  float dtX = dt * M;
  float dtY = dt * N;
  float dtZ = dt * O;

  int size = (M + 2) * (N + 2) * (O + 2);
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  advect_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, b, d, d0, u, v, w, dt, dtX, dtY, dtZ);
  cudaDeviceSynchronize();
  set_bnd(M, N, O, b, d);
}

__global__ void project_div_kernel(int M, int N, int O, float *div, float *u, float *v, float *w)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int size = (M + 2) * (N + 2) * (O + 2);

  if (idx < size)
  {
    int k = idx % (O + 2);
    int j = (idx / (O + 2)) % (N + 2);
    int i = idx / ((O + 2) * (N + 2));

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O)
    {
      div[IX(i, j, k)] =
          -0.5f *
          (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] + v[IX(i, j + 1, k)] -
           v[IX(i, j - 1, k)] + w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) /
          MAX(M, MAX(N, O));
    }
  }
}

__global__ void project_velocity_kernel(int M, int N, int O, float *u, float *v, float *w, float *p)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int size = (M + 2) * (N + 2) * (O + 2);

  if (idx < size)
  {
    int k = idx % (O + 2);
    int j = (idx / (O + 2)) % (N + 2);
    int i = idx / ((O + 2) * (N + 2));

    if (i >= 1 && i <= M && j >= 1 && j <= N && k >= 1 && k <= O)
    {
      u[IX(i, j, k)] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
      v[IX(i, j, k)] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
      w[IX(i, j, k)] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    }
  }
}

void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div)
{
  int size = (M + 2) * (N + 2) * (O + 2);
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  project_div_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, div, u, v, w);
  cudaDeviceSynchronize();
  set_bnd(M, N, O, 0, div);
  set_bnd(M, N, O, 0, p);
  lin_solve(M, N, O, 0, p, div, 1, 6);

  project_velocity_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, O, u, v, w, p);
  cudaDeviceSynchronize();
  set_bnd(M, N, O, 1, u);
  set_bnd(M, N, O, 2, v);
  set_bnd(M, N, O, 3, w);
}

void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v, float *w, float diff, float dt)
{
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0, float *v0, float *w0, float visc, float dt)
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
