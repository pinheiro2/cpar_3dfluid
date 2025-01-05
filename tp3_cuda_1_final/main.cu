#include "EventManager.h"
#include "fluid_solver.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define SIZE 84

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))

// Globals for the grid size
static int M = SIZE;
static int N = SIZE;
static int O = SIZE;
static float dt = 0.1f;      // Time delta
static float diff = 0.0001f; // Diffusion constant
static float visc = 0.0001f; // Viscosity constant

// Fluid simulation arrays
static float *u, *v, *w, *u_prev, *v_prev, *w_prev;
static float *dens, *dens_prev;

// Function to allocate simulation data
int allocate_data()
{
  size_t size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

  // Allocate and copy velocity arrays
  cudaMalloc(&u, size);
  cudaMalloc(&v, size);
  cudaMalloc(&w, size);
  cudaMalloc(&u_prev, size);
  cudaMalloc(&v_prev, size);
  cudaMalloc(&w_prev, size);
  cudaMalloc(&dens, size);
  cudaMalloc(&dens_prev, size);

  if (!u || !v || !w || !u_prev || !v_prev || !w_prev || !dens || !dens_prev)
  {
    std::cerr << "Cannot allocate memory" << std::endl;
    return 0;
  }
  return 1;
}

// Function to clear the data (set all to zero)
void clear_data()
{
  int size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

  // Set all elements to zero in the device arrays
  cudaMemset(u, 0, size);
  cudaMemset(v, 0, size);
  cudaMemset(w, 0, size);
  cudaMemset(u_prev, 0, size);
  cudaMemset(v_prev, 0, size);
  cudaMemset(w_prev, 0, size);
  cudaMemset(dens, 0, size);
  cudaMemset(dens_prev, 0, size);

  cudaDeviceSynchronize();
}

// Free allocated memory
void free_data()
{
  cudaFree(u);
  cudaFree(v);
  cudaFree(w);
  cudaFree(u_prev);
  cudaFree(v_prev);
  cudaFree(w_prev);
  cudaFree(dens);
  cudaFree(dens_prev);
}

__global__ void apply_density_source_kernel(int M, int N, int O, float *dens, int i, int j, int k, float density)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) // Single thread for this operation
  {
    int idx = IX(i, j, k);
    dens[idx] = density;
  }
}

__global__ void apply_force_kernel(int M, int N, int O, float *u, float *v, float *w, int i, int j, int k, float fx, float fy, float fz)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) // Single thread for this operation
  {
    int idx = IX(i, j, k);
    u[idx] = fx;
    v[idx] = fy;
    w[idx] = fz;
  }
}

void apply_events(const std::vector<Event> &events)
{
  for (const auto &event : events)
  {
    int i = M / 2, j = N / 2, k = O / 2; // Center of the grid

    if (event.type == ADD_SOURCE)
    {
      // Apply density source at the center of the grid
      apply_density_source_kernel<<<1, 1>>>(M, N, O, dens, i, j, k, event.density);
      cudaDeviceSynchronize(); // Ensure the operation is completed
    }
    else if (event.type == APPLY_FORCE)
    {
      // Apply forces based on the event's vector (fx, fy, fz)
      apply_force_kernel<<<1, 1>>>(M, N, O, u, v, w, i, j, k, event.force.x, event.force.y, event.force.z);
      cudaDeviceSynchronize(); // Ensure the operation is completed
    }
  }
}

// Function to sum the total density
float sum_density()
{
  size_t size = (M + 2) * (N + 2) * (O + 2) * sizeof(float);

  // Allocate host memory to copy the device array
  float *h_dens = new float[(M + 2) * (N + 2) * (O + 2)];

  // Copy the density array from device to host
  cudaMemcpy(h_dens, dens, size, cudaMemcpyDeviceToHost);

  // Calculate the total density on the host
  float total_density = 0.0f;
  for (int i = 0; i < (M + 2) * (N + 2) * (O + 2); i++)
  {
    total_density += h_dens[i];
  }

  // Free the host memory
  delete[] h_dens;

  return total_density;
}

// // Function to sum the total density
// float sum_density()
// {
//   float total_density = 0.0f;
//   int size = (M + 2) * (N + 2) * (O + 2);
//   for (int i = 0; i < size; i++)
//   {
//     total_density += dens[i];
//   }
//   return total_density;
// }

// Simulation loop
void simulate(EventManager &eventManager, int timesteps)
{
  for (int t = 0; t < timesteps; t++)
  {
    // Get the events for the current timestep
    std::vector<Event> events = eventManager.get_events_at_timestamp(t);

    // Apply events to the simulation
    apply_events(events);

    // Perform the simulation steps

    vel_step(M, N, O, u, v, w, u_prev, v_prev, w_prev, visc, dt);
    dens_step(M, N, O, dens, dens_prev, u, v, w, diff, dt);
  }
}

int main()
{
  // Initialize EventManager
  EventManager eventManager;
  eventManager.read_events("events.txt");

  // Get the total number of timesteps from the event file
  int timesteps = eventManager.get_total_timesteps();

  // Allocate and clear data
  if (!allocate_data())
    return -1;
  clear_data();

  // Run simulation with events
  simulate(eventManager, timesteps);

  // Print total density at the end of simulation
  float total_density = sum_density();
  std::cout << "Total density after " << timesteps
            << " timesteps: " << total_density << std::endl;

  // Free memory
  free_data();

  return 0;
}