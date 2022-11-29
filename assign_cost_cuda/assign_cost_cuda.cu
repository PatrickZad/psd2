#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
static void HandleError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess)
  {
    printf("%s in %s at line %d\n", cudaGetErrorString(err),
           file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void cost(int **img_pids, int **batch_pids, float **costs, int dl_w, int bb_w, int p, int k)
{
  int x = blockIdx.x; // x for num of columns
  // int y = threadIdx.x;
  int y = blockIdx.y;
  int *img_pids_arr = img_pids[y];
  int *batch_pids_arr = batch_pids[x];
  int max_w = bb_w + dl_w;
  int *count_pids = (int *)malloc(sizeof(int) * max_w);
  int *counts = (int *)malloc(sizeof(int) * max_w);
  int len_counts = 0;
  // init
  for (int i = 0; i < max_w; i++)
  {
    count_pids[i] = -2;
    counts[i] = 0;
  }

  // printf("%d %d \n", x, y);
  for (int i = 0; i < bb_w; i++)
  {
    int pid = batch_pids_arr[i];
    if (pid > -1)
    {
      int in_flag = 0;
      for (int j = 0; j < len_counts; j++)
      {
        if (count_pids[j] == pid)
        {
          counts[j] += 1;
          in_flag = 1;
          break;
        }
      }
      if (in_flag == 0)
      {
        count_pids[len_counts] = pid;
        counts[len_counts] = 1;
        len_counts += 1;
      }
    }
  }
  for (int i = 0; i < dl_w; i++)
  {
    int pid = img_pids_arr[i];
    if (pid > -1)
    {
      int in_flag = 0;
      for (int j = 0; j < len_counts; j++)
      {
        if (count_pids[j] == pid)
        {
          counts[j] += 1;
          in_flag = 1;
          break;
        }
      }
      if (in_flag == 0)
      {
        count_pids[len_counts] = pid;
        counts[len_counts] = 1;
        len_counts += 1;
      }
    }
  }
  float cost_k, cost_p;
  if (len_counts == 0)
  {
    cost_k = (float)k * k;
  }
  else
  {
    int sum_diffs = 0;
    for (int i = 0; i < len_counts; i++)
    {
      sum_diffs += (counts[i] - k) * (counts[i] - k);
    }
    cost_k = sum_diffs / (float)len_counts;
  }
  cost_p = (p - len_counts) * (p - len_counts);
  costs[y][x] = cost_p + cost_k;
  // printf("%d %d %d %f %f\n", x, y, len_counts, cost_p, cost_k);
  /*
  if (x == 0 && y == 0)
  {
    printf("count_pids:\n");
    for (int i = 0; i < max_w; i++)
    {
      int pid = count_pids[i];
      printf("%d ", pid);
    }
    printf("\n");
    printf("counts:\n");
    for (int i = 0; i < max_w; i++)
    {
      int c = counts[i];
      printf("%d ", c);
    }
    printf("\n");

  }*/
  free(count_pids);
  free(counts);
}

void run_kernel(int **gpu_dl_ptr, int **gpu_bb_ptr, float **gpu_res, int dl_h, int bb_h, int dl_w, int bb_w, int p, int k)
{
  // dim3 dimGrid(bb_h); //(w,h)
  // dim3 dimBlk(dl_h);
  // cost<<<dimGrid, dimBlk>>>(gpu_dl_ptr, gpu_bb_ptr, gpu_res, dl_w, bb_w, p, k);
  dim3 dimGrid(bb_h, dl_h);
  cost<<<dimGrid, 1>>>(gpu_dl_ptr, gpu_bb_ptr, gpu_res, dl_w, bb_w, p, k);
  // cudaError_t error = cudaGetLastError();
  // HANDLE_ERROR()
}

void copy_mat_to_dev(pybind11::array_t<int> mat, int **gpu_fl_ptr_ptr, int ***gpu_mat_ptr_ptr, int *mat_h_ptr, int *mat_w_ptr)
{
  pybind11::buffer_info ha = mat.request();
  int h = ha.shape[0];
  int w = ha.shape[1];
  *mat_h_ptr = h;
  *mat_w_ptr = w;
  int **gpu_mat_hptr;
  // int *gpu_data;
  int *host_data;
  // malloc
  HANDLE_ERROR(cudaMalloc((void **)(gpu_mat_ptr_ptr), sizeof(int *) * h));
  HANDLE_ERROR(cudaMalloc((void **)(gpu_fl_ptr_ptr), h * w * sizeof(int)));
  int *gpu_fl_ptr = *gpu_fl_ptr_ptr;
  int **gpu_mat_ptr = *gpu_mat_ptr_ptr;
  gpu_mat_hptr = (int **)malloc(sizeof(int *) * h);
  for (int i = 0; i < h; i++)
  {
    gpu_mat_hptr[i] = gpu_fl_ptr + i * w;
  }
  // copy

  host_data = (int *)malloc(sizeof(int) * h * w);
  for (int i = 0; i < h; i++)
  {
    for (int j = 0; j < w; j++)
    {
      host_data[i * w + j] = mat.at(i, j);
    }
  }
  // data: cpu-> gpu
  HANDLE_ERROR(cudaMemcpy((void *)(gpu_fl_ptr), (void *)(host_data), sizeof(int) * w * h, cudaMemcpyHostToDevice));
  // addr ptrs: cpu->gpu
  HANDLE_ERROR(cudaMemcpy((void *)(gpu_mat_ptr), (void *)(gpu_mat_hptr), h * sizeof(int *), cudaMemcpyHostToDevice));
  free(gpu_mat_hptr);
  free(host_data);
}
pybind11::array_t<float> assign_cost_mat(pybind11::array_t<int> data_left, pybind11::array_t<int> batch_built, int p, int k)
{
  int dl_h, dl_w; // dl: data_left matrix
  int bb_h, bb_w; // bb: batch_built matrix
  int **gpu_dl_ptr;
  int *gpu_dl_fl_ptr;
  int **gpu_bb_ptr;
  int *gpu_bb_fl_ptr;
  float **gpu_res; // res: result matrix
  float **res;
  float *gpu_res_data;
  float *res_data; // data: flattened matrix
  copy_mat_to_dev(data_left, &gpu_dl_fl_ptr, &gpu_dl_ptr, &dl_h, &dl_w);
  copy_mat_to_dev(batch_built, &gpu_bb_fl_ptr, &gpu_bb_ptr, &bb_h, &bb_w);
  // init res mat on device
  HANDLE_ERROR(cudaMalloc((void **)(&gpu_res), sizeof(float *) * dl_h));
  HANDLE_ERROR(cudaMalloc((void **)(&gpu_res_data), dl_h * bb_h * sizeof(float)));
  res = (float **)malloc(sizeof(float *) * dl_h);
  for (int i = 0; i < dl_h; i++)
  {
    res[i] = gpu_res_data + i * bb_h;
  }
  HANDLE_ERROR(cudaMemcpy((void *)(gpu_res), (void *)(res), dl_h * sizeof(float *), cudaMemcpyHostToDevice));
  // compute cost
  run_kernel(gpu_dl_ptr, gpu_bb_ptr, gpu_res, dl_h, bb_h, dl_w, bb_w, p, k);
  // init res mat on host
  res_data = (float *)malloc(dl_h * bb_h * sizeof(float));
  HANDLE_ERROR(cudaMemcpy((void *)(res_data), (void *)(gpu_res_data), dl_h * bb_h * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < dl_h; i++)
  {
    res[i] = res_data + i * bb_h;
  }
  pybind11::array_t<float> res_arr = pybind11::array_t<float>(dl_h * bb_h);
  pybind11::buffer_info buf = res_arr.request();
  float *ptr_res = (float *)buf.ptr;
  for (int i = 0; i < dl_h; i++)
  {
    for (int j = 0; j < bb_h; j++)
    {
      // printf("%d %d %f\n", i, j, res[i][j]);
      ptr_res[i * bb_h + j] = res[i][j];
    }
  }
  res_arr.resize({dl_h, bb_h});
  // free
  // gpu_mat_free(gpu_dl_ptr);
  // gpu_mat_free(gpu_bb_ptr);
  cudaFree((void *)(gpu_dl_fl_ptr));
  cudaFree((void *)(gpu_dl_ptr));
  cudaFree((void *)(gpu_bb_fl_ptr));
  cudaFree((void *)(gpu_bb_ptr));
  cudaFree((void *)(gpu_res_data));
  cudaFree((void *)(gpu_res));
  free(res_data);
  free(res);
  return res_arr;
}

PYBIND11_MODULE(assign_cost_cuda, m)
{
  m.def("assign_cost", assign_cost_mat, pybind11::return_value_policy::copy);
}
