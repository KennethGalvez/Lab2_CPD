#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void Ax_b(int m, int n, double *A, double *x, double *b, int block_size);

int main(int argc, char *argv[]) {
  double *A, *x, *b;
  int m, n, block_size;

  printf("Ingrese las dimensiones m y n de la matriz: ");
  scanf("%d %d", &m, &n);

  printf("Ingrese el tamaño de bloque: ");
  scanf("%d", &block_size);

  A = (double *)malloc(m * n * sizeof(double));
  x = (double *)malloc(n * sizeof(double));
  b = (double *)malloc(m * sizeof(double));

  for (int j = 0; j < n; j++)
    x[j] = rand() % 7 + 1;

  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      A[i * n + j] = rand() % 13 + 1;

  double start_time = omp_get_wtime();
  Ax_b(m, n, A, x, b, block_size);
  double end_time = omp_get_wtime();

  printf("\nb: \n");
  for (int i = 0; i < m; i++)
    printf("\t%.0f ", b[i]);
  printf("\n\n");

  free(A);
  free(x);
  free(b);

  printf("Time taken: %f seconds\n", end_time - start_time);

  return 0;
}

void Ax_b(int m, int n, double *A, double *x, double *b, int block_size) {
    #pragma omp parallel for shared(A, x, b) firstprivate(m, n) schedule(dynamic, block_size)
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        int row_index = i * n;
        for (int j = 0; j < n; j++) {
            sum += A[row_index + j] * x[j];
        }
        b[i] = sum;
    }
}