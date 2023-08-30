#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // Incluimos la biblioteca OpenMP
#include <time.h> // Incluimos la biblioteca para medir el tiempo

void Ax_b(int m, int n, double* A, double* x, double* b, int block_size);

int main(int argc, char *argv[]) {
  double *A, *x, *b;
  int i, j, m, n;
  int block_size;

  printf("Ingrese las dimensiones m y n de la matriz: ");
  scanf("%d %d", &m, &n);

  printf("Ingrese el tamaño de bloque: ");
  scanf("%d", &block_size);

  //---- Asignación de memoria para la matriz A ----
  if ((A = (double *)malloc(m * n * sizeof(double))) == NULL)
    perror("memory allocation for A");

  //---- Asignación de memoria para el vector x ----
  if ((x = (double *)malloc(n * sizeof(double))) == NULL)
    perror("memory allocation for x");

  //---- Asignación de memoria para el vector b ----
  if ((b = (double *)malloc(m * sizeof(double))) == NULL)
    perror("memory allocation for b");

  printf("Initializing matrix A and vector x\n");

  //---- Inicialización con elementos aleatorios entre 1-7 y 1-13
  for (j = 0; j < n; j++)
    x[j] = rand() % 7 + 1;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i * n + j] = rand() % 13 + 1;

  clock_t start_time = clock(); // Medimos el tiempo antes de la llamada a la función
  //#pragma omp parallel for schedule(static, block_size) --STATIC
  //#pragma omp parallel for schedule(dynamic, block_size) --DYNAMIC
  #pragma omp parallel for shared(m,n,A,x,b) private(i,j) schedule(guided, block_size) //GUIDED
  for (i = 0; i < m; i++) {
    b[i] = 0.0;
    for (j = 0; j < n; j++) {
      b[i] += A[i * n + j] * x[j];
    }
  }
  clock_t end_time = clock(); // Medimos el tiempo después de la llamada a la función


  printf("\nb: \n");
  for (j = 0; j < n; j++)
    printf("\t%0.0f ", b[j]);
  printf("\n\n");

  free(A);
  free(x);
  free(b);

  printf("Time taken: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

  return 0;
}

/* ------------------------
 * Ax_b
 * ------------------------
 */
void Ax_b(int m, int n, double *A, double *x, double *b, int block_size) {
  int i, j;
  
//#pragma omp parallel for shared(m,n,A,x,b) private(i,j) schedule(static, block_size) --STATIC
//#pragma omp parallel for shared(m,n,A,x,b) private(i,j) schedule(dynamic, block_size) --DTNAMIC
#pragma omp parallel for shared(m,n,A,x,b) private(i,j) schedule(guided, block_size) //GUIDED
 for (i = 0; i < m; i++) {
    b[i] = 0.0; // inicialización elemento i del vector
    for (j = 0; j < n; j++) {
      b[i] += A[i * n + j] * x[j]; // producto punto
    }
  } /*−−-Fin de parallel for−−−*/
}
