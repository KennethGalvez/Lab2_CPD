//--------------------------------------------------------------
// prodAx_.c
//--------------------------------------------------------------
// Calcula el producto de una matriz m x n por un vector tamaño n
// y mide el tiempo de ejecución.
//--------------------------------------------------------------
// Auth.  JJCelada - Universidad del Valle de Guatemala
// Date   2021-10-06
// Vers.  1.0


#include <stdio.h>
#include <stdlib.h>
#include <time.h> // Incluimos la biblioteca para medir el tiempo

void prodAx(int m, int n, double * restrict A, double * restrict x,
  double * restrict b);

int main(int argc, char *argv[]) {
  double *A,*x,*b;
  int i, j, m, n;

  printf("Ingrese las dimensiones m y n de la matriz: ");
  scanf("%d %d",&m,&n);

  //---- Asignación de memoria para la matriz A ----
  if ( (A=(double *)malloc(m*n*sizeof(double))) == NULL )
    perror("memory allocation for A");

  //---- Asignación de memoria para el vector x ----
  if ( (x=(double *)malloc(n*sizeof(double))) == NULL )
    perror("memory allocation for x");

  //---- Asignación de memoria para el vector b ----
  if ( (b=(double *)malloc(m*sizeof(double))) == NULL )
    perror("memory allocation for b");

  printf("Initializing matrix A and vector x\n");

  //---- Inicialización con elementos aleatorios entre 1-7 y 1-13
  for (j=0; j<n; j++)
    x[j] = rand()%7+1;

  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      A[i*n+j] = rand()%13+1;

  clock_t start_time = clock(); // Medimos el tiempo antes de la llamada a la función
  (void) prodAx(m, n, A, x, b);
  clock_t end_time = clock(); // Medimos el tiempo después de la llamada a la función

  printf("Time taken: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

  printf("\nb: \n");
  for(j=0; j<n; j++)
    printf("\t%0.0f ",b[j]);
  printf("\n\n");

  free(A);free(x);free(b);

  return(0);
}

/* ------------------------
 * prodAx
 * ------------------------
 */
void prodAx(int m, int n, double * restrict A, double * restrict x,
  double * restrict b) {

  int i, j;

  for(i = 0; i < m; i++) {
    b[i] = 0.0;

    for(j = 0; j < n; j++) {
      b[i] += A[i*n + j] * x[j];
    }
  }
}
