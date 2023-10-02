#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * Cross approximation with full pivoting
 */

// Fortran housekeeping
int ione = 1;
char cN = 'N';
char cT = 'T';
char cR = 'R';
char cU = 'U';
double done = 1.0;
double dzero = 0.0;

// lapack functions
extern void dcopy_(int*, double*,int*,double*,int*);
extern void dscal_(int*,double*,double*,int*);
extern void dger_(int*,int*,double*,double*,int*,double*,int*,double*,int*);
extern void dgeqrf_(int*,int*,double*,int*,double*,double*,int*,int*);
extern void dtrmm_(char*,char*,char*,char*,int*,int*,double*,double*,int*,double*,int*);
extern void dorgqr_(int*,int*,int*,double*,int*,double*,double*,int*,int*);
// extern void icopy_(int*,int*,int*,int*,int*);

//! \brief clone of lapack icopy 
// (somehow not in my lapack library)
void icopy_(int* N,int* SX,int* INCX,int* SY,int* INCY){
  int i, ix, iy, m, mp1;

  if (*N < 0)
    return;
  if (*INCX == 1 && *INCY == 1)
  {
    m = *N % 7;
    if (m != 0){
      for (i=0;i<m;++i)
        SY[i] = SX[i];
      if (*N < 7)
        return;
    }
    mp1 = m + 1;
    for (i=mp1;i<*N;i+=7)
    {
      SY[i] = SX[i];
      SY[i+1] = SX[i+1];
      SY[i+2] = SX[i+2];
      SY[i+3] = SX[i+3];
      SY[i+4] = SX[i+4];
      SY[i+5] = SX[i+5];
      SY[i+6] = SX[i+6];
    }
  }
  else{
    ix=0; iy=0;
    if (*INCX < 0)
      ix = (-*N+1) * *INCX;
    if (*INCY < 0)
      iy = (-*N+1) * *INCY;
    for (i =0; i<*N;++i)
    {
      SY[iy] = SX[ix];
      ix = ix + *INCX;
      iy = iy + *INCY;
    }
  }
  return;
}

/*! \brief Clone of print_matrix_colmajor from lapacke.
 *
 * Print the n x m matrix mat.
*/
void print_matrix_colmajor(char* desc, int n, int m, double* mat)
{
  int i,j;
  printf("\n %s [%i x %i]\n", desc, n, m);
  for(i=0;i<n;++i)
  {
    for(j=0;j<m;++j)
      printf(" %6.3f", mat[i+j*n]);
    printf("\n");
  }
}


/*! \brief Compute localcross (double version)
 *
 * @param Y input of size nxm
 * @param n number of rows of Y
 * @param m number of columns of Y
 * @param tol tolerance
 * @param U will point to array containing U of size nxr
 * @param V will point to array containing V of size rxm
 * @param I will point to array containing indices of size r
 * @param rank rank of the decomposition
*/
void localcross_f64(double* Y, int n, int m, double tol, double** U, double** V, double** I, int* rank, char order)
{
  double *u, *vt, *res, *ind, *work, val_max, val;
  int minsz, sz, r, piv1, piv2, i;

  minsz = (m<n) ? m : n;

  u = (double *)malloc(sizeof(double)*n*minsz);
  vt = (double *)malloc(sizeof(double)*minsz*m);
  res = (double *)malloc(sizeof(double)*n*m);
  ind = (double *)malloc(sizeof(double)*minsz);

  sz = n*m;
  switch (order)
  {
    case 'F': dcopy_(&sz, Y, &ione, res, &ione); break;
    case 'C': for(i=0;i<m;++i) dcopy_(&n, &Y[i], &m, &res[i*n], &ione); break;
    default: printf("order must be \'C\' or \'F\'\n");
  }

  // find max element value of Y
  val_max = 0.0;
  for (i=0; i<n*m; i++)
    if (fabs(Y[i])>val_max)
      val_max = fabs(Y[i]);

  // Main loop 
  for (r=0; r<minsz; r++) {
    // Find the maximal element of res
    val=0.0;
    for (i=0; i<n*m; i++)
      if (fabs(res[i])>val){
        piv2 = i;
        val = fabs(res[piv2]);
      }

    if (val<=tol*val_max) break;

    // compute index
    piv1 = piv2 % n;
    piv2 = piv2 / n;
    // update index array
    ind[r] = (double) piv1;
    // update u and v
    dcopy_(&n, &res[piv2*n], &ione, &u[r*n], &ione);
    dcopy_(&m, &res[piv1], &n, &vt[r*m], &ione);
    // scale column of vt by pivot 
    val = 1.0 / res[piv1+piv2*n];
    dscal_(&m,&val,&vt[r*m],&ione);
    // compute res = res - u(:,r)*v(r,:)
    val = -1.0;
    dger_(&n, &m, &val, &u[r*n], &ione, &vt[r*m],&ione, res, &n);
  }
  if (r==0) {
    // There was a zero matrix
    r=1;
    memset(u, 0, sizeof(double)*n);
    memset(vt, 0, sizeof(double)*m);
    ind[0] = 1.;
  }
  *rank = r;  

  // QR u
  sz = -1;
  dgeqrf_(&n, &r, u, &n, &res[1], &res[0], &sz, &i);
  sz = (int)res[0];
  work = (double *)malloc(sizeof(double)*sz);
  dgeqrf_(&n, &r, u, &n, res, work, &sz, &i);
  dtrmm_(&cR, &cU, &cT, &cN, &m, &r, &done, u, &n, vt, &m);
  dorgqr_(&n, &r, &r, u, &n, res, work, &sz, &i);
  
  free(work);
  free(res);

  // Return outputs

  sz = n*r;
  *U = (double *)malloc(sizeof(double)*sz);
  // output in C order
  for(i=0; i<r; ++i) dcopy_(&n, &u[i*n], &ione, &(*U)[i], &r);
  // dcopy_(&sz, u, &ione, *U, &ione);

  sz = m*r;
  *V = (double *)malloc(sizeof(double)*sz);
  // vt should be transposed
  // but output in C order
  dcopy_(&sz, vt, &ione, *V, &ione);
  // for (i=0; i<r; i++) dcopy_(&m, &vt[i*m], &ione, &(*V)[i], &r);
    
  *I = (double *)malloc(sizeof(double)*r);
  dcopy_(&r, ind, &ione, *I, &ione);

  free(u);
  free(vt);
  free(ind);
}