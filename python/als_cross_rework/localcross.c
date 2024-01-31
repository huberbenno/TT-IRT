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
 * @param Y input of size n*m (will be overwritten)
 * @param n number of rows of Y
 * @param m number of columns of Y
 * @param tol tolerance
 * @param u pointer to array u in F order
 * @param vt pointer to transposed array vt in F order
 * @param I pointer to index array
 * @param rank rank of the decomposition
*/
void localcross_f64(double* Y, int n, int m, double tol, double* u, double* vt, double* I, int* rank)
{
  double *work, val_max, val;
  int minsz, sz, r, piv1, piv2, i;

  minsz = (m<n) ? m : n;

  sz = n*m;

  // find max element value of Y
  val_max = 0.0;
  for (i=0; i<sz; i++)
    if (fabs(Y[i])>val_max)
      val_max = fabs(Y[i]);

  // Main loop 
  for (r=0; r<minsz; r++) {
    // Find the maximal element of Y
    val=0.0;
    for (i=0; i<sz; i++)
      if (fabs(Y[i])>val){
        piv2 = i;
        val = fabs(Y[piv2]);
      }

    if (val<=tol*val_max) break;

    // compute index
    piv1 = piv2 % n;
    piv2 = piv2 / n;
    // update index array
    I[r] = (double) piv1;
    // update u and v
    dcopy_(&n, &Y[piv2*n], &ione, &u[r*n], &ione);
    dcopy_(&m, &Y[piv1], &n, &vt[r*m], &ione);
    // scale column of vt by pivot 
    val = 1.0 / Y[piv1+piv2*n];
    dscal_(&m,&val,&vt[r*m],&ione);
    // compute Y = Y - u(:,r)*v(r,:)
    val = -1.0;
    dger_(&n, &m, &val, &u[r*n], &ione, &vt[r*m],&ione, Y, &n);
  }
  if (r==0) {
    // There was a zero matrix
    r=1;
    memset(u, 0, sizeof(double)*n);
    memset(vt, 0, sizeof(double)*m);
    I[0] = 1.;
  }
  else {
    // QR u
    sz = -1;
    dgeqrf_(&n, &r, u, &n, &Y[1], &Y[0], &sz, &i);
    sz = (int)Y[0];
    work = (double *)malloc(sizeof(double)*sz);
    dgeqrf_(&n, &r, u, &n, Y, work, &sz, &i);
    dtrmm_(&cR, &cU, &cT, &cN, &m, &r, &done, u, &n, vt, &m);
    dorgqr_(&n, &r, &r, u, &n, Y, work, &sz, &i);
    free(work);
  }

  *rank = r;

  // Return outputs

  // sz = n*r;
  // *U = (double *)malloc(sizeof(double)*sz);
  // output in C order
  // for(i=0; i<r; ++i) dcopy_(&n, &u[i*n], &ione, &(*U)[i], &r);
  // dcopy_(&sz, u, &ione, *U, &ione);

  // sz = m*r;
  // *V = (double *)malloc(sizeof(double)*sz);
  // vt should be transposed
  // but output in C order
  // dcopy_(&sz, vt, &ione, *V, &ione);
  // for (i=0; i<r; i++) dcopy_(&m, &vt[i*m], &ione, &(*V)[i], &r);
    
  // *I = (double *)malloc(sizeof(double)*r);
  // dcopy_(&r, ind, &ione, *I, &ione);

  // free(u);
  // free(vt);
  // free(ind);
}