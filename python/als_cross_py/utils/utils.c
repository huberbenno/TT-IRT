#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>


/*
 * Accelerated routines for als_cross_parametric
 */

// Fortran housekeeping
int ione = 1;
char cN = 'N';
char cT = 'T';
double done = 1.0;
double dzero = 0.0;

// lapack functions
extern void dcopy_(int*, double*,int*,double*,int*);
extern void dgemv_(char*,int*,int*,double*,double*,int*,double*,int*,double*,double*,int*);
extern void dgesv_(int*,int*,double*,int*,int*,double*,int*,int*);
extern void dgemm_(char*,char*,int*,int*,int*,double*,double*,int*,double*,int*,double*,double*,int*);
extern void daxpy_(int*,double*,double*,int*,double*,int*);

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



/*! \brief Solve block diagonal system
 *
 * @param XAX shape (rx1,rx1,rc1)
 * @param CiX2 shape (rc1,n1,rx2)
 * @param rhs shape (rx1,n1,rx2)
 * @param sol shape (rx1,n1,rx2)
 * @param rx1
 * @param rx2
 * @param rc1
 * @param n1
*/
void solve_blockdiag(double* XAX1, double* CiXC2, double* rhs, double* sol, int rx1, int rx2, int rc1, int n1)
{
  /* Todo: Make it adaptive to complex arithmetics! */
  double *Ai, *XAX1_F, *CiXC2_F, *sol_F;
  int i,j, tmpsize, *ipiv, info;

  // get Fortran order matrixes
  tmpsize = rx1*rx1;
  XAX1_F = (double*) malloc(sizeof(double)*tmpsize*rc1);
  for (i=0;i<rx1;++i)
    for (j=0;j<rx1;++j)
      dcopy_(&rc1, &XAX1[(i*rx1+j)*rc1], &ione, &XAX1_F[i+j*rx1], &tmpsize);

  tmpsize = rc1*n1;
  CiXC2_F = (double*) malloc(sizeof(double)*tmpsize*rx2);
  for (i=0;i<rc1;++i)
    for (j=0;j<n1;++j)
      dcopy_(&rx2, &CiXC2[(i*n1+j)*rx2], &ione, &CiXC2_F[i+j*rc1], &tmpsize);

  tmpsize = rx1*n1;
  sol_F = (double*) malloc(sizeof(double)*tmpsize*rx2);
  for (i=0;i<rx1;++i)
    for (j=0;j<n1;++j)
      dcopy_(&rx2, &rhs[(i*n1+j)*rx2], &ione, &sol_F[i+j*rx1], &tmpsize);
  // dcopy_(&tmpsize, rhs, &ione, sol, &ione);

  // rxr storage
  Ai = (double*) malloc(sizeof(double)*rx1*rx1);

  // for gesv
  ipiv = (int*) malloc(sizeof(int)*rx1);

  // Main loop -- generate and solve
  tmpsize = rx1*rx1;
  for (i=0; i<n1*rx2; i++) {
    dgemv_(&cN, &tmpsize, &rc1, &done, XAX1_F, &tmpsize, &CiXC2_F[i*rc1], &ione, &dzero, Ai, &ione);
    dgesv_(&rx1, &ione, Ai, &rx1, ipiv, &sol_F[i*rx1], &rx1, &info);
  }

  // get solution in C-order
  tmpsize = rx1*n1;
  for (i=0;i<rx1;++i)
    for (j=0;j<n1;++j)
      dcopy_(&rx2, &sol_F[i+j*rx1], &tmpsize, &sol[(i*n1+j)*rx2], &ione);

  free(ipiv);
  free(Ai);
  free(XAX1_F);
  free(CiXC2_F);
  free(sol_F);
}

struct solve_block_common_args{
  int rx1, rx2, rc1, n1;
  double* const XAX1_F, *CiXC2_F, *sol_F, *sol;
};

struct solve_block_args{
  int i_start;
  int stride;
  struct solve_block_common_args* common;
};


void* solve_block(void* ptr)
{
  struct solve_block_args* args = (struct solve_block_args*) ptr;

  double* Ai;
  int *ipiv, tmpsize, info, offset, i, j, k;

  offset = args->common->rx2 * args->common->n1;
  tmpsize = args->common->rx1 * args->common->rx1;

  // rxr storage
  Ai = (double*) malloc(sizeof(double) * tmpsize);

  // for gesv
  ipiv = (int*) malloc(sizeof(int) * args->common->rx1);


  for (i=args->i_start; i<offset;i += args->stride){
    dgemv_(
      &cN,
      &tmpsize,
      &args->common->rc1,
      &done,
      args->common->XAX1_F,
      &tmpsize,
      &args->common->CiXC2_F[i * args->common->rc1],
      &ione,
      &dzero,
      Ai,
      &ione);
    dgesv_(
      &args->common->rx1,
      &ione,
      Ai,
      &args->common->rx1,
      ipiv,
      &args->common->sol_F[i * args->common->rx1],
      &args->common->rx1,
      &info);

    // copy into output array (C-order)
    j = i % args->common->n1;
    k = i / args->common->n1;
    dcopy_(
      &args->common->rx1,
      &args->common->sol_F[i * args->common->rx1],
      &ione,
      &args->common->sol[j*args->common->rx2 + k],
      &offset);
  }

  free(Ai);
  free(ipiv);

  pthread_exit(NULL);
}

/*! \brief Solve block diagonal system in parallel
 *
 * @param XAX shape (rx1,rx1,rc1)
 * @param CiX2 shape (rc1,n1,rx2)
 * @param rhs shape (rx1,n1,rx2)
 * @param sol shape (rx1,n1,rx2)
 * @param rx1
 * @param rx2
 * @param rc1
 * @param n1
 * @param nthread number of CPU threads to use
*/
void solve_blockdiag_parallel(double* XAX1, double* CiXC2, double* rhs, double* sol, int rx1, int rx2, int rc1, int n1, int nthread)
{
  /* Todo: Make it adaptive to complex arithmetics! */
  double *XAX1_F, *CiXC2_F, *sol_F;
  int i,j, tmpsize;

  // Fortran order matrixes
  tmpsize = rx1*rx1;
  XAX1_F = (double*) malloc(sizeof(double)*tmpsize*rc1);
  for (i=0;i<rx1;++i)
    for (j=0;j<rx1;++j)
      dcopy_(&rc1, &XAX1[(i*rx1+j)*rc1], &ione, &XAX1_F[i+j*rx1], &tmpsize);

  tmpsize = rc1*n1;
  CiXC2_F = (double*) malloc(sizeof(double)*tmpsize*rx2);
  for (i=0;i<rc1;++i)
    for (j=0;j<n1;++j)
      dcopy_(&rx2, &CiXC2[(i*n1+j)*rx2], &ione, &CiXC2_F[i+j*rc1], &tmpsize);

  tmpsize = rx1*n1;
  sol_F = (double*) malloc(sizeof(double)*tmpsize*rx2);
  for (i=0;i<rx1;++i)
    for (j=0;j<n1;++j)
      dcopy_(&rx2, &rhs[(i*n1+j)*rx2], &ione, &sol_F[i+j*rx1], &tmpsize);

  // Main loop is parallel
  pthread_t threads[nthread];
  struct solve_block_common_args common = {rx1, rx2, rc1, n1, XAX1_F, CiXC2_F, sol_F, sol};
  struct solve_block_args args[nthread];

  for (i=0;i<nthread;++i){
    args[i].i_start = i;
    args[i].stride = nthread;
    args[i].common = &common;
  }

  for(i=0;i<nthread;++i)
    j = pthread_create(&threads[i], NULL, solve_block, (void*) &args[i]);

  for(i=0;i<nthread;++i)
    pthread_join(threads[i], NULL);

  free(XAX1_F);
  free(CiXC2_F);
  free(sol_F);

  return;
}

/*! \brief Project diagonal system
 *
 * @param XAX shape (rx1,rx1,rc1)
 * @param Ci shape (rc1,n1,rc2)
 * @param X shape (rx1,n1,rx2)
 * @param XAX_new shape (rx2,rx2,rc2)
 * @param rx1
 * @param rx2
 * @param rc1
 * @param rc2
 * @param n1
*/
void project_blockdiag(double* XAX1, double* Ci, double* X, double* XAX1_new, int rx1, int rx2, int rc1, int rc2, int n1)
/* RHS: XAX1 [rx1',rx1,rc1], Ci [rc1,n,rc2], X [rx1,n,rx2]
   LHS: XAX1' [rx2',rx2,rc2]
*/
{
  /* Todo: Make it adaptive to complex arithmetics! */
  double *Xperm, *Cperm, *XAX1_F;
  double *dtmp1, *dtmp2;
  int i,j,k, tmpsize;

  // Clear output
  memset(XAX1_new, 0, sizeof(double)*rx2*rx2*rc2);

  // Permute X and C for easier BLAS (and transfer to Fortran order)
  Xperm = (double*) malloc(sizeof(double)*rx1*rx2*n1);
  for (i=0; i<rx1; i++)
    for (j=0; j<n1; j++)
      dcopy_(&rx2, &X[(i*n1+j)*rx2], &ione, &Xperm[i+j*rx1*rx2], &rx1);

  Cperm = (double*) malloc(sizeof(double)*rc1*rc2*n1);
  for (i=0; i<rc1; i++)
    for (j=0; j<n1; j++)
      dcopy_(&rc2, &Ci[(i*n1+j)*rc2], &ione, &Cperm[i+j*rc1*rc2], &rc1);

  // get XAX1 in Fortran order
  tmpsize = rx1*rx1;
  XAX1_F = (double*) malloc(sizeof(double)*tmpsize*rc1);
  for (i=0;i<rx1;++i)
    for (j=0;j<rx1;++j)
      dcopy_(&rc1, &XAX1[(i*rx1+j)*rc1], &ione, &XAX1_F[i+j*rx1], &tmpsize);

  // Temp storage
  // Determine the maximal size we will need
  tmpsize = rx2>rx1 ? rx2*rx2 : rx2*rx1;
  tmpsize *= rc2>rc1 ? rc2 : rc1;
  dtmp1 = (double*) malloc(sizeof(double)*tmpsize);
  dtmp2 = (double*) malloc(sizeof(double)*tmpsize);

  // Main loop -- reduce over n
  for (i=0; i<n1; i++) {
    tmpsize = rx1*rc1;
    // tmp1 = X'*XAX1
    dgemm_(&cT,&cN,&rx2,&tmpsize,&rx1,&done,&Xperm[i*rx1*rx2],&rx1, XAX1_F,&rx1,&dzero,dtmp1,&rx2);
    tmpsize = rx2*rx1;
    // tmp2 = tmp1*C
    dgemm_(&cN,&cN,&tmpsize,&rc2,&rc1, &done,dtmp1,&tmpsize, &Cperm[i*rc1*rc2],&rc1,&dzero,dtmp2,&tmpsize);
    // Permute this partial projection
    // It was rx2, rx1*rc2, should become rx1*rc2, rx2
    tmpsize = rx1*rc2;
    for (j=0; j<rx2; j++)
      dcopy_(&tmpsize, &dtmp2[j], &rx2, &dtmp1[j*tmpsize], &ione);
    // tmp2 = tmp1*X
    tmpsize = rc2*rx2;
    dgemm_(&cT,&cN,&rx2,&tmpsize,&rx1, &done,&Xperm[i*rx1*rx2],&rx1, dtmp1,&rx1,&dzero,dtmp2,&rx2);
    // Permute tmp2
    // Was rx2*rc2, rx2', needed rx2', rx2*rc2
    for (j=0; j<rx2; j++)
      dcopy_(&tmpsize, &dtmp2[j*tmpsize], &ione, &dtmp1[j], &rx2);
    // Add to XAX1_new (from F order to C order)
    tmpsize = rx2*rc2;
    for (j=0;j<rx2;++j)
      for (k=0;k<rc2;++k)
        daxpy_(&rx2, &done, &dtmp1[rx2*(j+rx2*k)], &ione, &XAX1_new[j*rc2+k], &tmpsize);
  }

  free(dtmp1);
  free(dtmp2);
  free(XAX1_F);
  free(Cperm);
  free(Xperm);
}

struct project_block_common_args{
  int rx1, rx2, rc1, rc2, n1, max_tmpsize;
  double *Xperm, *Cperm, *XAX1_F;
};

struct project_block_args{
  int i_start;
  int stride;
  double* XAX1_new_local;
  struct project_block_common_args* common;
};


void* project_block(void* ptr)
{
  struct project_block_args* args = (struct project_block_args*) ptr;

  double* dtmp1, *dtmp2;
  int tmpsize, i, j, k;

  dtmp1 = (double*) malloc(sizeof(double)*args->common->max_tmpsize);
  dtmp2 = (double*) malloc(sizeof(double)*args->common->max_tmpsize);

  for (i=args->i_start; i<args->common->n1;i += args->stride){
    tmpsize = args->common->rx1 * args->common->rc1;
    // tmp1 = X'*XAX1
    dgemm_(
      &cT, &cN,
      &args->common->rx2, &tmpsize, &args->common->rx1,
      &done,
      &args->common->Xperm[i*args->common->rx1*args->common->rx2], &args->common->rx1, 
      args->common->XAX1_F,&args->common->rx1,
      &dzero,
      dtmp1, &args->common->rx2
    );
    tmpsize = args->common->rx2 * args->common->rx1;
    // tmp2 = tmp1*C
    dgemm_(
      &cN, &cN,
      &tmpsize, &args->common->rc2, &args->common->rc1, 
      &done,
      dtmp1, &tmpsize, 
      &args->common->Cperm[i * args->common->rc1 * args->common->rc2], &args->common->rc1,
      &dzero,
      dtmp2, &tmpsize
    );
    // Permute this partial projection
    // It was rx2, rx1*rc2, should become rx1*rc2, rx2
    tmpsize = args->common->rx1 * args->common->rc2;
    for (j=0; j<args->common->rx2; j++)
      dcopy_(&tmpsize, &dtmp2[j], &args->common->rx2, &dtmp1[j*tmpsize], &ione);
    // tmp2 = tmp1*X
    tmpsize = args->common->rc2 * args->common->rx2;
    dgemm_(
      &cT,&cN,
      &args->common->rx2,&tmpsize,&args->common->rx1, 
      &done,
      &args->common->Xperm[i * args->common->rx1 * args->common->rx2], &args->common->rx1, 
      dtmp1, &args->common->rx1,
      &dzero, 
      dtmp2, &args->common->rx2
    );
    // Permute tmp2
    // Was rx2*rc2, rx2', needed rx2', rx2*rc2
    for (j=0; j<args->common->rx2; j++)
      dcopy_(&tmpsize, &dtmp2[j*tmpsize], &ione, &dtmp1[j], &args->common->rx2);
    // Add to XAX1_new (from F order to C order)
    tmpsize = args->common->rx2 * args->common->rc2;
    for (j=0;j<args->common->rx2;++j)
      for (k=0;k<args->common->rc2;++k)
        daxpy_(
          &args->common->rx2, 
          &done, 
          &dtmp1[args->common->rx2 * (j + args->common->rx2 * k)], &ione, 
          &args->XAX1_new_local[j*args->common->rc2+k], &tmpsize
        );
  }

  free(dtmp1);
  free(dtmp2);

  pthread_exit(NULL);
}

/*! \brief Project diagonal system in parallel
 *
 * @param XAX shape (rx1,rx1,rc1)
 * @param Ci shape (rc1,n1,rc2)
 * @param X shape (rx1,n1,rx2)
 * @param XAX_new shape (rx2,rx2,rc2)
 * @param rx1
 * @param rx2
 * @param rc1
 * @param n1
 * @param nthread number of CPU threads to use
*/
void project_blockdiag_parallel(double* XAX1, double* Ci, double* X, double* XAX1_new, int rx1, int rx2, int rc1, int rc2, int n1, int nthread)
{
  /* Todo: Make it adaptive to complex arithmetics! */
  double *Xperm, *Cperm, *XAX1_F;
  int i,j, tmpsize;

  // Clear output
  memset(XAX1_new, 0, sizeof(double)*rx2*rx2*rc2);

  // Permute X and C for easier BLAS (and transfer to Fortran order)
  Xperm = (double*) malloc(sizeof(double)*rx1*rx2*n1);
  for (i=0; i<rx1; i++)
    for (j=0; j<n1; j++)
      dcopy_(&rx2, &X[(i*n1+j)*rx2], &ione, &Xperm[i+j*rx1*rx2], &rx1);

  Cperm = (double*) malloc(sizeof(double)*rc1*rc2*n1);
  for (i=0; i<rc1; i++)
    for (j=0; j<n1; j++)
      dcopy_(&rc2, &Ci[(i*n1+j)*rc2], &ione, &Cperm[i+j*rc1*rc2], &rc1);

  // get XAX1 in Fortran order
  tmpsize = rx1*rx1;
  XAX1_F = (double*) malloc(sizeof(double)*tmpsize*rc1);
  for (i=0;i<rx1;++i)
    for (j=0;j<rx1;++j)
      dcopy_(&rc1, &XAX1[(i*rx1+j)*rc1], &ione, &XAX1_F[i+j*rx1], &tmpsize);

  // Temp storage
  // Determine the maximal size we will need
  tmpsize = rx2>rx1 ? rx2*rx2 : rx2*rx1;
  tmpsize *= rc2>rc1 ? rc2 : rc1;

  // Main loop is parallel
  pthread_t threads[nthread];
  double* XAX1_new_local[nthread];
  struct project_block_common_args common = {rx1, rx2, rc1, rc2, n1, tmpsize, Xperm, Cperm, XAX1_F};
  struct project_block_args args[nthread];
  
  tmpsize = rx2*rx2*rc2;
  for (i=0;i<nthread;++i){
    XAX1_new_local[i] = (double*) malloc(sizeof(double)*tmpsize);
    memset(XAX1_new_local[i], 0, sizeof(double)*tmpsize);
    args[i].i_start = i;
    args[i].stride = nthread;
    args[i].XAX1_new_local = XAX1_new_local[i];
    args[i].common = &common;
  }

  for(i=0;i<nthread;++i)
    j = pthread_create(&threads[i], NULL, project_block, (void*) &args[i]);

  for(i=0;i<nthread;++i)
  {
    pthread_join(threads[i], NULL);
    daxpy_(&tmpsize, &done, XAX1_new_local[i], &ione, XAX1_new, &ione);
    free(XAX1_new_local[i]);
  }


  free(XAX1_F);
  free(Cperm);
  free(Xperm);
}