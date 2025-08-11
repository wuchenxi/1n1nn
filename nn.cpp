//usage: ./nn 1000

#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<ctime>

int dim;
int nsample=200;

double relu(double x){
    return (x>0?x:0);
}

int initsample(double* &x, double* &y, int n){
    x=(double*)malloc(n*sizeof(double));
    y=(double*)malloc(n*sizeof(double));
    for(int i=0;i<n;i++){
        x[i]=((double)rand())/RAND_MAX*2-1;
        y[i]=relu(x[i]);
    }
    return 0;
}

int init_nn(int nneuron, double**& nn, double var=1){
    nn=(double**)malloc(3*sizeof(double*));
    for(int i=0;i<2;i++){
        nn[i]=(double*)malloc(nneuron*sizeof(double));
    }
    double bound1=var;
    double bound2=var/sqrt(nneuron);
    for(int i=0;i<2;i++){
        for(int j=0;j<nneuron;j++){
            nn[i][j]=((double)rand())/RAND_MAX*2*bound1-bound1;
        }
    }
    nn[2]=(double*)malloc((nneuron+1)*sizeof(double));
    for(int j=0;j<nneuron+1;j++){
        nn[2][j]=((double)rand())/RAND_MAX*2*bound2-bound2;
    }
    return 0;
}

int randomize_nn(int nneuron, double** nn, double var=1){
    double bound1=var;
    double bound2=var/sqrt(nneuron);
    for(int i=0;i<2;i++){
        for(int j=0;j<nneuron;j++){
            nn[i][j]=((double)rand())/RAND_MAX*2*bound1-bound1;
        }
    }
    for(int j=0;j<nneuron+1;j++){
        nn[2][j]=((double)rand())/RAND_MAX*2*bound2-bound2;
    }
    return 0;
}

int lin_comb(double** nn, double** nn1, double** out, int dim, double scale1, double scale2){
    for(int i=0;i<3;i++){
        for(int j=0;j<dim;j++){
            out[i][j]=scale1*nn[i][j]+scale2*nn1[i][j];
        }
     out[2][dim]=scale1*nn[2][dim]+scale2*nn1[2][dim];
    }
    return 0;
}

int lin_comb2(double** nn, double** nn1, double** out, int dim, double scale1, double scale2){
    for(int i=0;i<3;i++){
        for(int j=0;j<dim;j++){
            out[i][j]=scale1*nn[i][j]+scale2*nn1[i][j]*nn1[i][j];
        }
        out[2][dim]=scale1*nn[2][dim]+scale2*nn1[2][dim]*nn1[2][dim];
    }
    return 0;
}


double sumroot(double** nn, int dim){
    double s=0;
    for(int i=0;i<3;i++){
        for(int j=0;j<dim;j++){
            s+=nn[i][j]*nn[i][j];
        }
    }
    s+=nn[2][dim]*nn[2][dim];
    return sqrt(s);
}

int zero_out(int nneuron, double ** nn){
    for(int i=0;i<3;i++){
        for(int j=0;j<nneuron;j++){
            nn[i][j]=0;
        }
    }
    nn[2][nneuron]=0;
    return 0;
}

int batch_eval(double* x, double* y, int ns, double** nn, int dim){
    for(int i=0;i<ns;i++){
        y[i]=nn[2][dim];
        for(int j=0;j<dim;j++){
            y[i]+=relu(x[i]*nn[0][j]+nn[1][j])*nn[2][j];
        }
    }
    return 0;
}

int get_grad(int dim, int ns, double* x, double* y, double* yhat, double** nn, double** grad){
    zero_out(dim, grad);
    batch_eval(x, yhat, ns, nn, dim);
    for(int i=0;i<ns;i++){
        grad[2][dim]+=y[i]-yhat[i];
        for(int j=0;j<dim;j++){
            double t=x[i]*nn[0][j]+nn[1][j];
            grad[2][j]+=(y[i]-yhat[i])*t;
            if(t>0){
                grad[1][j]+=(y[i]-yhat[i])*nn[2][j];
                grad[0][j]+=(y[i]-yhat[i])*nn[2][j]*x[i];
            }
        }
    }
    return 0;
}

int rand_idx(int n, int* r){
    for(int i=n-1;i>0;i--){
        int j=rand()%(i+1);
        int tmp=r[j];
        r[j]=r[i];
        r[i]=tmp;
    }
    return 0;
}

int sgd(int dim, int ns,  double* x, double* y, double* xb, double* yb, double *ybhat, double** nn, double** grad, double** m, double** v, int* idx, int nbatch=128, double lr=0.0001, int nsteps=100, double step=0.001){
    rand_idx(ns, idx);
    int id=0;
    int szbatch=nbatch;
    while(id<ns){
        if(id+nbatch>ns)
            szbatch=ns-id;
        for(int i=0;i<szbatch;i++){
             xb[i]=x[id+i];
             yb[i]=y[id+i];
        }
       //adam
       zero_out(dim,m);
       zero_out(dim, v);
       double beta1=0.9;
       double beta2=0.999;
       double nbeta1=beta1;
       double nbeta2=beta2;
       double epsilon=0.00000001;
       for(int t=0;t<nsteps;t++){
           get_grad(dim, szbatch, xb, yb, ybhat, nn, grad);
           lin_comb(m, grad, m, dim, beta1, 1-beta1);
           lin_comb2(v, grad, v, dim, beta2, 1-beta2);
           lin_comb(nn, m, nn, dim, 1, step/((sumroot(v, dim)/(1-nbeta2)+epsilon)/(1-nbeta1)));
           nbeta1*=beta1;
           nbeta2*=beta2;
       }
       id+=nbatch;
    }
    return 0;
}

int main(int argc, char** argv){
    //srand(time(0));
    srand(42);
    sscanf(argv[1], "%d", &dim);
    double* xs;
    double* ys;
    double* xt;
    double* yt;
    initsample(xs, ys, nsample);
    initsample(xt, yt, nsample);
    int* idx=(int*)malloc(nsample*sizeof(int));
    for(int i=0;i<nsample;i++)idx[i]=i;
    double sum=0;
    for(int i=0;i<nsample;i++)sum+=yt[i];
    sum/=nsample;
    double vars=0;
    for(int i=0;i<nsample;i++)vars+=(yt[i]-sum)*(yt[i]-sum);
    //printf("%g\n", vars);
    double* yt_hat=(double*)malloc(nsample*sizeof(double));
    double** nn;
    double** grad;
    double** m;
    double** v;
    init_nn(dim, nn);
    init_nn(dim, grad);
    init_nn(dim, m);
    init_nn(dim, v);
    double* xb=(double*)malloc(128*sizeof(double));
    double* yb=(double*)malloc(128*sizeof(double));
    double* ybhat=(double*)malloc(128*sizeof(double));
    int count=0;
    for(int ntrial=0; ntrial<100; ntrial++){
       for(int t=0;t<=100;t++){
          sgd(dim, nsample, xs, ys, xb, yb, ybhat, nn, grad, m, v,idx);
       }
       batch_eval(xt, yt_hat, nsample, nn, dim);
       double s=0;
        for(int i=0;i<nsample;i++){
            s+=(yt[i]-yt_hat[i])*(yt[i]-yt_hat[i]);
          }
        if(s<0.01*vars){
            count+=1;
        }
        randomize_nn(dim, nn);
    }
    printf("%d\n", count);
    return 0;
}


