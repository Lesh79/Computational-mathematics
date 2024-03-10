#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 498
#define CHUNK 50
#define EPS 0.01

double g(double x, double y) {
    if (x == 0) return 100 - 200 * x;
    if (y == 0) return 100 - 200 * y;
    if (x == 1) return 200 * x - 100;
    if (y == 1) return 200 * y - 100;
}

double f(double x, double y) {
    return 0.0;
}

typedef struct {
    double** u;
    double h;
    int n;
    double** f;
} Matrix;

Matrix * createMatrix(double(*f_fun)(double, double), double(*g_fun)(double, double)) {
    Matrix * net = (Matrix *)malloc(sizeof(Matrix));
    net->n = N;
    net->h = 1.0 / (net->n + 1);
    net->u = (double**)malloc((net->n + 2) * sizeof(double*));
    net->f = (double**)malloc((net->n + 2) * sizeof(double*));

    for (int i = 0; i < net->n + 2; i++) {
        net->u[i] = (double*)calloc((net->n + 2), sizeof(double));
        net->f[i] = (double*)malloc((net->n + 2) * sizeof(double));
    }

    for (int i = 1; i <= net->n; i++) {
        for (int j = 1; j <= net->n; j++) {
            net->f[i][j] = f_fun(i * net->h, j * net->h);
        }
    }

    for (int i = 0; i <= net->n + 1; i++) {
        net->u[i][0] = g_fun(i * net->h, 0);
        net->u[i][net->n + 1] = g_fun(i * net->h, (net->n + 1) * net->h);
        net->u[0][i] = g_fun(0, i * net->h);
        net->u[net->n + 1][i] = g_fun((net->n + 1) * net->h, i * net->h);
    }

    return net;
}

void freeMatrix(Matrix * net){
    for (int i = 0; i < N + 2; i++) {
        free(net->u[i]);
        free(net->f[i]);
    }
    free(net->u);
    free(net->f);
    free(net);
}

int sequential(Matrix * net){
    int iteration = 0;
    double dmax;
    double temp;
    do{
        dmax = 0;

        for (int i = 1; i <= net->n; i++){
            for (int j = 1; j <= net->n; j++){
                temp = net->u[i][j];
                net->u[i][j] = 0.25 * (net->u[i - 1][j] + net->u[i + 1][j] + net->u[i][j - 1] + net->u[i][j + 1] - net->h * net->h * net->f[i][j]);
                double dm = fabs(temp - net->u[i][j]);
                if (dm > dmax) dmax = dm;
            }
        }
        iteration++;
    }while (dmax > EPS);
    return iteration;
}

int algorithm_11_3(Matrix * net) {
    int iteration = 0;
    double dmax, temp, d, dm;
    omp_lock_t dmax_lock;
    omp_init_lock(&dmax_lock);

    do {
        dmax = 0;

#pragma omp parallel for shared(net, dmax) private(temp, d, dm) default(none)
        for (int i = 1; i < net->n + 1; i++) {
            dm = 0;

            for (int j = 1; j < net->n + 1; j++) {
                temp = net->u[i][j];
                net->u[i][j] = 0.25 * (net->u[i - 1][j] + net->u[i + 1][j] + net->u[i][j - 1] + net->u[i][j + 1] - net->h * net->h * net->f[i][j]);
                d = fabs(temp - net->u[i][j]);

                if (dm < d) dm = d;
            }

#pragma omp critical
            {
                if (dmax < dm) dmax = dm;
            }
        }

        iteration++;
    } while (dmax > EPS);

    omp_destroy_lock(&dmax_lock);
    return iteration;
}


double MatrixBlock(Matrix * net, int x, int y) {
    int start_i = 1 + x * CHUNK;
    int start_j = 1 + y * CHUNK;

    double dmax = 0;
    for (int i = start_i; i <= fmin(start_i + CHUNK - 1, net->n); i++) {
        for (int j = start_j; j <= fmin(start_j + CHUNK - 1, net->n); j++) {
            double temp = net->u[i][j];
            net->u[i][j] = 0.25 * (net->u[i - 1][j] + net->u[i + 1][j] + net->u[i][j - 1] + net->u[i][j + 1] - net->h * net->h * net->f[i][j]);
            double dm = fabs(temp - net->u[i][j]);
            if (dmax < dm) dmax = dm;
        }
    }
    return dmax;
}

int algorithm_11_6(Matrix * net) { // 11.6

    int nb = net->n / CHUNK + (net->n % CHUNK != 0);
    double dmax, dm[nb];
    int iteration = 0;
    do {
        dmax = 0;
#pragma omp parallel for shared(net, nb, dm) default(none)
        for (int nx = 0; nx < nb; nx++) { // нарастание волны
            dm[nx] = 0;

            int i, j;
            double d;

#pragma omp parallel for shared(nx, dm, net) private(i, j, d) default(none)
            for (i = 0; i <= nx; i++) {
                j = nx - i;
                d = MatrixBlock(net, i, j);
                dm[i] = fmax(dm[i], d);
            }
        }

        for (int nx = nb - 1; nx >= 1; nx--) { // затухание волны
            int i, j;
            double d;
#pragma omp parallel for shared(net, nb, dm, nx) private(i, j, d) default(none)
            for (i = nb - nx; i < nb; i++) {
                j = 2 * (nb - 1) - nx - i + 1;
                d = MatrixBlock(net, i, j);
                dm[i] = fmax(dm[i], d);
            }
        }

        for (int i = 0; i < nb; i++) {
            dmax = fmax(dmax, dm[i]);
        }
        iteration++;
    } while (dmax > EPS);
    return iteration;
}

void printRes(Matrix *net, double alg) {
    int iterations;
    double start_time = omp_get_wtime();
    if (alg == 11.6){
        iterations = algorithm_11_6(net);
    }
    else {
        iterations = algorithm_11_3(net) ;
    }
    double end_time = omp_get_wtime();
    printf("------------------\n");
    printf("||%.1f algorithm||\n", alg);
    printf("------------------\n");
    printf("Time: %f\n", end_time - start_time);
    printf("ITERATIONS: %d \n\n", iterations);
}


int main() {
    int TREADS[] = {1, 4, 6, 8, 12};

    for (int idx = 0; idx < sizeof(TREADS) / sizeof(TREADS[0]); idx++) {
        omp_set_num_threads(TREADS[idx]);


        printf("-=-=-=-=-=-=-=-=-=-=-=-= %d -=-=-=-=-=-=-=-=-=-=-=-=\n", TREADS[idx]);
        Matrix * net1 = createMatrix(f, g);
        printRes(net1, 11.6);
        freeMatrix(net1);

//        Matrix * net2 = createMatrix(f, g);
//        printRes(net2, 11.3);
//        freeMatrix(net2);

    }
    Matrix * net = createMatrix(f, g);
    double start_time = omp_get_wtime();
    int iterations = sequential(net);
    double end_time = omp_get_wtime();
    printf("-=-=-=-=-=-=-=-=-=-=-=-= SEQUENTIAL -=-=-=-=-=-=-=-=-=-=-=-=\n" );
    printf("TIME: %f\n", end_time - start_time);
    printf("ITERATION %d\n", iterations);

    freeMatrix(net);

    return 0;
}
