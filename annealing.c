#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define PI acos(-1.0)
#define E exp(1.0)

// Task represents function, dimension and bounds
// within which its optimized
struct task {
    int dim;
    double* low_b;
    double* up_b;
    double (*fun)(double*, int);
};

// Params struct contains hypeparams of the task
struct params {
    int fun_number;
    double t0;
    double eps;
};

double get_temperature(double t0, int step)
{
    return t0 / (1.0 + step);
}

double transition_prob(double delta, double temperature)
{
    return exp(-delta/temperature);
}

double sample_standard_uniform()
{
    return (double)rand() / (double)RAND_MAX;
}

double sample_uniform(double low, double up)
{
    return low + (up - low)*sample_standard_uniform();
}

// Generate sample from standard normal
// distribution with Box-Muller method
double sample_standard_normal()
{
    double uni1 = sample_standard_uniform();
    double uni2 = sample_standard_uniform();
    return sqrt(-2*log(uni1)) * cos(2*PI*uni2);
}

void gen_new_point(double* x, double sqrt_temp, int dim, double* result)
{
    for (int i = 0; i < dim; i++) {
        result[i] = x[i] + sqrt_temp*sample_standard_normal();
    }
}

// Retain generated point within [low_b, up_b]
double clip(double val, double low_b, double up_b)
{
    if (val > up_b) {
        return up_b;
    } else if (val < low_b) {
        return low_b;
    } else {
        return val;
    }
}

double annealing_optimizer(struct task* t, struct params* p, double* res)
{
    int dim = t->dim;
    double (*fun)(double*, int) = t->fun;
    double* low_b = t->low_b;
    double* up_b  = t->up_b;
    double* cur_x = malloc(dim*sizeof(double));
    double* cand  = malloc(dim*sizeof(double));
    for (int i = 0; i < dim; i++) {
        cur_x[i] = sample_uniform(low_b[i], up_b[i]);
    }
    double cur_f = (*fun)(cur_x, dim);
    double cand_f, alpha, delta;
    double cur_temp;

    int step = 1;
    do {
        cur_temp = get_temperature(p->t0, step);
        gen_new_point(cur_x, sqrt(cur_temp), dim, cand);
        for (int i = 0; i < dim; i++) {
            cand[i] = clip(cand[i], low_b[i], up_b[i]);
        }
        cand_f = (*fun)(cand, dim);
        alpha = sample_standard_uniform();
        delta = cand_f - cur_f;
        if (transition_prob(delta, cur_temp) > alpha) {
            for (int i = 0; i < dim; i++) {
                cur_x[i] = cand[i];
            }
            cur_f = cand_f;
            step += 1;
            if (step % 1000 == 0) {
                printf("step = %d; temperature = %f\n", step, cur_temp);
            }
        }
    } while (cur_temp > p->eps);
    
    for (int i = 0; i < dim; i++) {
        res[i] = cur_x[i];
    }
    return cur_f;
}

// Test functions
double sphere(double* x, int dim)
{
    double res = 0.0;
    for (int i = 0; i < dim; i++) {
        res += pow(x[i], 2);
    }
    return res;
}

double rosenbrock(double* x, int dim)
{
    double res = 0.0;
    for (int i = 0; i < dim - 1; i++) {
        res += 100*pow(x[i+1] - pow(x[i], 2), 2) + pow(x[i] - 1, 2);
    }
    return res;
}

double ackley(double* x, int dim)
{
    double summand1 = -20*exp(-0.2*sqrt(0.5*(pow(x[0], 2) + pow(x[1], 2))));
    double summand2 = -exp(0.5*(cos(2*PI*x[0]) + cos(2*PI*x[1])));
    return summand1 + summand2 + E + 20;
}

double rastrigin(double* x, int dim)
{
    double res = 0.0;
    for (int i = 0; i < dim; i++) {
        res += pow(x[i], 2) - 10*cos(2*PI*x[i]);
    }
    return res + 10*dim;
}

double levy(double* x, int dim)
{
    double summand1 = pow(sin(3*PI*x[0]), 2) +
        pow(x[0] - 1, 2)*(1 + pow(sin(3*PI*x[1]), 2));
    double summand2 = pow(x[1] - 1, 2)*(1 + pow(sin(2*PI*x[1]), 2));
    return summand1 + summand2;
}

double himmelblau(double* x, int dim)
{
    return pow(pow(x[0], 2) + x[1] - 11, 2) + pow(x[0] + pow(x[1], 2) - 7, 2);
}

double mccormick(double* x, int dim)
{
    return sin(x[0] + x[1]) + pow(x[0] - x[1], 2) - 1.5*x[0] + 2.5*x[1] + 1;
}

double eggholder(double* x, int dim)
{
    double summand1 = -(x[1] + 47)*sin(sqrt(abs(x[0]/2 + x[1] + 47)));
    double summand2 = -x[0]*sin(sqrt(abs(x[0] - x[1] - 47)));
    return summand1 + summand2;
}

double easom(double* x, int dim)
{
    double aux = -(pow(x[0] - PI, 2) + pow(x[1] - PI, 2));
    return -cos(x[0])*cos(x[1])*exp(aux);
}

//TODO
double cross_in_tray(double* x, int dim)
{
    double aux1 = abs(100 - sqrt(pow(x[0], 2) + pow(x[1], 2))/PI);
    double aux2 = abs(sin(x[0])*sin(x[1])*exp(aux1));
    return -0.0001*pow(aux2 + 1, 0.1);
}

double camel(double* x, int dim)
{
    return 2*pow(x[0], 2) - 1.05*pow(x[0], 4) +
        pow(x[0], 6) / 6 + x[0]*x[1] + pow(x[1], 2);
}

double but(double* x, int dim)
{
    return pow(x[0] + 2*x[1] - 7, 2) + pow(2*x[0] + x[1] - 5, 2);
}

double mathias(double* x, int dim)
{
    return 0.26*(pow(x[0], 2) + pow(x[1], 2)) - 0.48*x[0]*x[1];
}

double bukin(double* x, int dim)
{
    double aux = 100*sqrt(abs(x[1] - 0.01*pow(x[0], 2)));
    return aux + 0.01*abs(x[0] + 10);
}

double bill(double* x, int dim)
{
    double s1 = pow(1.5 - x[0] + x[0]*x[1], 2);
    double s2 = pow(2.25 - x[0] + x[0]*pow(x[1], 2), 2);
    double s3 = pow(2.625 - x[0] + x[0]*pow(x[1], 3), 2);
    return s1 + s2 + s3;
}

double goldstein_price(double* x, int dim)
{
    double m1 = 19 - 14*x[0] + 3*pow(x[0], 2) - 14*x[1] +
        6*x[0]*x[1] + 3*pow(x[1], 2);
    double m2 = 1 + pow(x[0] + x[1] + 1, 2)*m1;
    double m3 = 18 - 32*x[0] + 12*pow(x[0], 2) + 48*x[1] -
        36*x[0]*x[1] + 27*pow(x[1], 2);
    double m4 = 30 + pow(2*x[0] - 3*x[1], 2)*m3;
    return m2*m4;
}

double holder(double* x, int dim)
{
    double m1 = exp(abs(1 - sqrt(pow(x[0], 2) + pow(x[1], 2)) / PI));
    return -abs(sin(x[0])*cos(x[1])*m1);
}

double sheffer2(double* x, int dim)
{
    double a1 = pow(sin(pow(x[0], 2) - pow(x[1], 2)), 2) - 0.5;
    double a2 = pow(1 + 0.001*(pow(x[0], 2) + pow(x[1], 2)), 2);
    return 0.5 + a1 / a2;
}

double sheffer4(double* x, int dim)
{
    double a1 = pow(cos(sin(abs(pow(x[0], 2) - pow(x[1], 2)))), 2) - 0.5;
    double a2 = pow(1 + 0.001*(pow(x[0], 2) + pow(x[1], 2)), 2);
    return 0.5 + a1 / a2;
}

double tang(double* x, int dim)
{
    double res = 0.0;
    for (int i = 0; i < dim; i++) {
        res += pow(x[i], 4) - 16*pow(x[i], 2) + 5*x[i];
    }
    return res / 2.;
}

// Generate task where function is considered within symmetic
// bounds in each component
struct task* make_symmetric_bounds_task(int dim, double boundary,
      double (*fun)(double*, int))
{
    struct task* cur = malloc(sizeof(struct task));
    cur->fun = fun;
    cur->dim = dim;
    cur->low_b = malloc(dim*sizeof(double));
    cur->up_b  = malloc(dim*sizeof(double));
    for (int i = 0; i < dim; i++) {
        cur->low_b[i] = -boundary;
        cur->up_b[i]  = boundary;
    }
    return cur;
}

struct task* make_bukin_task()
{
    struct task* cur = malloc(sizeof(struct task));
    int dim = 2;
    cur->dim = dim;
    cur->fun = bukin;
    cur->low_b = malloc(dim*sizeof(double));
    cur->up_b  = malloc(dim*sizeof(double));
    cur->low_b[0] = -15.;
    cur->low_b[1] = -3.;
    cur->up_b[0]  = -5.;
    cur->up_b[1]  = 3.;
    return cur;
}

struct task* make_mccormick_task()
{
    struct task* cur = malloc(sizeof(struct task));
    int dim = 2;
    cur->dim = dim;
    cur->fun = mccormick;
    cur->low_b = malloc(dim*sizeof(double));
    cur->up_b  = malloc(dim*sizeof(double));
    cur->low_b[0] = -1.5;
    cur->low_b[1] = -3.;
    cur->up_b[0]  = 4.;
    cur->up_b[1]  = 4.;
    return cur;
}

struct task* make_task(int n, int dim, double* boundaries, double (*(funs[])) (double*, int))
{
    if (n >= 1 && n <= 18) {
        return make_symmetric_bounds_task(dim, boundaries[n-1], funs[n-1]);
    } else if (n == 19) {
        return make_bukin_task();
    } else {
        return make_mccormick_task();
    }
}

void print_result(double* min_x, double min_f, int dim) {
    printf("Minimum value = %f\nmin_x = (\n", min_f);
    for (int i = 0; i < dim; i++) {
        printf("  %f\n", min_x[i]);
    }
    printf(")\n");
}

// Select function, initial temperature and stop condition
struct params* dialogue()
{
    struct params* cur = malloc(sizeof(struct params));
    printf("Choose test function.\nOptions:\n");
    printf("1:  Rastrigin\n");
    printf("2:  Ackley\n");
    printf("3:  Sphere\n");
    printf("4:  Rosencrock\n");
    printf("5:  Bill\n");
    printf("6:  Goldstein-Price\n");
    printf("7:  Butt\n");
    printf("8:  Mathias\n");
    printf("9:  Levy\n");
    printf("10: Himmelblau\n");
    printf("11: Camel\n");
    printf("12: Easom\n");
    printf("13: Cross in tray\n");
    printf("14: Eggholder\n");
    printf("15: Holder\n");
    printf("16: Sheffer2\n");
    printf("17: Sheffer4\n");
    printf("18: Tang\n");
    printf("19: Bukin\n");
    printf("20: McCormick\n");
    printf("Choice: ");
    scanf("%d", &(cur->fun_number));
    printf("Initial temperature: ");
    scanf("%lf", &(cur->t0));
    printf("Eps (stop condition, e.g. 0.001): ");
    scanf("%lf", &(cur->eps));
    return cur;
}

int ask_dimension()
{
    int dim;
    printf("Dimension: ");
    scanf("%d", &dim);
    return dim;
}

int main()
{
    double (*funs[])(double*, int) = { rastrigin, ackley, sphere, rosenbrock,
        bill, goldstein_price, but, mathias, levy, himmelblau, camel, easom,
        cross_in_tray, eggholder, holder, sheffer2, sheffer4, tang
    };
    double boundaries[] = { 5.12, 5., 100., 100., 4.5, 2., 10., 10., 10.,
        5., 5., 100., 10., 512., 10., 100., 100., 5.
    };
    int dims[] = { 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 0, 2, 2 };

    srand(time(NULL));
    
    struct params* p = dialogue();
    int n = p->fun_number;
    int dim = 2;
    if (dims[n - 1] == 0) {
        dim = ask_dimension();
    }
    struct task* t = make_task(n, dim, boundaries, funs);
    
    // min_x is an extremum point
    double* min_x = malloc(t->dim * sizeof(double));
    double min_f = annealing_optimizer(t, p, min_x);

    print_result(min_x, min_f, t->dim);
    return 0;
}
