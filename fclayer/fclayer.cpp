#include <fstream> 
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <sys/time.h>
#include "polaris.h"
using namespace std;

#define SPARSE_DIM 260
#define DENSE_DIM 13
#define BATCH 100
#define TOTAL 100

//#define CPU

typedef struct
{
    int height;
    int width;
    float *elements;
} Matrix;


/************** files *******************/
char file_sparse_embed[1000] = "sparse_embed.txtb";
char file_dense_input[1000] = "dense_input.txtb";

char file_dense_fm[1000] = "dense_fm.txtb";
char file_sparse_fm[1000] = "sparse_fm.txtb";

char file_weight_1_sparse[1000] = "weight_1_sparse.txt";
char file_weight_1_dense[1000] = "weight_1_dense.txt";
char file_weight_2[1000] = "weight_2.txt";
char file_weight_3[1000] = "weight_3.txt";
char file_weight_4_dense[1000] = "weight_4_dense.txt";
char file_weight_4_sparse[1000] = "weight_4_sparse.txt";
char file_weight_4[1000] = "weight_4.txt";

char file_bias_1[1000] = "bias_1.txt";
char file_bias_2[1000] = "bias_2.txt";
char file_bias_3[1000] = "bias_3.txt";
char file_bias_4[1000] = "bias_4.txt";

char file_output_result[1000] = "output_result.txt";

/*************************************************
* CPU
*************************************************/
/************* EMBED *****************/
float test_tt1[TOTAL * DENSE_DIM];
float dense_input[TOTAL * DENSE_DIM];
float sparse_embed[TOTAL * SPARSE_DIM];

/*************  FM  *****************/
float dense_fm[TOTAL];
float sparse_fm[TOTAL];

/*************  weight  *****************/
float weight_1_sparse[SPARSE_DIM * 400];
float weight_1_dense[DENSE_DIM * 400];
float weight_2[400 * 400];
float weight_3[400 * 400];
float weight_4_dense[1];
float weight_4_sparse[1];
float weight_4[400];

/*************  BIAS  *****************/
float bias_1[400];
float bias_2[400];
float bias_3[400];
float bias_4[1];

/*************  BIAS ZERO *****************/
float bias_zero[400];

/*************  OUTPUT *****************/
float output_1[BATCH * 400];
float output_2[BATCH * 400];
float output_3[BATCH * 400];
float output_result[TOTAL];
float test_tt2[TOTAL];
/*************************************************
* CPU END
*************************************************/

/*************************************************
* Polaris
*************************************************/
/*************  CTXT  *****************/
PolarisContext *ctxt = NULL;

/************* EMBED *****************/
float *polars_sparse_embed = NULL;
float *polars_dense_input = NULL;

/*************  FM  *****************/
float *polars_dense_fm = NULL;
float *polars_sparse_fm = NULL;

/*************  weight  *****************/
float *polars_weight_1_sparse = NULL;
float *polars_weight_1_dense = NULL;
float *polars_weight_2 = NULL;
float *polars_weight_3 = NULL;
float *polars_weight_4_dense = NULL;
float *polars_weight_4_sparse = NULL;
float *polars_weight_4 = NULL;

/*************  BIAS  *****************/
float *polars_bias_1 = NULL;
float *polars_bias_2 = NULL;
float *polars_bias_3 = NULL;
float *polars_bias_4 = NULL;

/*************  BIAS ZERO *****************/
float *polars_bias_zero = NULL;

/*************  OUTPUT *****************/
float *polars_output_1 = NULL;
float *polars_output_2 = NULL;
float *polars_output_3 = NULL;
float *polars_output_result = NULL;
/*************************************************
* Polaris END
*************************************************/


void fill_zero(float a[], int length)
{
    for(int i = 0; i < length; i++)
    {
        a[i] = 0.0;
    }
}

void simple_sgemm(const float *A, const float *B, float *C, const int M, const int N, const int K, const float *bia)
{
    int i, j, e;
    #pragma omp parallel for num_threads(16)
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
        {
            for (e = 0; e < K; e++)
            {
                C[i * N + j] += A[i * K + e] * B[j * K + e];                
            }
            C[i * N + j] += bia[j];
        }
}

#define MAX_LEN 65535
PolarisStatus polaris_matmul(Matrix A, Matrix B, Matrix C, float *bias)
{
    return polaris_gemm(ctxt, C.height, C.width, A.width, A.elements, B.elements, C.elements, bias);
}
{
    int m = C.height;
    int n = C.width;
    int k = A.height;

    int A_lenght
}

void simple_matmul(Matrix A, Matrix B, Matrix C, float *bias)
{
    simple_sgemm(A.elements, B.elements, C.elements, C.height, C.width, A.width, bias);
}

void fc_layer_kernel(Matrix *input, Matrix *weight, Matrix C, float *bias, PolarisActivationType act_tpye, int input_size, bool use_CPU)
{
    for (int i = 0; i < input_size; i++)
    {
        if(use_CPU)
        {
            if (i != input_size - 1)
                simple_matmul(input[i], weight[i], C, bias_zero);
            else
                simple_matmul(input[i], weight[i], C, bias);
        }    
        else
        {
            if (i != input_size - 1)
                polaris_matmul(input[i], weight[i], C, polars_bias_zero);
            else
                polaris_matmul(input[i], weight[i], C, bias);
        }
    }
    
    if(use_CPU) 
        for(int i = 0; i < C.height * C.width; i++)
            if(act_tpye == POLARIS_RELU)        
                C.elements[i] = C.elements[i] > 0? C.elements[i] : 0;
            else
                C.elements[i] = (1 / (1 + exp(-C.elements[i])));
    else
        polaris_activation(ctxt, act_tpye, C.height * C.width, 1, C.elements, 0, C.elements);
}

void fc_layer_1(float *sparse_layer, float *dense_layer, float *weight0, float *weight1, float *output, float *bias, bool use_CPU)
{
    Matrix fc1_input[2];
    fc1_input[0].height = BATCH;
    fc1_input[0].width = SPARSE_DIM;
    fc1_input[0].elements = sparse_layer;
    fc1_input[1].height = BATCH;
    fc1_input[1].width = DENSE_DIM;
    fc1_input[1].elements = dense_layer;

    Matrix fc1_weght[2];
    fc1_weght[0].height = SPARSE_DIM;
    fc1_weght[0].width = 400;
    fc1_weght[0].elements = weight0;
    fc1_weght[1].height = DENSE_DIM;
    fc1_weght[1].width = 400;
    fc1_weght[1].elements = weight1;


    if(use_CPU)
        fill_zero(output, BATCH * 400);
    else
        polaris_elementwise(ctxt, POLARIS_ADD, BATCH * 400, 0, output, 0, output, 0, output);

    //polaris_memset(ctxt, output, BATCH * 400);
    Matrix fc1_output;
    fc1_output.height = BATCH;
    fc1_output.width = 400;
    fc1_output.elements = output;

    fc_layer_kernel(fc1_input, fc1_weght, fc1_output, bias, POLARIS_RELU, 2, use_CPU);
}

void fc_layer_2_3(float *input, float *weight, float *output, float *bias, bool use_CPU)
{
    Matrix fc1_input[1];
    fc1_input[0].height = BATCH;
    fc1_input[0].width = 400;
    fc1_input[0].elements = input;

    Matrix fc1_weght[1];
    fc1_weght[0].height = 400;
    fc1_weght[0].width = 400;
    fc1_weght[0].elements = weight;

    if(use_CPU)
        fill_zero(output, BATCH * 400);
    else
        polaris_elementwise(ctxt, POLARIS_ADD, BATCH * 400, 0, output, 0, output, 0, output);

    //polaris_memset(ctxt, output, BATCH * 400);
    Matrix fc1_output;
    fc1_output.height = BATCH;
    fc1_output.width = 400;
    fc1_output.elements = output;

    fc_layer_kernel(fc1_input, fc1_weght, fc1_output, bias, POLARIS_RELU, 1, use_CPU);
}

void fc_layer_4(float *dense_fm, float *sparse_fm, float *input, float *weight0, float *weight1, float *weight2, float *output, float *bias, bool use_CPU)
{
    Matrix fc1_input[3];
    fc1_input[0].height = BATCH;
    fc1_input[0].width = 1;
    fc1_input[0].elements = dense_fm;
    fc1_input[1].height = BATCH;
    fc1_input[1].width = 1;
    fc1_input[1].elements = sparse_fm;
    fc1_input[2].height = BATCH;
    fc1_input[2].width = 400;
    fc1_input[2].elements = input;


    Matrix fc1_weght[3];
    fc1_weght[0].height = 1;
    fc1_weght[0].width = 1;
    fc1_weght[0].elements = weight0;
    fc1_weght[1].height = 1;
    fc1_weght[1].width = 1;
    fc1_weght[1].elements = weight1;
    fc1_weght[2].height = 400;
    fc1_weght[2].width = 1;
    fc1_weght[2].elements = weight2;

    if(use_CPU)
        fill_zero(output, BATCH);
    else
        polaris_elementwise(ctxt, POLARIS_ADD, BATCH, 0, output, 0, output, 0, output);

    //polaris_memset(ctxt, output, BATCH);
    Matrix fc1_output;
    fc1_output.height = BATCH;
    fc1_output.width = 1;
    fc1_output.elements = output;

    fc_layer_kernel(fc1_input, fc1_weght, fc1_output, bias, POLARIS_SIGMOID, 3, use_CPU);
}

void verify(float *a, float *b, int length)
{
    for (int i = 0; i < length; i++)
    {
        float err = fabs(a[i] - b[i]);
        if (err > 1e-3 && err > a[i] * 1e-3)
        {
            printf(" %d : %f, %f \n", i, a[i], b[i]);
            printf("ERR!\n");//, exit(-1);
        }
    }
}

void four_fc_layer()
{
    const int batch_num = TOTAL / BATCH;

    const int offset_sparse_embed = BATCH * SPARSE_DIM;
    const int offset_dense_input = BATCH * DENSE_DIM;
    const int offset_dense_fm = BATCH;
    const int offset_sparse_fm = BATCH;
    const int offset_output_result = BATCH;

#ifndef CPU   
    float *sparse_embed_polaris = polars_sparse_embed;
    float *dense_input_polaris = polars_dense_input;
    float *dense_fm_polaris = polars_dense_fm;
    float *sparse_fm_polaris = polars_sparse_fm;
    float *output_result_polaris = polars_output_result;
#else    
    float *sparse_embed_cpu = sparse_embed;
    float *dense_input_cpu = dense_input;
    float *dense_fm_cpu = dense_fm;
    float *sparse_fm_cpu = sparse_fm;
    float *output_result_cpu = output_result;
#endif

    for (int i = 0; i < batch_num; i++)
    {
    printf("batch id  : %d\n", i);

        #ifndef CPU         

        fc_layer_1(sparse_embed_polaris, dense_input_polaris, polars_weight_1_sparse, polars_weight_1_dense, polars_output_1, polars_bias_1, false);
        fc_layer_2_3(polars_output_1, polars_weight_2, polars_output_2, polars_bias_2, false);
        fc_layer_2_3(polars_output_2, polars_weight_3, polars_output_3, polars_bias_3, false);
        fc_layer_4(dense_fm_polaris, sparse_fm_polaris, polars_output_3, polars_weight_4_dense, polars_weight_4_sparse, polars_weight_4, output_result_polaris, polars_bias_4, false);
        
        sparse_embed_polaris += offset_sparse_embed;
        dense_input_polaris += offset_dense_input;
        dense_fm_polaris += offset_dense_fm;
        sparse_fm_polaris += offset_sparse_fm;
        output_result_polaris += offset_output_result;

        #else
        fc_layer_1(sparse_embed_cpu, dense_input_cpu, weight_1_sparse, weight_1_dense, output_1, bias_1, true);

/*
        float sparse_embed_temp[TOTAL * 260];    
        float dense_input_temp[TOTAL * 13]; 
        float weight_1_sparse_temp[260 * 400]; 
        float weight_1_dense_temp[13 * 400];       
        float bias_1_temp[13 * 400];       
        float output_1_tmep[BATCH * 400]; 
        
        polaris_memcpy(ctxt, POLARIS_DEVICE_TO_HOST, sparse_embed_temp, sparse_embed_polaris, TOTAL * 260 * sizeof(float));
        polaris_memcpy(ctxt, POLARIS_DEVICE_TO_HOST, dense_input_temp, dense_input_polaris, TOTAL * 13 * sizeof(float));
        polaris_memcpy(ctxt, POLARIS_DEVICE_TO_HOST, weight_1_sparse_temp, polars_weight_1_sparse, 260 * 400 * sizeof(float));
        polaris_memcpy(ctxt, POLARIS_DEVICE_TO_HOST, weight_1_dense_temp, polars_weight_1_dense, 13 * 400 * sizeof(float));
        polaris_memcpy(ctxt, POLARIS_DEVICE_TO_HOST, bias_1_temp, polars_bias_1, 1 * 400 * sizeof(float));
        polaris_memcpy(ctxt, POLARIS_DEVICE_TO_HOST, output_1_tmep, polars_output_1, BATCH * 400 * sizeof(float));

        printf("\nsparse_embed: ");
        verify(sparse_embed_temp, sparse_embed, TOTAL * 260);
        printf("\ndense_input: ");
        verify(dense_input_temp, dense_input, TOTAL * 13);
        printf("\nweight_1_sparse: ");
        verify(weight_1_sparse_temp, weight_1_sparse, 260 * 400);
        printf("\nweight_1_dense: ");
        verify(weight_1_dense_temp, weight_1_dense, 13 * 400);
        printf("\nbias_1: ");
        verify(bias_1_temp, bias_1, 1 * 400);
        
        printf("\noutput_1: ");
        verify(output_1_tmep, output_1, BATCH * 400);
        printf("\n");
*/

        fc_layer_2_3(output_1, weight_2, output_2, bias_2, true);
        fc_layer_2_3(output_2, weight_3, output_3, bias_3, true);
        fc_layer_4(dense_fm_cpu, sparse_fm_cpu, output_3, weight_4_dense, weight_4_sparse, weight_4, output_result_cpu, bias_4, true);

        sparse_embed_cpu += offset_sparse_embed;
        dense_input_cpu += offset_dense_input;
        dense_fm_cpu += offset_dense_fm;
        sparse_fm_cpu += offset_sparse_fm;
        output_result_cpu += offset_output_result;
        #endif
        
    }
}

void read_file(char file[], float* data, int M, int N)
{
    ifstream in;
    in.open(file);
    int m, n;
    in >> m;
    in >> n;
    if((m != M || n != N) && M != TOTAL)
    {
        cout << "Wrong file : " << file <<endl;
        //exit(1);
    }
    cout << file << " : " << m << " * " << n <<endl;
    
    float t_f;
    int i = 0;
    while(!in.eof()) 
    {
        in >> t_f;
        //cout << t_f <<endl;
        data[i] = t_f;
        i++;
        if(i == M * N)
            break;
    }
    if(i != M * N)
    {
        cout << "Wrong file : " << file <<endl;
        cout << " i : " << i << endl << " M * N " << M * N <<endl;
        exit(1);
    }
    in.close();
}


void polaris_transpose(float *mat, const int m, const int n)
{
    float *tem = (float *)malloc(m * n * sizeof(float));
    
    for (int i = 0; i < m * n; i++)
    {
        tem[i] = mat[i];
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mat[j * m + i] = tem[i * n + j];
        }
    }
}

void read_file(char file[], float* data, int M, int N, bool tran)
{
    ifstream in;
    in.open(file);
    int m, n;
    in >> m;
    in >> n;
    if(m != M || n != N)
    {
        cout << "Wrong file : " << file <<endl;
        //exit(1);
    }
    cout << file << " : " << m << " * " << n <<endl;
    
    float t_f;
    int i = 0;
    while(!in.eof()) 
    {
        in >> t_f;
        data[i] = t_f;
        i++;
        if(i == M * N)
            break;
    }
    if(i != M * N)
    {
        cout << "Wrong file : " << file <<endl;
        cout << " i : " << i << endl << " M * N " << M * N <<endl;
        exit(1);
    }
    polaris_transpose(data, M, N);
    in.close();
}


void read_data()
{   
    read_file(file_sparse_embed, sparse_embed, TOTAL, SPARSE_DIM);
    read_file(file_dense_input, dense_input, TOTAL, DENSE_DIM);

    read_file(file_dense_fm, dense_fm, TOTAL, 1);
    read_file(file_sparse_fm, sparse_fm, TOTAL, 1);

    read_file(file_weight_1_sparse, weight_1_sparse, SPARSE_DIM, 400, true);
    read_file(file_weight_1_dense, weight_1_dense, DENSE_DIM, 400, true);
    read_file(file_weight_2, weight_2, 400, 400, true);
    read_file(file_weight_3, weight_3, 400, 400, true);
    read_file(file_weight_4_dense, weight_4_dense, 1, 1, true);
    read_file(file_weight_4_sparse, weight_4_sparse, 1, 1, true);
    read_file(file_weight_4, weight_4, 400, 1, true);    

    read_file(file_bias_1, bias_1, 1, 400);
    read_file(file_bias_2, bias_2, 1, 400);
    read_file(file_bias_3, bias_3, 1, 400);
    read_file(file_bias_4, bias_4, 1, 1);

    fill_zero(bias_zero, 400);
    fill_zero(output_1, BATCH * 400);
    fill_zero(output_2, BATCH * 400);
    fill_zero(output_3, BATCH * 400);
    fill_zero(output_result, TOTAL);
}

void write_result()
{
    
    #ifndef CPU
    polaris_memcpy(ctxt, POLARIS_DEVICE_TO_HOST, output_result, polars_output_result, TOTAL * sizeof(float));
    #endif
    
    ofstream out;
    time_t currtime = time(NULL);
	tm* p = localtime(&currtime);
	char filename[100] = {0};
 
    sprintf(filename,"%s_%d%02d%02d%02d%02d%02d.txt",file_output_result, p->tm_year+1900,p->tm_mon+1,p->tm_mday,p->tm_hour,p->tm_min,p->tm_sec);

    out.open(filename, ios::trunc);
    for(int i = 0; i < TOTAL; i++)
    {
        out << output_result[i] <<endl;
    }
    out.close();
}


void polaris_memcpy_self(PolarisContext* ctxt, PolarisMemcpyKind kind,
                             void* dest, const void* src, size_t size)

{
    float* ptr1 =  (float* )dest;
    float* src1 =  (float* )src;
    int size_m = 100;
    int n = size / size_m;
    for(int i =0 ;i < n ;i ++)
    {
    polaris_memcpy_self(ctxt, kind, ptr1, src1, size_m);
 
       ptr1+=size_m;
       src1+=size_m;
    }

}

void init()
{
        #ifndef CPU   
    printf("polaris_malloc\n");
    ctxt = polaris_create_context(0);


    /*************************************************
    * Polaris
    *************************************************/
    /************* EMBED *****************/
    

    /*************************************************
    * Polaris
    *************************************************/
    /************* EMBED ****************/
    polaris_malloc(ctxt, TOTAL * SPARSE_DIM * sizeof(float), (void **)&polars_sparse_embed);
    polaris_malloc(ctxt, TOTAL * DENSE_DIM * sizeof(float), (void **)&polars_dense_input);

    /*************  FM  ****************/
    polaris_malloc(ctxt, TOTAL * sizeof(float), (void **)&polars_dense_fm);
    polaris_malloc(ctxt, TOTAL * sizeof(float), (void **)&polars_sparse_fm);

    /*************  weight  ****************/
    polaris_malloc(ctxt, SPARSE_DIM * 400 * sizeof(float), (void **)&polars_weight_1_sparse);
    polaris_malloc(ctxt, DENSE_DIM * 400 * sizeof(float), (void **)&polars_weight_1_dense);
    polaris_malloc(ctxt, 400 * 400 * sizeof(float), (void **)&polars_weight_2);
    polaris_malloc(ctxt, 400 * 400 * sizeof(float), (void **)&polars_weight_3);
    polaris_malloc(ctxt, 1 * sizeof(float), (void **)&polars_weight_4_dense);
    polaris_malloc(ctxt, 1 * sizeof(float), (void **)&polars_weight_4_sparse);
    polaris_malloc(ctxt, 400 * sizeof(float), (void **)&polars_weight_4);

    /*************  BIAS  *****************/
    polaris_malloc(ctxt, 400 * sizeof(float), (void **)&polars_bias_1);
    polaris_malloc(ctxt, 400 * sizeof(float), (void **)&polars_bias_2);
    polaris_malloc(ctxt, 400 * sizeof(float), (void **)&polars_bias_3);
    polaris_malloc(ctxt, 1 * sizeof(float), (void **)&polars_bias_4);

    /*************  BIAS ZERO *****************/
    polaris_malloc(ctxt, 400 * sizeof(float), (void **)&polars_bias_zero);

    /*************  OUTPUT *****************/
    polaris_malloc(ctxt, BATCH * 400 * sizeof(float), (void **)&polars_output_1);
    polaris_malloc(ctxt, BATCH * 400 * sizeof(float), (void **)&polars_output_2);
    polaris_malloc(ctxt, BATCH * 400 * sizeof(float), (void **)&polars_output_3);
    polaris_malloc(ctxt, TOTAL * sizeof(float), (void **)&polars_output_result);
    /*************************************************
    * Polaris END
    *************************************************/
        #endif


        #ifndef CPU   
    printf("polaris_memcpy\n");
    /*************************************************
    * Polaris
    *************************************************/
    /************* EMBED *****************/
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_sparse_embed, sparse_embed, TOTAL * SPARSE_DIM * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_dense_input, dense_input, TOTAL * DENSE_DIM * sizeof(float));

    /*************  FM  *****************/
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_dense_fm, dense_fm, TOTAL * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_sparse_fm, sparse_fm, TOTAL * sizeof(float));

    /*************  weight  *****************/
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_weight_1_sparse, weight_1_sparse, SPARSE_DIM * 400 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_weight_1_dense, weight_1_dense, DENSE_DIM * 400 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_weight_2, weight_2, 400 * 400 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_weight_3, weight_3, 400 * 400 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_weight_4_dense, weight_4_dense, 1 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_weight_4_sparse, weight_4_sparse, 1 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_weight_4, weight_4, 400 * sizeof(float));

    /*************  BIAS  *****************/
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_bias_1, bias_1, 400 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_bias_2, bias_2, 400 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_bias_3, bias_3, 400 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_bias_4, bias_4, 1 * sizeof(float));

    /*************  BIAS ZERO *****************/
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_bias_zero, bias_zero, 400 * sizeof(float));

    /*************  OUTPUT *****************/
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_output_1, output_1, BATCH * 400 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_output_2, output_2, BATCH * 400 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_output_3, output_3, BATCH * 400 * sizeof(float));
    polaris_memcpy(ctxt, POLARIS_HOST_TO_DEVICE, polars_output_result, output_result, TOTAL * sizeof(float));
        #endif
    /*************************************************
    * Polaris END
    *************************************************/
}

void free_polaris()
{
    /*************************************************
    * Polaris
    *************************************************/
   
        #ifndef CPU   
    /************* EMBED *****************/
    polaris_free(ctxt, polars_sparse_embed);
    polaris_free(ctxt, polars_dense_input);

    /*************  FM  *****************/
    polaris_free(ctxt, polars_dense_fm);
    polaris_free(ctxt, polars_sparse_fm);

    /*************  weight  *****************/
    polaris_free(ctxt, polars_weight_1_sparse);
    polaris_free(ctxt, polars_weight_1_dense);
    polaris_free(ctxt, polars_weight_2);
    polaris_free(ctxt, polars_weight_3);
    polaris_free(ctxt, polars_weight_4_dense);
    polaris_free(ctxt, polars_weight_4_sparse);
    polaris_free(ctxt, polars_weight_4);

    /*************  BIAS  *****************/
    polaris_free(ctxt, polars_bias_1);
    polaris_free(ctxt, polars_bias_2);
    polaris_free(ctxt, polars_bias_3);
    polaris_free(ctxt, polars_bias_4);

    /*************  BIAS ZERO *****************/
    polaris_free(ctxt, polars_bias_zero);

    /*************  OUTPUT *****************/
    polaris_free(ctxt, polars_output_1);
    polaris_free(ctxt, polars_output_2);
    polaris_free(ctxt, polars_output_3);
    polaris_free(ctxt, polars_output_result);
    /*************************************************
    * Polaris END
    *************************************************/

    polaris_destroy_context(ctxt);
        #endif
}

int main()
{
    
    struct timeval time1, time2, time3, time4;
    
    printf("read_data\n");
    read_data();
    //for(int i = 0; i< 10; i++)
    {
    gettimeofday(&time3, NULL);
    printf("init\n");
    init();

    gettimeofday(&time4, NULL);
    long int deltaT2 = (time4.tv_sec-time3.tv_sec) * 1000000 + time4.tv_usec-time3.tv_usec;
    printf("IO  : %lf\n", double(deltaT2) / 1000000);
    }
    //for(int i = 0; i< 10; i++)
    {
    printf("four_fc_layer\n");
    gettimeofday(&time1, NULL);
    four_fc_layer();
    gettimeofday(&time2, NULL);
    long int deltaT1 = (time2.tv_sec-time1.tv_sec) * 1000000 + time2.tv_usec-time1.tv_usec;
    printf("polaris  : %lf\n", double(deltaT1) / 1000000);

    }

    printf("rite_result\n");
    write_result();

    printf("free_polaris\n");
    free_polaris();
}