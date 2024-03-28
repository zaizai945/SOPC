#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "SOPC_HW1.h"
// #include "SOPC_HW1_new2.h"
// #include "sopc_hw1_new.h"

int size;

// Aim to check the answer
void print_matrix(float **mat, int r, int w)
{
    int i, j;
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < w; j++)
            printf("%f ", mat[i][j]);
        printf("\n");
    }
}

// a: MxW
// b: WxN
float **matrix_mul(float **a, float **b, int m, int w, int n)
{
    int i, j, k;

    float **result = (float **)malloc(m * sizeof(float *));
    for (i = 0; i < m; i++)
        result[i] = (float *)malloc(n * sizeof(float));

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            result[i][j] = 0.0;
            k = 0;
            while (k < w)
            {
                result[i][j] = result[i][j] + a[i][k] * b[k][j];
                k++;
            }
        }
    }

    return result;
}

float **conv2D(float **c, float **d, int l, int k)
{
    int i, j, x, y;
    // trasform weight matrix
    size = l - k + 1;
    int row = size * size;
    int col = k * k;
    float **tc = (float **)malloc(row * sizeof(float *));
    for (i = 0; i < row; i++)
        tc[i] = (float *)malloc(col * sizeof(float));

    for (i = 0, x = 0, y = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
        {
            tc[i][j] = c[x + i / size][y + i % size];
            if (y == (k - 1))
            {
                if (x == (k - 1))
                {
                    x = 0;
                    y = 0;
                }
                else
                {
                    x++;
                    y = 0;
                }
            }
            else
                y++;
        }
    }

    // transform kernel matrix
    float **d_1d = (float **)malloc(col * sizeof(float *));
    for (i = 0; i < col; i++)
        d_1d[i] = (float *)malloc(1 * sizeof(float));

    for (i = 0, x = 0, y = 0; i < col; i++)
    {
        d_1d[i][0] = d[x][y];
        if (y == (k - 1))
        {
            x++;
            y = 0;
        }
        else
            y++;
    }

    // mat_mul
    float **ans_1d = matrix_mul(tc, d_1d, row, col, 1);

    // transform 1d matrix to 2d
    float **conv_ans = (float **)malloc(size * sizeof(float *));
    for (i = 0; i < size; i++)
        conv_ans[i] = (float *)malloc(size * sizeof(float));

    for (i = 0, x = 0, y = 0; i < row; i++)
    {
        conv_ans[x][y] = ans_1d[i][0];
        if (y == (size - 1))
        {
            x++;
            y = 0;
        }
        else
            y++;
    }

    // free the useless memory
    return conv_ans;
}

// attention(IN,Wq,Wk,Wv)
float **attention(float **in, float **wq, float **wk, float **wv)
{
    int i, j;
    float max, sum;
    /*
    Q = In @ Wq
    K = In @ Wk
    V = In @ wv
    */
    float **q = matrix_mul(in, wq, S1, S2, S3);
    float **k = matrix_mul(in, wk, S1, S2, S3);
    float **v = matrix_mul(in, wv, S1, S2, S4);

    // K transpose
    float **kt = (float **)malloc(S3 * sizeof(float *));
    for (i = 0; i < S3; i++)
        kt[i] = (float *)malloc(S1 * sizeof(float));

    for (i = 0; i < S3; i++)
        for (j = 0; j < S1; j++)
            kt[i][j] = k[j][i];

    // alpha = Q @ K.T
    float **alpha = matrix_mul(q, kt, S1, S3, S1);

    // alpha = alpha / sqrt(dk)
    for (i = 0; i < S1; i++)
    {
        for (j = 0; j < S1; j++)
        {
            alpha[i][j] /= sqrt(S3);
        }
    }

    // Sub the max value in the vector to that vector for avoiding overflow
    for (i = 0; i < S1; i++)
    {
        max = alpha[i][0];
        for (j = 1; j < S1; j++)
        {
            if (max < alpha[i][j])
                max = alpha[i][j];
        }
        for (j = 0; j < S1; j++)
            alpha[i][j] -= max;
    }

    // s = softmax(alpha)
    float **nume = (float **)malloc(S1 * sizeof(float *));
    for (i = 0; i < S1; i++)
        nume[i] = (float *)malloc(S1 * sizeof(float));

    float **deno = (float **)malloc(S1 * sizeof(float *));
    for (i = 0; i < S1; i++)
        deno[i] = (float *)malloc(S1 * sizeof(float));

    float **s = (float **)malloc(S1 * sizeof(float *));
    for (i = 0; i < S1; i++)
        s[i] = (float *)malloc(S1 * sizeof(float));

    for (i = 0; i < S1; i++)
    {
        for (j = 0; j < S1; j++)
        {
            nume[i][j] = exp(alpha[i][j]);
            deno[i][j] = exp(alpha[i][j]);
        }
    }

    for (i = 0; i < S1; i++)
    {
        sum = 0.0;
        for (j = 0; j < S1; j++)
            sum += deno[i][j];

        for (j = 0; j < S1; j++)
            deno[i][j] = sum;
    }

    for (i = 0; i < S1; i++)
        for (j = 0; j < S1; j++)
            s[i][j] = nume[i][j] / deno[i][j];

    float **out = matrix_mul(s, v, S1, S1, S4);

    // free memory
    return out;
}

int main()
{
    int i, j;

    // first
    float **a = (float **)malloc(M * sizeof(float *));
    for (i = 0; i < M; i++)
        a[i] = (float *)malloc(W * sizeof(float));
    for (i = 0; i < M; i++)
        for (j = 0; j < W; j++)
            a[i][j] = A[i][j];

    float **b = (float **)malloc(W * sizeof(float *));
    for (i = 0; i < W; i++)
        b[i] = (float *)malloc(N * sizeof(float));
    for (i = 0; i < W; i++)
        for (j = 0; j < N; j++)
            b[i][j] = B[i][j];

    float **ans1 = matrix_mul(a, b, M, W, N);

    // second
    float **c = (float **)malloc(L * sizeof(float *));
    for (i = 0; i < L; i++)
        c[i] = (float *)malloc(L * sizeof(float));
    for (i = 0; i < L; i++)
        for (j = 0; j < L; j++)
            c[i][j] = C[i][j];

    float **d = (float **)malloc(K * sizeof(float *));
    for (i = 0; i < K; i++)
        d[i] = (float *)malloc(K * sizeof(float));
    for (i = 0; i < K; i++)
        for (j = 0; j < K; j++)
            d[i][j] = D[i][j];

    float **ans2 = conv2D(c, d, L, K);

    // third
    float **in = (float **)malloc(S1 * sizeof(float *));
    for (i = 0; i < S1; i++)
        in[i] = (float *)malloc(S2 * sizeof(float));

    float **wq = (float **)malloc(S2 * sizeof(float *));
    for (i = 0; i < S2; i++)
        wq[i] = (float *)malloc(S3 * sizeof(float));

    float **wk = (float **)malloc(S2 * sizeof(float *));
    for (i = 0; i < S2; i++)
        wk[i] = (float *)malloc(S3 * sizeof(float));

    float **wv = (float **)malloc(S2 * sizeof(float *));
    for (i = 0; i < S2; i++)
        wv[i] = (float *)malloc(S4 * sizeof(float));

    for (i = 0; i < S1; i++)
        for (j = 0; j < S2; j++)
            in[i][j] = IN[i][j];

    for (i = 0; i < S2; i++)
    {
        for (j = 0; j < S3; j++)
        {
            wq[i][j] = Wq[i][j];
            wk[i][j] = Wk[i][j];
        }
    }

    for (i = 0; i < S2; i++)
        for (j = 0; j < S4; j++)
            wv[i][j] = Wv[i][j];

    float **ans3 = attention(in, wq, wk, wv);

    // write the answer
    FILE *fptr;
    fptr = fopen("Output.txt", "w");

    fprintf(fptr, "1.Matrix_mul\n");
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            fprintf(fptr, "%f ", ans1[i][j]);
        fprintf(fptr, "\n");
    }

    fprintf(fptr, "\n2.conv2D\n");
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
            fprintf(fptr, "%f ", ans2[i][j]);
        fprintf(fptr, "\n");
    }

    fprintf(fptr, "\n3.attention\n");
    for (i = 0; i < S1; i++)
    {
        for (j = 0; j < S4; j++)
            fprintf(fptr, "%f ", ans3[i][j]);
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    // free memory
    return 0;
}