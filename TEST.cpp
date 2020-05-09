#include <stdio.h>
#include <string.h>
#include <math.h>

#include "Struct_test.h"

using namespace std;

const int img_size = 28 * 28;	//sizes of data
const int classes = 10;		//nums of classes
const int numbers = 5000;

const char add_imgs[] = "D:\\MnistHandWriting\\t10k-images.idx3-ubyte";
const char add_labs[] = "D:\\MnistHandWriting\\t10k-labels.idx1-ubyte";
//const char add_imgs[] = "D:\\MnistHandWriting\\train-images.idx3-ubyte";
//const char add_labs[] = "D:\\MnistHandWriting\\train-labels.idx1-ubyte";

/*************************************************************************************/
int swap(int x)
{
    return (((int)(x) & 0xff000000) >> 24) | \
        (((int)(x) & 0x00ff0000) >> 8) | \
        (((int)(x) & 0x0000ff00) << 8) | \
        (((int)(x) & 0x000000ff) << 24);
}
double* matrix_add(double* a, double* b, int j,int* m) {
    //两矩阵相加
    double* c = new double[m[j]];

    for (int n = 0; n < m[j]; n++) {
        c[n] = a[n] + b[n];
    }
    return c;
}
double* matrix_rot(double** a, double* b, int j, int* m) {
    //两矩阵相乘
    double* c = new double[m[j]];

    for (int n = 0; n < m[j]; n++) {
        c[n] = 0;
        for (int jj = 0; jj < m[j - 1]; jj++) {
            //cout << "a[" << n << "][" << jj << "]: " << a[n][jj] << endl;
            c[n] += a[n][jj] * b[jj];
        }
    }

    return c;
}
double* sigmod_l(double* a, int j,int* m) {
    //Sigmod函数：Logistic $\sigma(x)=\frac{1}{1+exp(-x)}$
    double* c = new double[m[j]];

    for (int n = 0; n < m[j]; n++) {
        c[n] = 1 / (1 + exp(-a[n]));
    }

    return c;
    delete[] c;
}
bool forward(FNN& fnn, int label) {
    //正向传递更新结点输入输出：List2& layers_in, List2& layers，并返回Loss
    int L = fnn.length - 1;
    for (int j = 1; j < L + 1; j++) {
        double* p;
        p = matrix_rot(fnn.weight_arrays.List2_elem[j - 1].list2_elem, fnn.layers.List_elem[j - 1].list_elem, j, fnn.nodes);
        double* q;
        q = matrix_add(p, fnn.bias.List_elem[j - 1].list_elem, j, fnn.nodes);
        fnn.layers_in.List_elem[j - 1].list_elem = q;
        fnn.layers.List_elem[j].list_elem = sigmod_l(q, j, fnn.nodes);
    }

    //for (int i = 0; i < 10; i++) {
    //    printf("%f  ", fnn.layers.List_elem[L].list_elem[i]);
    //}
    //printf("\n");

    double* s = fnn.layers.List_elem[L].list_elem;
    //cout << "Listsize:" << layers_test.List_elem[L].list_elem_size << endl;
    double max = s[0];
    int temp = 0;
    for (int k = 1; k < classes; k++) {
        if (s[k] > max){
            temp = k;
            max = s[k];
        }
    }
    if (temp == label)
        return true;
    return false;
}

bool read_weight_arrays(List2& weight_arrays, int L) {
    FILE* fin_weight;
    errno_t err_weight = 0;
    err_weight = fopen_s(&fin_weight, "..\\TRAIN\\weight_arrays.txt", "r");
    if (err_weight != 0) {
        printf("Can't read the data of weight_arrays!!!\n");
    }

    for (int k = 0; k < L; k++) {
        for (int i = 0; i < weight_arrays.List2_elem[k].rows; i++) {
            for (int j = 0; j < weight_arrays.List2_elem[k].cols; j++) {
                fscanf_s(fin_weight, "%lf", &weight_arrays.List2_elem[k].list2_elem[i][j]);
            }
        }
    }

    fclose(fin_weight);
    return true;
}
bool read_bias(List& bias, int L) {
    FILE* fin_bias;
    errno_t err_bias = 0;
    err_bias = fopen_s(&fin_bias, "..\\TRAIN\\bias.txt", "r");
    if (err_bias != 0) {
        printf("Can't read the data of bias!!!\n");
    }

    for (int k = 0; k < L; k++) {
        for (int i = 0; i < bias.List_elem[k].list_elem_size; i++) {
            fscanf_s(fin_bias, "%lf", &bias.List_elem[k].list_elem[i]);
            //printf("%lf\n", bias.List_elem[k].list_elem[i]);
        }
    }

    fclose(fin_bias);
    return true;
}
float FNNTest(FNN& fnn, int L, int* m) {
    Init_Network_FNN(fnn, L, m);
    read_weight_arrays(fnn.weight_arrays, fnn.length - 1);
    read_bias(fnn.bias, fnn.length - 1);
    int correct = 0;
    bool score = true;
    float accuracy = 0;

    FILE* imgs;
    FILE* labs;
    errno_t err_img = 0; errno_t err_lab = 0;
    err_img = fopen_s(&imgs, add_imgs, "rb");
    err_lab = fopen_s(&labs, add_labs, "rb");
    if (err_img != 0) {
        printf("Can't Open This File: Images!!!\n");
    }
    if (err_img != 0) {
        printf("Can't Open This File: Labels!!!\n");
    }

    int magic;				//文件中的魔术数(magic number)  
    int num_items;			//mnist图像集文件中的图像数目  
    int num_label;			//mnist标签集文件中的标签数目  
    int rows;				//图像的行数  
    int cols;				//图像的列数  

    fread(&magic, sizeof(int), 1, imgs);
    if (swap(magic) != 2051) {
        printf("This isn't the Mnist Images_Test File!!!\n");
    }
    fread(&magic, sizeof(int), 1, labs);
    if (swap(magic) != 2049) {
        printf("This isn't the Mnist Labels_Test File!!!\n");
    }

    fread(&num_items, sizeof(int), 1, imgs);
    fread(&num_label, sizeof(int), 1, labs);
    if (swap(num_items) != swap(num_label)) {
        printf("The Image File and Label File are not a Pair!!!\n");
    }

    fread(&rows, sizeof(int), 1, imgs);
    fread(&cols, sizeof(int), 1, imgs);
    rows = swap(rows); cols = swap(cols);

    int sizes = rows * cols;
    if (sizes != img_size) {
        printf("The Size of Picture is False!!!\n");
    }

    char* pixels_img = new char[sizes];
    char label;
    //int c;
    for (int i = 0; i < numbers; i++) {
        fread(pixels_img, sizeof(char), sizes, imgs);
        fread(&label, sizeof(char), 1, labs);
        //图像数据
        for (int j = 0; j < sizes; j++) {
            double mn = pixels_img[j];
            //cout<< m << "--";
            if (mn == 0) {
                fnn.layers.List_elem[0].list_elem[j] = 0;
                //printf("%d***", (int)fnn.layers.List_elem[0].list_elem[j]);
            }
            else {
                fnn.layers.List_elem[0].list_elem[j] = 1;
                //printf("%d***", (int)fnn.layers.List_elem[0].list_elem[j]);
            }
            //printf("%f***", mn);
        }
        //标签数据
        //printf("++++++++++++++++++++++++++\n");
        int classes_k = (int)label;
        //printf("%d\n", classes_k);
        score=forward(fnn, classes_k);
        if (score) {
            correct++;
        }
    }
    delete[] pixels_img;
    
    fclose(imgs);
    fclose(labs);

    accuracy=(float)correct / numbers;
    return accuracy;
}

int main() {
    //srand((unsigned)time(NULL));
    int L = 3;											    //神经网络的层数
    int m[] = { img_size, 300,50, classes };				//每一层的结点数：m[1:]
    float accuracy = 0;

    FNN Net;
    accuracy=FNNTest(Net,  L, m);
    printf("Accuracy is: %f", accuracy);

    memset(&Net, 0, sizeof(Net));
    return 1;
}