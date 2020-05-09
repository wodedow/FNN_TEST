#ifndef STRUCT_TEST_H
#define STRUCT_TEST_H
struct list {
	double* list_elem;
	int list_elem_size;
};

struct List {
	list* List_elem;
	int list_size;
};

struct list2 {
	double** list2_elem;
	int cols;
	int rows;
};

struct List2 {
	list2* List2_elem;
	int list2_size;
};

void InitList_elem(list& List_elem, int initListElemSize) {
	//��ʼ�����Ա�
	List_elem.list_elem = new double[initListElemSize];
	List_elem.list_elem_size = initListElemSize;
}

void InitList(List& LList, int initListSize) {
	//��ʼ�����Ա�
	LList.List_elem = new list[initListSize];
	LList.list_size = initListSize;
}

void InitList2_elem(list2& List2_elem, int list2_rows, int list2_cols) {
	//��ʼ�����Ա�
	List2_elem.list2_elem = new double* [list2_rows];
	for (int i = 0; i < list2_rows; i++) {
		List2_elem.list2_elem[i] = new double[list2_cols];
	}
	List2_elem.cols = list2_cols;
	List2_elem.rows = list2_rows;
}

void InitList2(List2& LList2, int List2_Size) {
	//��ʼ�����Ա�
	LList2.List2_elem = new list2[List2_Size];
	LList2.list2_size = List2_Size;
}

/*************************************************************************************/
struct FNN {
	bool first;
	int length;
	int batch_size;
	int* nodes;
	double loss;						//��ʧֵ $ E_{z}=\sum_{p^m=0}\frac{1}{2}\sum_{l=0}(t_{l}^{L,p^m}-x_{l}^{L,p^m})^2 $
	//char regulation[] = "L2";				//Ĭ��ȡL2����
	List bias;								//ƫ�ã�$ b $
	List layers_in;						//�����������
	List layers;							//����������루����ʼֵ��
	List2 weight_arrays;					//Ȩ�ؾ���$ W $
};
void Init_Network_FNN(FNN& Net, int L, int* m) {
	Net.length = L + 1;
	Net.nodes = m;
	InitList(Net.bias, L);
	InitList(Net.layers, L + 1);
	InitList2(Net.weight_arrays, L);
	InitList(Net.layers_in, L);
	//��ʼ�����ݽṹ��
	for (int i = 1; i <= L + 1; i++) {
		InitList_elem(Net.layers.List_elem[i - 1],m[i - 1]);
		if (i == L + 1) {
			break;
		}
		InitList2_elem(Net.weight_arrays.List2_elem[i - 1], m[i], m[i - 1]);
		InitList_elem(Net.layers_in.List_elem[i - 1], m[i]);
		InitList_elem(Net.bias.List_elem[i - 1], m[i]);
	}
}

#pragma once
#endif