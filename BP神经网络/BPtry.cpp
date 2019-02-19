
//designed by zzw --2019.1.21
#include "pch.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>//������ȡ���ݼ�
#include <sstream>//ʹ��stringstream
#include <string>
#include<math.h>//sigmoid������exp
#include <vector>
#include<algorithm>
using namespace std;
#define TIMES    1000     //���ѵ������
struct flower {
	float index1;
	float index2;
	float index3;
	float index4;
	float flags1[3] = { 0.0,0.0,0.0 };
	int flag;//��¼���ֵΪ����
};

//��strת��Ϊfloat
template <class Type>
Type stringToNum(string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}

//char flowertype[3][100] = { "Iris-setosa","Iris-versicolor","Iris-virginica" };
string setosa = "Iris-setosa";
string versicolor = "Iris-versicolor";
string virginica = "Iris-virginica";
flower *flower_data=new flower[150];//ÿ��50��,������
int num_max = 0;//�ܹ������ݸ���
int num_all = 150;//��������������
const int adder = 50;//ÿ�ζ�̬����50��

//������Ļ�������תΪ��Ӧ��100��010��001��ʶ
void find_str(flower *s,int index,string str)
{
	if (str == setosa)
	{
		s[index].flags1[0] = 1.0;
		s[index].flag = 0;
		return ;
	}
	if (str == versicolor)
	{
		s[index].flags1[1] = 1.0;
		s[index].flag = 1;
		return ;
	}
	if (str == virginica)
	{
		s[index].flags1[2] = 1.0;
		s[index].flag = 2;
		return ;
	}
	else
	{
		cout << "�������������" << endl;
		exit(1);
		return ;
	}
}
void read()
{
	// ���ļ�
	//ifstream inFile("D:\\�Ұ�ѧϰ\\ѧУ ����\\���ֿγ�\\������\\�˹�����\\����ҵ\\BP������\\iris.csv", ios::in);
	ifstream inFile("./iris.csv", ios::in);
	string lineStr;
	vector<vector<string>> strArray; 
	int index = 0;//������¼���ݵ��±�
	int num = 0;
	while (getline(inFile, lineStr))
	{
		//cout << num++ << endl;
		// ��ӡ�����ַ���
		//cout << lineStr << endl;
		// ��ɶ�ά��ṹ
		stringstream ss(lineStr);
		string str;
		vector<string> lineArray;
		
		// ���ն��ŷָ�
		int flag = 4;//��������
		while (getline(ss, str, ','))
		{
			
			lineArray.push_back(str);
			switch (flag)  {
			case 4: {
				flower_data[index].index1 = stringToNum<float>(str);
				break;
			}
			case 3: {
				flower_data[index].index2 = stringToNum<float>(str);
				break;
			}
			case 2: {
				flower_data[index].index3 = stringToNum<float>(str);
				break;
			}
			case 1: {
				flower_data[index].index4 = stringToNum<float>(str);
				break;
			}
			case 0: {
				find_str(flower_data,index,str);
				break;
			}
			}
			//cout << str <<" ";
			/*if (flag == 0)
			{
				cout << endl;
			}*/
			flag--;
		}
		num_max++;
		index++;
		if (index > num_all)//���ݶ�̬����ռ�
		{
			
			flower *data_new = new flower[num_all+adder];
			memcpy(data_new, flower_data, num_all);
			delete[]flower_data;
			num_all += adder;
			flower_data = data_new;
		}
		strArray.push_back(lineArray);
	}
	//getchar();
	return ;
//��ȡcsv�ο����� https ://blog.csdn.net/u012234115/article/details/64465398 

}

//����ѵ���õ�����
float data1[2] = { 1,1 };
float data2[2] = { -1,-1 };
//�������ݵı�׼����
float data1_class = 1;
//����data1������Ϊ
float data2_class = 0;
//����data2������Ϊ0

/*��һ��*/
//����Ȩ�غ�ƫ�� ǰΪ���±�
float w[3][4] = { { 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 } };
//����������ֵ��ʼ�����ɣ�ѵ����Ŀ�ľ����Զ��������ֵ�Ĵ�С
float b[3] = { 0,0,0 };
float sum[10000][3];                //��ż�Ȩ��͵�ֵ
float ypxl[5];
float yy[5];

/*�ڶ���*/
//����Ȩ�غ�ƫ��
float w2[3][3] = { { 0,0,0 },{ 0,0,0 },{ 0,0,0 } };
//����������ֵ��ʼ�����ɣ�ѵ����Ŀ�ľ����Զ��������ֵ�Ĵ�С
float b2[3] = { 0,0,0 };
float sum2[10000][3];                //��ż�Ȩ��͵�ֵ
float ypxl2[4];
float yy2[4];

//����㵽���ز��Ȩ���
float sumfun(flower *data, float *weight, float bias, int index_data)
{
	float result;
	result = flower_data[index_data].index1*weight[0] +
		flower_data[index_data].index2*weight[1] +
		flower_data[index_data].index3*weight[2] +
		flower_data[index_data].index4*weight[3] + bias;
	return result;
}

//���ز㵽�����
float sumfun2(float *data, float *weight, float bias, int index_data)
{
	float result;
	result = data[0]*weight[0] +
		data[1]*weight[1] +
		data[2]*weight[2] + bias;
	return result;
}
//��Ԫȡֵ����sigmoid
float sigmoid(float sum)
{
	sum = 1.0 / (1.0 + exp(-sum));
	return sum;
}

//sigmoid��
float sigmoid_dao(float sum)
{
	sum = sum*(1.0-sum);
	return sum;
}

//���������J
float J(float a, float b)
{
	return 0.5*(a - b)*(a - b);
}

//���㷴���������
float yp(float e,float y)
{
	return (e - y)*sigmoid_dao(y);
}


float E[10000];                  //���ÿ�ε����
//float E_ALL;					 //�����
float study_s = 0.5;			//ѧϰЧ��

float minNum, maxNum;//���ֵ��Сֵ����һ��ʹ��

//��һ����ʽ:(x-min)/(max-min)
float cal_data(float a)
{
	return (a - minNum) / (maxNum - minNum);
}

//���ݹ�һ������
 void changeData()
 {
	     //��һ����ʽ:(x-min)/(max-min)
		 
	     int i, j;
	     minNum = 1000.0;
	     maxNum = -100.0;
	     //�������Сֵ
		 for (j = 0; j < num_max; j++)
			{
			 minNum = min(minNum, flower_data[j].index1);
			 maxNum = max(maxNum, flower_data[j].index1);
			 minNum = min(minNum, flower_data[j].index2);
			 maxNum = max(maxNum, flower_data[j].index2);
			 minNum = min(minNum, flower_data[j].index3);
			 maxNum = max(maxNum, flower_data[j].index3);
			 minNum = min(minNum, flower_data[j].index4);
			 maxNum = max(maxNum, flower_data[j].index4);
			}
		     
	     //��һ��
		     for (i = 0; i < num_max; i++)
		     {
				 flower_data[i].index1 = cal_data(flower_data[i].index1);
				 flower_data[i].index2 = cal_data(flower_data[i].index2);
				 flower_data[i].index3 = cal_data(flower_data[i].index3);
				 flower_data[i].index4 = cal_data(flower_data[i].index4);
		     }
	 }

 //Ѱ�����ֵ���±�
 int max_index(float *a)
 {
	 int index;
	 float x=-1.0;
	 for (int i = 0; i < 3; i++)
	 {
		 if (x < a[i]) {
			 x = a[i];
			 index = i;
		 }
	 }
	 return index;
 }

int main() {
	//memset(E, 0.0, sizeof(E));
	//float output1 = 0, output2 = 0;  //�Ѽ�Ȩ��͵�ֵ���뼤������õ������ֵ
	//int count = 0;                 //ѵ�������ļ�������
	float err = 0;                //����������ڶ�Ȩֵ��ƫ�õ�����
	//int flag1 = 0, flag2 = 0;         //ѵ����ɵı�־�����ĳ������ѵ�������꣬��ѱ�־��1��������0
	read();//��ȡ����
	changeData();//��һ������
	int times = TIMES;//ѵ�������ļ�������
	while (times--)//������1000��
	{
		cout << "��" << TIMES - times << "�ε���" << endl;
		//int flag_s = 1;//��ʼ��Ϊ1���ж��Ƿ�ȫ�����ϸ�������һ�β�����Ҫ������
			for (int index_data = 0; index_data < num_max; index_data++)
			{
				/*���ز����*/
				for (int outer = 0; outer < 3; outer++) {
					sum[index_data][outer] = sumfun(flower_data, w[outer], b[outer], index_data);  //�����һ��data���м���
					//output1 = step(sum);
					sum[index_data][outer] = sigmoid(sum[index_data][outer]);
				}

				/*��������*/
				for (int out = 0; out < 3; out++) {
					sum2[index_data][out] = sumfun2(sum[index_data], w2[out], b2[out], index_data);  //�������ز�data���м���
					//output1 = step(sum);
					sum2[index_data][out] = sigmoid(sum2[index_data][out]);
				}
				//if (sum[index_data] > sigmoid(ranger[flower_data[index_data].flags]) && sum[index_data] <= sigmoid(ranger[flower_data[index_data].flags + 1]))//�ж�����Ƿ��꣬�������ѱ�־��1����������Ȩֵ��ƫ��{
				//{
				//	if (flag_s == 1)
				//	{
				//		flag_s = 1;
				//	}
				//}

				//else//Ȩֵ��ƫ��ֵ�޸�
				//{
					//E[index_data]= J(sigmoid(flower_data[index_data].flags), sum[index_data]);

				/*����㷵�ؼ���ƫ��*/ //���޸�study_s��+ -
				for (int out = 0; out < 3; out++) {
					ypxl2[out] = yp(flower_data[index_data].flags1[out], sum2[index_data][out]);//
					//yy2[out] = sum2[index_data][out];
					for (int x = 0; x < 3; x++)
					{
						float wj = study_s * ypxl2[out]*sum[index_data][x];
						w2[out][x] += wj;
						//b2[out] = b[out] - study_s * ypxl2[out];
					}
				}

				/*���ز㷵�ؼ���ƫ��*/
				for (int out = 0; out < 3; out++) {

					ypxl[out] = 0.0;
					for (int j = 0; j < 3; j++)
					{
						ypxl[out] = ypxl[out] + ypxl2[j] * w2[j][out];//���޸� j out˳��
					}
					ypxl[out] = ypxl[out] * sigmoid_dao(sum[index_data][out]);

					float wj = study_s * ypxl[out] * flower_data[index_data].index2;
					w[out][1] += wj;

					wj = study_s * ypxl[out] * flower_data[index_data].index3;
					w[out][2] += wj;

					wj = study_s * ypxl[out] * flower_data[index_data].index4;
					w[out][3] += wj;

					wj = study_s * ypxl[out] * flower_data[index_data].index1;
					w[out][0] += wj;

					//b[out] = b[out] - study_s * ypxl[out];
					
				}

				/*�������ڵ���ֵ*/
				//�����
				for (int x = 0; x < 3; x++) {
					b2[x] += study_s * ypxl2[x];
				}
				//���ز�
				for (int x = 0; x < 3; x++) {
					b[x] += study_s * ypxl[x];
				}
				/*flag_s = 0;
				err = J(sigmoid(flower_data[index_data].flags) , sum[index_data]);
				w[0] = w[0] + err * sigmoid(flower_data[index_data].index1);
				w[1] = w[1] + err * sigmoid(flower_data[index_data].index2);
				w[2] = w[2] + err * sigmoid(flower_data[index_data].index3);
				w[3] = w[3] + err * sigmoid(flower_data[index_data].index4);
				b = b + err;*/
				//}
			}
		
		/*if (flag_s)
		{
			break;
		}*/
		
	}
	float totle = 0;//�洢��ȷ����
	float inner[3];
	float outer[3];
	//�ý�������ж� ���������ֵ������Ϊ����Ӧ��ֵ
	for (int index_num = 0; index_num < num_max; index_num++)
	{
		memset(inner, 0, sizeof(inner));
		memset(outer, 0, sizeof(outer));
		for (int index = 0; index < 3; index++)
		{
			for (int index_2 = 0; index_2 < 3; index_2++)
			{
				inner[index_2] = sigmoid(sumfun(flower_data, w[index_2], b[index_2], index_num));
					/*
					���²�������sumfun�������
					outer[index_2] + w[0][index_2] * flower_data[index].index1;
				outer[index_2] = outer[index_2] + w[1][index_2] * flower_data[index].index2;
				outer[index_2] = outer[index_2] + w[2][index_2] * flower_data[index].index3;
				outer[index_2] = outer[index_2] + w[3][index_2] * flower_data[index].index4;
				*/
				
			}
			outer[index] = sigmoid(sumfun2(inner, w2[index], b2[index], index_num));
			//cout << outer[index] << " ";
		}
		int x = max_index(outer);
		if (x == flower_data[index_num].flag)//����жϷ��Ͻ��������ϵĸ���+1
		{
			totle += 1.0;
		}
		//cout << endl;
	}
		//cout << endl;
		//cout << endl;
	
	cout << endl;
	cout << "������ȷ�ʣ� " << totle / num_max * 100.0 << "%" << endl;
	printf("\n\nThe traning done!\n\n");    
	
	/*for (int index_data = 0; index_data < num_max; index_data++)
	{
		cout << flower_data[index_data].flag << " " << flower_data[index_data].flags1[0] << flower_data[index_data].flags1[1] << flower_data[index_data].flags1[2] << endl;

	}*/
	system("pause");
	return 0;
}