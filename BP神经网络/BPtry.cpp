
//designed by zzw --2019.1.21
#include "pch.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>//用来读取数据集
#include <sstream>//使用stringstream
#include <string>
#include<math.h>//sigmoid函数中exp
#include <vector>
#include<algorithm>
using namespace std;
#define TIMES    1000     //最大训练次数
struct flower {
	float index1;
	float index2;
	float index3;
	float index4;
	float flags1[3] = { 0.0,0.0,0.0 };
	int flag;//记录最大值为分类
};

//将str转换为float
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
flower *flower_data=new flower[150];//每组50个,共三组
int num_max = 0;//总共的数据个数
int num_all = 150;//数组分配的最大个数
const int adder = 50;//每次动态分配50个

//将读入的花朵类型转为对应的100，010，001标识
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
		cout << "不存在这个类型" << endl;
		exit(1);
		return ;
	}
}
void read()
{
	// 读文件
	//ifstream inFile("D:\\我爱学习\\学校 各种\\各种课程\\大三上\\人工智能\\大作业\\BP神经网络\\iris.csv", ios::in);
	ifstream inFile("./iris.csv", ios::in);
	string lineStr;
	vector<vector<string>> strArray; 
	int index = 0;//用来记录数据的下标
	int num = 0;
	while (getline(inFile, lineStr))
	{
		//cout << num++ << endl;
		// 打印整行字符串
		//cout << lineStr << endl;
		// 存成二维表结构
		stringstream ss(lineStr);
		string str;
		vector<string> lineArray;
		
		// 按照逗号分隔
		int flag = 4;//四种数据
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
		if (index > num_all)//扩容动态分配空间
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
//读取csv参考链接 https ://blog.csdn.net/u012234115/article/details/64465398 

}

//定义训练用的数据
float data1[2] = { 1,1 };
float data2[2] = { -1,-1 };
//定义数据的标准分类
float data1_class = 1;
//假设data1的类型为
float data2_class = 0;
//假设data2的类型为0

/*第一层*/
//定义权重和偏置 前为后下标
float w[3][4] = { { 0,0,0,0 },{ 0,0,0,0 },{ 0,0,0,0 } };
//这里用任意值初始化即可，训练的目的就是自动调整这个值的大小
float b[3] = { 0,0,0 };
float sum[10000][3];                //存放加权求和的值
float ypxl[5];
float yy[5];

/*第二层*/
//定义权重和偏置
float w2[3][3] = { { 0,0,0 },{ 0,0,0 },{ 0,0,0 } };
//这里用任意值初始化即可，训练的目的就是自动调整这个值的大小
float b2[3] = { 0,0,0 };
float sum2[10000][3];                //存放加权求和的值
float ypxl2[4];
float yy2[4];

//输入层到隐藏层加权求和
float sumfun(flower *data, float *weight, float bias, int index_data)
{
	float result;
	result = flower_data[index_data].index1*weight[0] +
		flower_data[index_data].index2*weight[1] +
		flower_data[index_data].index3*weight[2] +
		flower_data[index_data].index4*weight[3] + bias;
	return result;
}

//隐藏层到输出层
float sumfun2(float *data, float *weight, float bias, int index_data)
{
	float result;
	result = data[0]*weight[0] +
		data[1]*weight[1] +
		data[2]*weight[2] + bias;
	return result;
}
//神经元取值函数sigmoid
float sigmoid(float sum)
{
	sum = 1.0 / (1.0 + exp(-sum));
	return sum;
}

//sigmoid求导
float sigmoid_dao(float sum)
{
	sum = sum*(1.0-sum);
	return sum;
}

//求整体误差J
float J(float a, float b)
{
	return 0.5*(a - b)*(a - b);
}

//计算反向误差因子
float yp(float e,float y)
{
	return (e - y)*sigmoid_dao(y);
}


float E[10000];                  //存放每次的误差
//float E_ALL;					 //总误差
float study_s = 0.5;			//学习效率

float minNum, maxNum;//最大值最小值，归一化使用

//归一化公式:(x-min)/(max-min)
float cal_data(float a)
{
	return (a - minNum) / (maxNum - minNum);
}

//数据归一化处理
 void changeData()
 {
	     //归一化公式:(x-min)/(max-min)
		 
	     int i, j;
	     minNum = 1000.0;
	     maxNum = -100.0;
	     //找最大最小值
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
		     
	     //归一化
		     for (i = 0; i < num_max; i++)
		     {
				 flower_data[i].index1 = cal_data(flower_data[i].index1);
				 flower_data[i].index2 = cal_data(flower_data[i].index2);
				 flower_data[i].index3 = cal_data(flower_data[i].index3);
				 flower_data[i].index4 = cal_data(flower_data[i].index4);
		     }
	 }

 //寻找最大值的下标
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
	//float output1 = 0, output2 = 0;  //把加权求和的值代入激活函数而得到的输出值
	//int count = 0;                 //训练次数的计数变量
	float err = 0;                //计算的误差，用于对权值和偏置的修正
	//int flag1 = 0, flag2 = 0;         //训练完成的标志，如果某组数据训练结果达标，则把标志置1，否则置0
	read();//读取数据
	changeData();//归一化处理
	int times = TIMES;//训练次数的计数变量
	while (times--)//最多迭代1000次
	{
		cout << "第" << TIMES - times << "次迭代" << endl;
		//int flag_s = 1;//初始化为1，判断是否全部都合格，若是又一次不符合要求，则不行
			for (int index_data = 0; index_data < num_max; index_data++)
			{
				/*隐藏层计算*/
				for (int outer = 0; outer < 3; outer++) {
					sum[index_data][outer] = sumfun(flower_data, w[outer], b[outer], index_data);  //代入第一组data进行计算
					//output1 = step(sum);
					sum[index_data][outer] = sigmoid(sum[index_data][outer]);
				}

				/*输出层计算*/
				for (int out = 0; out < 3; out++) {
					sum2[index_data][out] = sumfun2(sum[index_data], w2[out], b2[out], index_data);  //代入隐藏层data进行计算
					//output1 = step(sum);
					sum2[index_data][out] = sigmoid(sum2[index_data][out]);
				}
				//if (sum[index_data] > sigmoid(ranger[flower_data[index_data].flags]) && sum[index_data] <= sigmoid(ranger[flower_data[index_data].flags + 1]))//判断输出是否达标，若达标则把标志置1，否则修正权值和偏置{
				//{
				//	if (flag_s == 1)
				//	{
				//		flag_s = 1;
				//	}
				//}

				//else//权值和偏移值修改
				//{
					//E[index_data]= J(sigmoid(flower_data[index_data].flags), sum[index_data]);

				/*输出层返回计算偏差*/ //已修改study_s的+ -
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

				/*隐藏层返回计算偏差*/
				for (int out = 0; out < 3; out++) {

					ypxl[out] = 0.0;
					for (int j = 0; j < 3; j++)
					{
						ypxl[out] = ypxl[out] + ypxl2[j] * w2[j][out];//已修改 j out顺序
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

				/*计算各层节点阈值*/
				//输出层
				for (int x = 0; x < 3; x++) {
					b2[x] += study_s * ypxl2[x];
				}
				//隐藏层
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
	float totle = 0;//存储正确个数
	float inner[3];
	float outer[3];
	//用结果进行判断 计算出三个值，最大的为所对应的值
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
					以下部分已用sumfun函数替代
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
		if (x == flower_data[index_num].flag)//如果判断符合结果，则符合的个数+1
		{
			totle += 1.0;
		}
		//cout << endl;
	}
		//cout << endl;
		//cout << endl;
	
	cout << endl;
	cout << "分类正确率： " << totle / num_max * 100.0 << "%" << endl;
	printf("\n\nThe traning done!\n\n");    
	
	/*for (int index_data = 0; index_data < num_max; index_data++)
	{
		cout << flower_data[index_data].flag << " " << flower_data[index_data].flags1[0] << flower_data[index_data].flags1[1] << flower_data[index_data].flags1[2] << endl;

	}*/
	system("pause");
	return 0;
}