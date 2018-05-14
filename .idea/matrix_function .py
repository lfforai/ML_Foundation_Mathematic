
#矩阵范式与矩阵函数、矩阵偏导与积分的经典教程（中电数据分析师数学基础教程）
#1、 https://wenku.baidu.com/view/0d1dccea5901020206409ca2.html
#2、 https://wenku.baidu.com/view/38092a421611cc7931b765ce050876323112740b.html
#3、 https://wenku.baidu.com/view/936c9c7f7cd184254b3535e8.html
#4、下面代码针对文章1编写，文章2，3编写的更好建议仔细阅读
#5、矩阵范式和矩阵函数、矩阵偏导与积分是微分方程组和随机梯度等机器学习的重要基础数学基础知识
#6、代码作者：罗锋
import numpy as np
import sympy as sp


print("第一部分-------------------------矩阵范数、矩阵冥级数、矩阵函数----------------------")
#6.2--------------矩阵范式-----------------#
#m1,m2,m3型矩阵范式
def m1(a):
    a1=np.array(a)
    weight=a1.shape[0]
    height=a1.shape[1]
    result=0
    for i in  range(weight):
        for j in range(height):
            result=np.abs(a1[i][j])+result
    return result

def m2(a):#平方和法m2范式
    a1=np.array(a)
    weight=a1.shape[0]
    height=a1.shape[1]
    result=0
    for i in  range(weight):
        for j in range(height):
            result=np.power(a1[i][j],2)+result
    return np.power(result,0.5)

def m2_n(a):#轨迹法m2范式
    a1=np.array(a)
    result=np.power(np.trace(np.matmul(np.transpose(a),a1)),0.5)
    return result

def m3(a):
    a1=np.array(a)
    weight=a1.shape[0]
    height=a1.shape[1]
    result=0
    for i in  range(weight):
        for j in range(height):
            if result<np.abs(a[i][j]):
               result=np.abs(a[i][j])
    return result

#------------------矩阵范式------------------#
#1,2,3型矩阵范式
def m2_1(a):#谱范式
    a1=np.array(a)
    a_n,b_n=np.linalg.eig(a)
    return np.max(a_n)

def m1_1(a):#列范式
    a1=np.array(a)
    weight=a1.shape[0]
    height=a1.shape[1]
    result=0
    max=0
    for j in range(height):
        if j!=0:
            for i in  range(weight):
                result=np.abs(a[i][j])+result
            if max<result:
                max=result
        else:
            for i in  range(weight):
                result=np.abs(a[i][j])+result
            max=result
    return max

def m3_1(a):#行范式
    a1=np.array(a)
    weight=a1.shape[0]
    height=a1.shape[1]
    result=0
    max=0
    for i in range(weight):
        if i!=0:
            for j in  range(height):
                result=np.abs(a[i][j])+result
            if max<result:
               max=result
        else:
            for j in  range(height):
                result=np.abs(a[i][j])+result
            max=result
    return max

# a=[[1,2,3],[4,5,6],[7,8,9]]
# b=[[2,3,4],[5,6,7],[8,9,10]]
#
# print(m1_1(np.matmul(a,b)))
# print(m1_1(a)*m1_1(b))
from sympy.abc import r,x,y,t
from sympy import poly
from sympy import *
T=np.mat([[0,0,1],[1,0,1],[1,1,0]]) #矩阵b的过渡矩阵
b=np.mat([[3,-1,1],[2,0,1],[1,-1,2]])

c=np.mat([[1,0,0],
          [0,2,0],
          [0,1,2]])#矩阵b的若当标准形
print("矩阵b：=",b)
print("矩阵b的过渡矩阵:=",T)
E=sp.Matrix([[r,0,0],[0,r,0],[0,0,r]])
b=sp.Matrix([[3,-1,1],[2,0,1],[1,-1,2]])
eig=factor(sp.Matrix(E-b).det())#print("矩阵b的特征多项式：=",eig)
print("由特征多项式计算的若当标准型(r - 2)**2*(r - 1)：=",sp.Matrix(c))
print("由过渡矩阵求得的若当标准型T.I*b*T：=",T.I*b*T)
# print(sp.div(x**2-1,x-1))
#利用sympy求最大公因式,辗转相除法
# def factor_l(a=eig,b=eig):
#     temp,yu=sp.div(a,b)#yu余数
#     while  yu!=0:
#         a=b
#         b=yu
#         temp,yu=sp.div(a,b)#yu余数
#     return b
#
# print("diff(x**3, x, 3):=",diff((x*y)**3, x, 3))
#
# print(factor_l(x**2-1,x-1))
#
# print("剔除重因子：=",factor(sp.div(eig,factor_l(eig,diff(eig)))))
#
# print("eig:=",type(eig))
# print("matrix:+",(b-2*sp.eye(3))*(b-2*sp.eye(3))*(b-sp.eye(3)))
# print(type(factor(sp.Matrix(B-A).det())))
# f=sp.div((r+1)*(r-1)*(r+3),(r**2+2))
#
# print(f[1].evalf(subs={r:3}))
# print(expand((r+1)*(r-1)*(r+3)))
# print(expand((r**2 + r - 3)*(r+2)))




# print(f)
# print("a:=",f.evalf(subs={r:sp.Matrix(A)}))


# c=np.mat([[1,0,0],
#           [0,2,1],
#           [0,0,2]])#若当标准形
# print(T*c*T.I)
# # print(T*c*T.I)
# exit()

# c=np.mat([[1,0,0,0],
#           [0,1,0,0],
#           [0,0,2,0],
#           [0,0,1,2]])#若当标准形

#6.6.矩阵函数-------------------------------------
#方法一：定义法 exp(a)
def exp_n(a,dep):
    a1=np.mat(a)
    result=np.zeros((a1.shape[0],a1.shape[1]))
    for i in range(dep):
        if i==0:
            result=np.eye(a1.shape[0])
        else:
            temp=np.eye(a1.shape[0])
            for j in range(i):
                temp=temp*a1
            temp=temp/np.math.factorial(i)
            result=temp+result
    return result

#方法二：jordan标准形法,func=0:exp(x),1:sin(x),2:cos(x)
def jordan_n(a,T=np.zeros((3,3)),func=lambda x:np.exp(x)):
    #提取若当标准型中不同的若当块
    jordan_matrix=np.mat(a)
    weight=jordan_matrix.shape[0]
    height=jordan_matrix.shape[1]

    #1、获取若当矩阵中每个若当块的位置
    def catch_jordan_index(jordan_matrix=jordan_matrix):

        index=(0,0,0)#记录若当矩一个阵若当块的位置信息（开始位置，结束位置，元素值）
        index_list=[]
        i=0
        while i<height:
            if i==0:
               index=(0,0,np.array(jordan_matrix)[0][0])
               #判断对角线上相同元素
               temp=index[1]+1
               while temp<height:
                     if index[2]==np.array(jordan_matrix)[temp][temp] and np.array(jordan_matrix)[temp][temp-1]!=0:
                        temp=temp+1#沿对角线下移动一格
                     else:
                        break
               index=(0,temp-1,np.array(jordan_matrix)[0][0])
               index_list.append(index)
               i=index[1]+1
            else:
               first_index=index_list[-1][1]+1
               index=(first_index,first_index,np.array(jordan_matrix)[first_index][first_index])
               temp=index[1]+1
               if temp<height:
                  while temp<height:
                          if index[2]==np.array(jordan_matrix)[temp][temp] and np.array(jordan_matrix)[temp][temp-1]!=0:
                             temp=temp+1#沿对角线下移动一格
                          else:
                             break
                  index=(first_index,temp-1,np.array(jordan_matrix)[first_index][first_index])
                  index_list.append(index)
                  i=index[1]+1
               else:
                  index_list.append(index)
                  i=index[1]+1
        return index_list
    index_list=catch_jordan_index(jordan_matrix=jordan_matrix)
    # print(index_list)
    #2.计算每个若当块f（Ai）i=0，1，。。r,r是矩阵A若当块的个数
    #生成mi行为斜角的矩阵，案例中为半矩阵上斜角，本例中为下半矩阵斜角
    def eye(num,row):
        return np.eye(num, k=row)

    jordan_list=[]#存储每个若当块的f(a)
    for e in index_list:
        if e[1]-e[0]+1<2:#仅有1个元素
            jordan_list.append(np.array([func(e[2])]))
        else:
           length_e=e[1]-e[0]+1
           temp_matrix=np.zeros((length_e,length_e))
           for i in range(length_e):
               temp_matrix=temp_matrix+eye(e[1]-e[0]+1,-i)*func(e[2])/np.math.factorial(i)
           jordan_list.append(temp_matrix)

    #3.计算好的若当块f(Ai)组合成f(A)
    result=np.zeros((weight,height))
    length_list=index_list.__len__()

    for i in range(length_list):
        len_temp=list(jordan_list[i]).__len__()
        if len_temp==1:
           result[index_list[i][0]][index_list[i][0]]=jordan_list[i][0]
        else:
           for ii in range(len_temp):
               for ij in range(len_temp):
                   result[i+ii][i+ij]=jordan_list[i][ii][ij]
    # print("result:=",result)
    result=np.mat(T)*np.mat(result)*np.mat(T).I
    return result

#方法三，矩阵谱分析法
print("----------------谱值相等法参数求解------------------------")
#谱值相等法的求解思路是：利用求解矩阵A的最小多项式确定等价方程f（A）：其最高次数等于最小多项式首项系数为1的最高次项目-1
#用求多元线性方程组解的方式，计算等价方程参数a*x^2+b*x+c，a，b，c
A=sp.Matrix([[2,1,1],[1,2,1],[1,1,2]])
E=sp.Matrix([[r,0,0],[0,r,0],[0,0,r]])
b=sp.Matrix([[3,-1,1],[2,0,1],[1,-1,2]])
eig=factor(sp.Matrix(E-b).det())
print("谱值相等法特征多项式：=",eig)

#求解矩阵谱分析法的参数a*x^2+b*x+c，a，b，c
#具体方法见https://wenku.baidu.com/view/0d1dccea5901020206409ca2.html
temp_a=[[4,2,1],[4,1,0],[1,1,1]]#原理f(A)=g()
temp_y=np.exp(np.array([2,2,1]))
print("谱值相等法等价多项式系数：=",np.mat(temp_a).I*np.mat(temp_y).T)

print("-------------------矩阵函数各方法运算结果比较----------------------")
print("定义法：=",exp_n(b,100))
print("------------------------------------------------------------------")
pp=jordan_n(c,T)
print("若当矩阵法结果：=",pp)
print("------------------------------------------------------------------")
print("矩阵谱法:=",np.array(2.71828183*b*b-3.48407121*b+3.48407121*sp.eye(3)))
#可以用sin下，cos尝试下

print("第二部分-------------------矩阵微积分----------------------")
#举证微积分性质验证
#偏导公式
print("矩阵微积分偏导重要性质验证:")
A_t=sp.Matrix([[t,t**2],[t**2,t**3]])
B_t=sp.Matrix([[t**2,t**3],[t**3,t**4]])
# print("d(A_t)/dt:=",diff(A_t))
# print("d(B_t)/dt):=",diff(B_t))
#验证d(A_t*B_t)/dt=d(A_t)/dt*B_t+d(B_t)/dt*A_t,矩阵乘法是不满足交换律的
print("1、矩阵乘积偏导数性质：d(A_t*B_t)/dt=d(A_t)/dt*B_t+A_t*d(B_t)/dt")
print("d(A_t*B_t)/dt:= \n",diff(A_t*B_t))
print("d(A_t)/dt*B_t+d(B_t)/dt*A_t:= \n",diff(A_t)*B_t+A_t*diff(B_t))
print("相减判断是否相等：=\n",diff(A_t*B_t)-(diff(A_t)*B_t+A_t*diff(B_t)))

print("------------------------------------------------------------")
print("2、逆矩阵偏导数性质：diff(A_t.inv())=-A_t.inv()*diff(A_t)*A_t.inv()")
A_t=sp.Matrix([[t+1,t**2+5],[t**2,t**3]])
print("定义法逆矩阵求偏导:=\n",diff(A_t.inv()))
print("利用原矩阵法逆矩阵求偏导:=\n",-1*A_t.inv()*diff(A_t)*A_t.inv())
print("相减判断是否相等：=\n",expand(diff(A_t.inv())+A_t.inv()*diff(A_t)*A_t.inv()))
print("代码有问题！还是方程有问题，帮我检查下")

print("------------------------------------------------------------")
#矩阵积分
print("3、积分性质：")
A_t=sp.Matrix([[t+4,t**2+5],[t**2,t**3]])
B_b=sp.Matrix([[2,3],[4,5]])
print("A*B先乘再积分：=",integrate(A_t*B_b,t))
print("先积分再乘：=",integrate(A_t,t)*B_b)
print("相减判断是否相等：=\n",integrate(A_t*B_b,t)-integrate(A_t,t)*B_b)

print("---------------------------------------------------------------")
print("B*A先乘再积分：=",integrate(B_b*A_t,t))
print("先积分再乘：=",B_b*integrate(A_t,t))
print("相减判断是否相等：=\n",integrate(B_b*A_t,t)-B_b*integrate(A_t,t))

print("---------------------------------------------------------------")
print("4、部分积分性质：")
A_t=sp.Matrix([[t+4,t**2+5],[t**2,t**3]])
A_t_diff=diff(A_t)
B_t=sp.Matrix([[t**2,t**3],[t,t**2]])
B_t_diff=diff(B_t)
print("左边：=",integrate(A_t*B_t_diff,t))
print("右边：=",A_t*B_t-integrate(A_t_diff*B_t,t))
print("相减判断是否相等：=\n",expand(integrate(A_t*B_t_diff,t)-(A_t*B_t-integrate(A_t_diff*B_t,t))))

#数量矩阵对矩阵的导数性质，fun_y是多元矩阵到实数空间的函数（隐射）
print("--------------------------------------------------------------")
print("5、数量矩阵对矩阵的导数性质：")#多元函数对矩阵X的偏导，是矩阵对矩阵偏导的特殊情况1
fun_y1=x+y+r+t
fun_y2=x*2+y*3+r*4+t*2
B_t=sp.Matrix([[x,y],[t,r]])

def fun_diff_matrix(A_func,B=sp.eye(2,2)):#数量函数对矩阵求偏导
    weight=sp.Matrix(B).shape[1]
    height=sp.Matrix(B).shape[0]
    result=sp.zeros(height,weight)
    for j in  range(weight):
        for i in range(height):
            result[i,j]=diff(A_func,B[i,j])
    return result

def row_column_diff_matrix(A=sp.eye(2,1),B=sp.eye(2,2)):#行向量对矩阵求偏导
    A_weight=sp.Matrix(A).shape[1]
    A_height=sp.Matrix(A).shape[0]
    A_is_row_column=0#0是行向量,1是列向量

    B_weight=sp.Matrix(B).shape[1]
    B_height=sp.Matrix(B).shape[0]
    B_is_row_column=0#0是行向量,1是列向量

    if A_weight>1 and A_height==1:#列向量
       A_is_row_column=0
    else:
       A_is_row_column=1

    if B_weight>1 and B_height==1:#列向量
       B_is_row_column=0
    else:
       B_is_row_column=1

    result=0
    #一般是行对列求导或者列对行求导，当然也可以行对行求导，列对列求导，都满足矩阵对矩阵求导
    if A_is_row_column==1 and B_is_row_column==0:#A是列向量,B是行向量,行向量对列向量求导
       # print("列向量对行向量求导数")
       result=sp.zeros(A_height,B_weight)
       for i in range(sp.Matrix(A).shape[0]):
           for j in range(sp.Matrix(B).shape[1]):
               result[i,j]=diff(A[i,0],B[0,j])

    if A_is_row_column==0 and B_is_row_column==1:#A是行向量,B是行向量,行向量对列向量求导
        # print("行向量对列向量求导数")
        result=sp.zeros(B_height,A_weight)
        for i in range(sp.Matrix(A).shape[1]):
            for j in range(sp.Matrix(B).shape[0]):
                result[j,i]=diff(A[0,i],B[j,0])
    return result

print("(1)是否相等d（af(x)+bg(x)）/dx=a*df/dx+b*dg/dx:\n",fun_diff_matrix(2*fun_y1+4*fun_y2,B_t)-2*fun_diff_matrix(fun_y1,B_t)-4*fun_diff_matrix(fun_y2,B_t))
print("(2)d(f(x)*g(x))/dx=g(x)*df/dx+f(x)*dg/dx:\n",fun_diff_matrix(fun_y1*fun_y2,B_t)-fun_y2*fun_diff_matrix(fun_y1,B_t)-fun_y1*fun_diff_matrix(fun_y2,B_t))
A_t=sp.Matrix([[2,3],[4,5]])
print("(3)d(tr(Ax))/dx:\n",fun_diff_matrix(sp.trace(A_t*B_t),B_t))
print("(4)d(det(x)):\n",fun_diff_matrix(B_t.det(),B_t))

print("--------------------------------------------------------------")
print("6、列（行）向量对行（列）向量的导数：")#行（列）向量对矩阵X的偏导，是矩阵对矩阵偏导的特殊情况2
A_t=sp.Matrix([[x+y+r+t+4,t**2+5,r+y**2+t**2,x**2+t**3]])
B_t=sp.Matrix([[x,y,t,r]])

print("(2)dA^T/d=（dA/dX^T）^T")#行列向量相互求导，等于其转置
print(row_column_diff_matrix(A_t.transpose(),B_t))
print(row_column_diff_matrix(A_t,B_t.transpose()))
print("是否相等:",row_column_diff_matrix(A_t.transpose(),B_t).transpose()-row_column_diff_matrix(A_t,B_t.transpose()))

print("(3)d(MA(X))/dx^T=M*dA/dx^T")#列向量对行向量求导
A_t=sp.Matrix([[x+y+r+t+4,t**2+5,r+y**2+t**2,x**2+t**3]]).transpose()
M_t=sp.Matrix([[1,2,3,4],[5,6,7,8],[1,2,3,4],[8,9,10,11]])
B_t=sp.Matrix([[x,y,t,r]])
print("是否相等:",row_column_diff_matrix(M_t*A_t,B_t)-M_t*row_column_diff_matrix(A_t,B_t))

print("(4)d[f(x)*A(x)]/dx^T=A*df/dx^T+f*dA/x^T")
A_t=sp.Matrix([[x+y+r+t+4,t**2+5,r+y**2+t**2,x**2+t**3]]).transpose()
f_t=x**2+4*y**2+r+t**3
B_t=sp.Matrix([[x,y,t,r]])#x_t
print("是否相等:",row_column_diff_matrix(f_t*A_t,B_t)-(A_t*fun_diff_matrix(f_t,B_t)+f_t*row_column_diff_matrix(A_t,B_t)))

print("(5)d[A(x)^T*B(x)]/dx=dA^T/dx*B+dB^T/x*A")
A_t=sp.Matrix([[x+y+r+t+4,t**2+5,r+y**2+x**2,x**2+t**3]])
B_t=sp.Matrix([[x+y+r+t+4,x+r+t+5,x+y+r+6,y+r+t*2+7]]).transpose()
x_t=sp.Matrix([[x,y,t,r]]).transpose()#x_t
print("是否相等:",fun_diff_matrix(A_t*B_t,x_t)-(row_column_diff_matrix(A_t,x_t)*B_t+row_column_diff_matrix(B_t.transpose(),x_t)*A_t.transpose()))

print("--------------------------------------------------------------")
#其中A*t表示变量t和一个常数项矩阵相乘
print("7.用若当法计算矩阵函数，验证d(exp(A*t))/dt=A*exp(A*t)")#单变量矩阵求导法则
def jordan_t_sin(a,T=np.zeros((3,3))):
    #提取若当标准型中不同的若当块
    jordan_matrix=sp.Matrix(a)
    weight=jordan_matrix.shape[0]
    height=jordan_matrix.shape[1]

    #1、获取若当矩阵中每个若当块的位置
    def catch_jordan_index(jordan_matrix=jordan_matrix):

        index=(0,0,0)#记录若当矩一个阵若当块的位置信息（开始位置，结束位置，元素值）
        index_list=[]
        i=0
        while i<height:
            if i==0:
                index=(0,0,np.array(jordan_matrix)[0][0])
                #判断对角线上相同元素
                temp=index[1]+1
                while temp<height:
                    if index[2]==np.array(jordan_matrix)[temp][temp] and np.array(jordan_matrix)[temp][temp-1]!=0:
                        temp=temp+1#沿对角线下移动一格
                    else:
                        break
                index=(0,temp-1,np.array(jordan_matrix)[0][0])
                index_list.append(index)
                i=index[1]+1
            else:
                first_index=index_list[-1][1]+1
                index=(first_index,first_index,np.array(jordan_matrix)[first_index][first_index])
                temp=index[1]+1
                if temp<height:
                    while temp<height:
                        if index[2]==np.array(jordan_matrix)[temp][temp] and np.array(jordan_matrix)[temp][temp-1]!=0:
                            temp=temp+1#沿对角线下移动一格
                        else:
                            break
                    index=(first_index,temp-1,np.array(jordan_matrix)[first_index][first_index])
                    index_list.append(index)
                    i=index[1]+1
                else:
                    index_list.append(index)
                    i=index[1]+1
        return index_list
    index_list=catch_jordan_index(jordan_matrix=np.mat(jordan_matrix))
    # print(index_list)
    #2.计算每个若当块f（Ai）i=0，1，。。r,r是矩阵A若当块的个数
    #生成mi行为斜角的矩阵，案例中为半矩阵上斜角，本例中为下半矩阵斜角
    def eye(num,row):
        return np.eye(num, k=row)

    jordan_list=[]#存储每个若当块的f(a)
    for e in index_list:
        if e[1]-e[0]+1<2:#仅有1个元素
            jordan_list.append(np.array([sp.sin(e[2]*t)]))
        else:
            length_e=e[1]-e[0]+1
            temp_matrix=sp.zeros(length_e,length_e)
            for i in range(length_e):
                # f=sp.sin(x)
                # f=diff(f.replace(x,e[2]*t),t,i)
                # print("f:=",f,i)
                f=diff(sp.sin(x),x,i)
                f=f.replace(x,e[2]*t)
                # print("f_1:=",f_1,i)
                # f=sp.sin(x)
                # f=diff(f.replace(x,e[2]*t),t,i)
                temp_matrix=temp_matrix+sp.Matrix(eye(e[1]-e[0]+1,-i))*f*(t**i)/np.math.factorial(i)
            jordan_list.append(temp_matrix)

    #3.计算好的若当块f(Ai)组合成f(A)
    result=sp.zeros(weight,height)
    length_list=index_list.__len__()

    for i in range(length_list):


        if int(list(jordan_list[i]).__len__())==1:
           len_temp=1
        else:
           len_temp=int(list(jordan_list[i]).__len__()/2)#sp.matrix的length 是整个矩阵拉直后的长度

        if len_temp==1:
            result[index_list[i][0],index_list[i][0]]=jordan_list[i][0]
        else:
            for ii in range(len_temp):
                for ij in range(len_temp):
                    result[i+ii,i+ij]=jordan_list[i][ii,ij]
    result=sp.Matrix(T)*result*sp.Matrix(T).inv()
    return result

def jordan_t_cos(a,T=np.zeros((3,3))):
    #提取若当标准型中不同的若当块
    jordan_matrix=sp.Matrix(a)
    weight=jordan_matrix.shape[0]
    height=jordan_matrix.shape[1]

    #1、获取若当矩阵中每个若当块的位置
    def catch_jordan_index(jordan_matrix=jordan_matrix):

        index=(0,0,0)#记录若当矩一个阵若当块的位置信息（开始位置，结束位置，元素值）
        index_list=[]
        i=0
        while i<height:
            if i==0:
                index=(0,0,np.array(jordan_matrix)[0][0])
                #判断对角线上相同元素
                temp=index[1]+1
                while temp<height:
                    if index[2]==np.array(jordan_matrix)[temp][temp] and np.array(jordan_matrix)[temp][temp-1]!=0:
                        temp=temp+1#沿对角线下移动一格
                    else:
                        break
                index=(0,temp-1,np.array(jordan_matrix)[0][0])
                index_list.append(index)
                i=index[1]+1
            else:
                first_index=index_list[-1][1]+1
                index=(first_index,first_index,np.array(jordan_matrix)[first_index][first_index])
                temp=index[1]+1
                if temp<height:
                    while temp<height:
                        if index[2]==np.array(jordan_matrix)[temp][temp] and np.array(jordan_matrix)[temp][temp-1]!=0:
                            temp=temp+1#沿对角线下移动一格
                        else:
                            break
                    index=(first_index,temp-1,np.array(jordan_matrix)[first_index][first_index])
                    index_list.append(index)
                    i=index[1]+1
                else:
                    index_list.append(index)
                    i=index[1]+1
        return index_list
    index_list=catch_jordan_index(jordan_matrix=np.mat(jordan_matrix))
    # print(index_list)
    #2.计算每个若当块f（Ai）i=0，1，。。r,r是矩阵A若当块的个数
    #生成mi行为斜角的矩阵，案例中为半矩阵上斜角，本例中为下半矩阵斜角
    def eye(num,row):
        return np.eye(num, k=row)#sp.eye会有错误

    jordan_list=[]#存储每个若当块的f(a)
    for e in index_list:
        if e[1]-e[0]+1<2:#仅有1个元素
            jordan_list.append(np.array([sp.cos(e[2]*t)]))
        else:
            length_e=e[1]-e[0]+1
            temp_matrix=sp.zeros(length_e,length_e)
            for i in range(length_e):
                f=diff(sp.cos(x),x,i)
                f=f.replace(x,e[2]*t)
                # f=sp.cos(x)
                # f=diff(f.replace(x,e[2]*t),t,i)
                temp_matrix=temp_matrix+sp.Matrix(eye(e[1]-e[0]+1,-i))*f*(t**i)/np.math.factorial(i)
            jordan_list.append(temp_matrix)

    #3.计算好的若当块f(Ai)组合成f(A)
    result=sp.zeros(weight,height)
    length_list=index_list.__len__()

    for i in range(length_list):
        if int(list(jordan_list[i]).__len__())==1:
           len_temp=1
        else:
           len_temp=int(list(jordan_list[i]).__len__()/2)#sp.matrix的length 是整个矩阵拉直后的长度

        if len_temp==1:
            result[index_list[i][0],index_list[i][0]]=jordan_list[i][0]
        else:
            for ii in range(len_temp):
                for ij in range(len_temp):
                    result[i+ii,i+ij]=jordan_list[i][ii,ij]
    result=sp.Matrix(T)*result*sp.Matrix(T).inv()
    return result

A_o=sp.Matrix([[3,-1,1],[2,0,1],[1,-1,2]])
# c=np.mat([[1,0,0],
#           [0,2,0],
#           [0,1,2]])
T=sp.Matrix([[0,0,1],[1,0,1],[1,1,0]]) #矩阵b的过渡矩阵
A=T.inv()*A_o*T
A_t_sin=jordan_t_sin(A,T)
#print(A_t_sin)
A_t_cos=jordan_t_cos(A,T)
#print(A_t_cos)
#print(Matrix([[-2.0*sin(2*0) + 2.0*cos(2*0), 2.0*sin(2*0), -2.0*sin(2*0)], [-2.0*sin(2*0) - cos(0) + 2.0*cos(2*0), 2.0*sin(2*0) + cos(0), -2.0*sin(2*0)], [-cos(0) + 2.0*cos(2*0), cos(0) - 2.0*cos(2*0), 2.0*cos(2*0)]]))

# print("sin:\n",A_t_sin)
# print("cos:\n",A_t_cos)
# print("A_T_diff:\n",diff(A_t_sin,t))
print("diff(A_t_sin,t)=\n",diff(A_t_sin,t))
print("A_t_cos*A_o=\n",A_t_cos*A_o)
print("是否相等d(exp(A*t))/dt=A*exp(A*t)：\n",expand(diff(A_t_sin,t)-A_t_cos*A_o))

print("-------------------------------------------------------------")
print("8.用谱分析法，验证d(exp(A*t))/dt=A*exp(A*t)")#单变量矩阵求导法则
A_t=2.71828183*(A_o*t)*(A_o*t)-3.48407121*(A_o*t)+3.48407121*sp.eye(3)
print(expand(diff(A_t)-A_t*A_o))
print("用谱分析法计算f(A*t)的结论是不对的，为什么？")

