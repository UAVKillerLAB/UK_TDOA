![uk-logo](https://s2.ax1x.com/2020/01/19/1C8qXt.png)
# 无人机无源探测关键技术研发实验室（软件说明模板）
## UK_TDOA项目(软件)

### 1.项目开发环境

- MySQL版本:8.0.17 Community Server
- Python解释器版本:3.7.4
- 第三方依赖库
    
|Lib            |Ver    |
|:----          |:----  |
|numpy          |1.17.4 |
|pymysql        |0.9.3  |
|serial         |0.0.97 |
|matplotlib     |3.1.2  |
|geographiclib  |1.50   |
|django         |3.0.1  |

### 2. 项目源码简要说明
- 1.代码原理
    代码采用chan算法和泰勒算法联用，chan算法通过将双曲线方程组转化为线性方程组，并借助其他变量的方式得到目标位置的解析解。

    主接受站S0（x0，y0，z0），辅助观测站S1（x1,y1,z1）,S2(x2,y2,z2),S3(x3,y3,z3)。
    ![](http://latex.codecogs.com/gif.latex?%24%24%20%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bl%7D%20%7Br_%7B0%7D%3D%5Csqrt%7B%5Cleft%28x-x_%7B0%7D%5Cright%29%5E%7B2%7D&plus;%5Cleft%28y-y_%7B0%7D%5Cright%29%5E%7B2%7D&plus;%5Cleft%28z-z_%7B0%7D%5Cright%29%5E%7B2%7D%7D%7D%20%5C%5C%20%7Br_%7Bi%7D%3D%5Csqrt%7B%5Cleft%28x-x_%7Bi%7D%5Cright%29%5E%7B2%7D&plus;%5Cleft%28y-y_%7Bi%7D%5Cright%29%5E%7B2%7D&plus;%5Cleft%28z-z_%7Bi%7D%5Cright%29%5E%7B2%7D%7D%20%5Cquad%28i%3D1%2C2%2C3%2C4%29%7D%20%5C%5C%20%7Br_%7Bi%200%7D%3Dr_%7Bi%7D-r_%7B0%7D%3Dc%20%5Ccdot%20%5CDelta%20t_%7Bi%7D%7D%20%5Cend%7Barray%7D%5Cright.%20%24%24)

    采用chan算法对上诉方程式求解可以得到以下关系式：
    ![](http://latex.codecogs.com/gif.latex?%24%24%20%5Chat%7BX%7D%3D%5Cleft%28A%5E%7BT%7D%20A%5Cright%29%5E%7B-1%7D%20A%5E%7BT%7D%20F%20%24%24)

    其中：
    ![](http://latex.codecogs.com/gif.latex?%24%24%20%5Chat%7BX%7D%3D%5Cleft%28A%5E%7BT%7D%20A%5Cright%29%5E%7B-1%7D%20A%5E%7BT%7D%20F%20%24%24)
    ![](http://latex.codecogs.com/gif.latex?%24%24%20A%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Blll%7D%20%7Bx_%7B01%7D%7D%20%26%20%7By_%7B01%7D%7D%20%26%20%7Bz_%7B01%7D%7D%20%5C%5C%20%7Bx_%7B02%7D%7D%20%26%20%7By_%7B02%7D%7D%20%26%20%7Bz_%7B02%7D%7D%20%5C%5C%20%7Bx_%7B03%7D%7D%20%26%20%7By_%7B03%7D%7D%20%26%20%7Bz_%7B03%7D%7D%20%5C%5C%20%7Bx_%7B04%7D%7D%20%26%20%7By_%7B04%7D%7D%20%26%20%7Bz_%7B04%7D%7D%20%5Cend%7Barray%7D%5Cright%5D%2C%20%5Cquad%20x_%7B0%20i%7D%3Dx_%7B0%7D-x_%7Bi%7D%2C%20%5Cquad%20y_%7B0%20i%7D%3Dy_%7B0%7D-y_%7Bi%7D%2C%20%5Cquad%20z_%7B0%20i%7D%3Dz_%7B0%7D-z_%7Bi%7D%20%24%24)

    泰勒算法需要一个预估定位，该位置可由chan算法计算得到的定位提供，泰勒算法的基本思想是将时差看作是目标位置的函数，通过对其级数展开并利用关系式构造线性方程组，解算出无人机坐标坐标，再差值向量，然后进一步迭代，直到达到预设阈值之后方可退出迭代，即可得到当前迭代出的坐标。
    首先先将chan算法所得到的定位带入定位方程中进行泰勒展开得到下式：
    ![](http://latex.codecogs.com/gif.latex?%24%24%20%5Chat%7BX%7D%3D%5Cleft%28A%5E%7BT%7D%20A%5Cright%29%5E%7B-1%7D%20A%5E%7BT%7D%20F%20%24%24)

    其中：
    ![](http://latex.codecogs.com/gif.latex?%24%24%20e%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Blll%7D%20%7Be_%7B1%7D%7D%20%26%20%7Be_%7B2%7D%7D%20%26%20%7Be_%7B3%7D%7D%20%5Cend%7Barray%7D%5Cright%5D%5E%7B%5Cmathrm%7BT%7D%7D%20%24%24)

    上式代表时间测量误差，下式代表目标估计误差：
    ![](http://latex.codecogs.com/gif.latex?%24%24%20%5Cdelta%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Blll%7D%20%7B%5CDelta%20x%7D%20%26%20%7B%5CDelta%20y%7D%20%26%20%7B%5CDelta%20z%7D%20%5Cend%7Barray%7D%5Cright%5D%5E%7B%5Cmathrm%7BT%7D%7D%20%24%24)

    该式代表辐射源到目标真实值与测量值之差：
    ![](http://latex.codecogs.com/gif.latex?%24%24%20%5Cboldsymbol%7Bh%7D%3D%5Cleft%5B%5CDelta%20R_%7B1%7D-%5Cleft%28R_%7B1%7D-R_%7B0%7D%5Cright%29%20%5CDelta%20R_%7B2%7D-%5Cleft%28R_%7B2%7D-R_%7B0%7D%5Cright%29%20%5CDelta%20R_%7B3%7D-%5Cleft%28R_%7B3%7D-R_%7B0%7D%5Cright%29%5Cright%5D%5E%7B%5Cmathrm%7BT%7D%7D%20%24%24)

    再利用最小二乘法可以得到：
    ![](http://latex.codecogs.com/gif.latex?%5Cdelta%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Blll%7D%20%7B%5CDelta%20x%7D%20%26%20%7B%5CDelta%20y%7D%20%26%20%7B%5CDelta%20z%7D%20%5Cend%7Barray%7D%5Cright%5D%5E%7B%5Cmathrm%7BT%7D%7D%3D%5Cleft%28%5Cboldsymbol%7BG%7D%5E%7BT%7D%20%5Cboldsymbol%7BQ%7D%20%5Cboldsymbol%7BG%7D%5Cright%29%5E%7B-1%7D%20%5Cboldsymbol%7BG%7D%5E%7BT%7D%20Q%20%5Cboldsymbol%7Bh%7D)

    其中：
    ![](http://latex.codecogs.com/gif.latex?Q%3DE%5Cleft%5Be%20e%5E%7BT%7D%5Cright%5D)

    最后再将：
    ![](http://latex.codecogs.com/gif.latex?%5Ctext%20%7B%20If%20%7D%20%5Csqrt%7B%5CDelta%20x%5E%7B2%7D&plus;%5CDelta%20y%5E%7B2%7D&plus;%5CDelta%20z%5E%7B2%7D%7D)

    与预设阈值进行比较，如果大于阈值，则继续循环执行上诉步骤，如果小于等于阈值则退出循环得到当前坐标值
- 2.代码接口定义
    输入：时差（三站两个时差，四站三个时差），输出：无人机坐标以及坐标图（需要说明的输入输出变量等）
- 3.代码流程
   输入时差 -->得到定位方程-->方程线性化求解-->得到坐标-->泰勒迭代达到阈值-->跟新当前坐标-->绘图描点坐标 
- 4.预期结果
    输出定位坐标，并且生成无人机坐标图
- 5.实际结果
    与预期一致
- 6.原因分析
- 7.下一步工作
    测试代码与其他方向联调