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
    基于基站坐标（866，500），（-866，500），（0，-1000），随机输入一个信号源坐标，生成时差，再运用定位算法进行定位测试定位效果
- 2.效果说明
    在当前情形之下，若信号源位于三站以内或靠近三站范围附近，那么定位精度相当高，最大误差不超过2米，测试出来的平均误差在0.5m以内；
    若信号源位于三站以外较远，那么定位精度相当低，想准确定位是不可能的。
- 3.效果图形展示
    输入信号源坐标（300，300），定位结果为（299.99997757 299.99998034）：
    ![image](https://github.com/zhang271018/images/06Q2WRM{Q){{_V95W2KHNLN.png)
    ![image](https://github.com/zhang271018/images/KMK7XYR99C@2D653$X(XQW5.png)
    ![image](https://github.com/zhang271018/images/`[ILE0`4HS%YB23DAV)TBD4.png)
    输入信号源坐标（900，1000），定位结果为（899.99994865 999.99995445）：
    ![image](https://github.com/zhang271018/images/7UFWVKQNEWC[Y1Y4}P7WMUR.png)
    ![image](https://github.com/zhang271018/images/$OJY9659DFD16Q@FY$RTSOS.png)
    输入信号源坐标（1500，1500），定位结果为（1291.08723475 1318.96383173）：
    ![image](https://github.com/zhang271018/images/JA51LZ%X`V{][`LEG1A4SZR.png)
    ![image](https://github.com/zhang271018/images/(3FZEWBP6~@C${XQY_T~B1S.png)