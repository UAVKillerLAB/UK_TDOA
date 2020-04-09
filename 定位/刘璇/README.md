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
- 1.时差定位又称为双曲线定位，由三个或者以上的观测站构成,
这个代码中设置了三个辅站和一个主站，辅站把自己接收到的无人
机信号传给主站，由主站测量信号传给辅站和传给主站的时间差，
通过时间差画出双曲线，无人机的位置必定在轨线交点处。
![Picture](/C:\Users\tayloryoo\Pictures\Saved Pictures/1.png)

- 2.输入：三个时差，输出：无人机坐标以及示意图。
![Picture](/C:\Users\tayloryoo\Pictures\Saved Pictures/2.png)
![Picture](/C:\Users\tayloryoo\Pictures\Saved Pictures/4.png)
坐标为array([6930.73846652])

- 3.代码流程：先输入三个时差后设置三个辅站BS1~BS3在设置主站
MS，BS1~BS3坐标均已知(xi,yi,zi)，MS坐标未知设为（x,y,z）。
![Picture](/C:\Users\tayloryoo\Pictures\Saved Pictures/3.png)
第一步先计算出主站与辅站之间的距离，最终得到一个非线性方程组
，采用加权最小二乘法算出MS的估计值，通过Chan算下来的数值当作
Taylor的初值。然后给定门限的大小，若小于给定门限则停止迭代算法。


- 4.预期结果与实际吻合
- 5.实际结果：可接受范围内误差
- 6.原因分析：
- 7.优化算法