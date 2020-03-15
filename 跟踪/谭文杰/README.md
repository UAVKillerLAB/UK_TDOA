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
  代码采用矢量卡尔曼（二维）的算法
  对于从主站得到的位置信息，对它进行物理模型的建立，
  ![]（https://github.com/cannercan/-/blob/master/th.jpg?raw=true）
- 2.代码接口定义（需要说明的输入输出变量等）
- 3.代码流程
- 4.预期结果
- 5.实际结果
- 6.原因分析
- 7.下一步工作