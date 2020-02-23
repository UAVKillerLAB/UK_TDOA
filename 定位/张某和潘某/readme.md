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
$$
\left\{\begin{array}{l}
{r_{0}=\sqrt{\left(x-x_{0}\right)^{2}+\left(y-y_{0}\right)^{2}+\left(z-z_{0}\right)^{2}}} \\
{r_{i}=\sqrt{\left(x-x_{i}\right)^{2}+\left(y-y_{i}\right)^{2}+\left(z-z_{i}\right)^{2}} \quad(i=1,2,3,4)} \\
{r_{i 0}=r_{i}-r_{0}=c \cdot \Delta t_{i}}
\end{array}\right.
$$
\end{array}\right.
    其中▲ti为到第i站与到主站产生的时差，以上三式即为定位方程。
    将方程整理可以得到：
    % MathType!MTEF!2!1!+-
% feaagKart1ev2aaatCvAUfeBSn0BKvguHDwzZbqefSSZmxoarmWu51
% MyVXgatCvAUfeBSjuyZL2yd9gzLbvyNv2CaeHbd9wDYLwzYbItLDha
% ryavP1wzZbItLDhis9wBH5garqqtubsr4rNCHbGeaGqiVu0Je9sqqr
% pepC0xbbL8F4rqqrFfpeea0xe9Lq-Jc9vqaqpepm0xbba9pwe9Q8fs
% 0-yqaqpepae9pg0FirpepeKkFr0xfr-xfr-xb9adbaqaaeGaciGaai
% aabeqaamaaeaqbaeaakeaacaGGOaGaamiEaSGaaGimaOGaeyOeI0Ia
% amiEaSGaamyAaOGaaiykaiaadIhacqGHRaWkcaGGOaGaamyEaSGaaG
% imaOGaeyOeI0IaamyEaSGaamyAaOGaaiykaiaadMhacqGHRaWkcaGG
% OaGaamOEaSGaaGimaOGaeyOeI0IaamOEaSGaamyAaOGaaiykaiaadQ
% hacqGH9aqpcaWGRbWccaWGPbGccqGHRaWkcaWGYbWccaaIWaGccqGH
% flY1cqGHuoarcaWGYbWccaWGPbaaaa!615E!
\[(x0 - xi)x + (y0 - yi)y + (z0 - zi)z = ki + r0 \cdot \Delta ri\]
    其中，% MathType!MTEF!2!1!+-
% feaagKart1ev2aaatCvAUfeBSn0BKvguHDwzZbqefSSZmxoarmWu51
% MyVXgatCvAUfeBSjuyZL2yd9gzLbvyNv2CaeHbd9wDYLwzYbItLDha
% ryavP1wzZbItLDhis9wBH5garqqtubsr4rNCHbGeaGqiVu0Je9sqqr
% pepC0xbbL8F4rqqrFfpeea0xe9Lq-Jc9vqaqpepm0xbba9pwe9Q8fs
% 0-yqaqpepae9pg0FirpepeKkFr0xfr-xfr-xb9adbaqaaeGaciGaai
% aabeqaamaaeaqbaeaakeaacaWGRbWccaWGPbGccqGH9aqpdaWcaaqa
% aiaaigdaaeaacaaIYaaaaiaacUfacqGHuoarcaWGYbWccaWGPbGcda
% ahaaWcbeqaaiaaikdaaaGccqGHRaWkcaGGOaGaamiEaSGaaGimaOWa
% aWbaaSqabeaacaaIYaaaaOGaey4kaSIaamyEaSGaaGimaOWaaWbaaS
% qabeaacaaIYaaaaOGaey4kaSIaamOEaSGaaGimaOWaaWbaaSqabeaa
% caaIYaaaaOGaaiykaiabgkHiTiaacIcacaWG4bWccaWGPbGcdaahaa
% WcbeqaaiaaikdaaaGccqGHRaWkcaWG5bWccaWGPbGcdaahaaWcbeqa
% aiaaikdaaaGccqGHRaWkcaWG6bWccaWGPbGcdaahaaWcbeqaaiaaik
% daaaGccaGGPaGaaiyxaaaa!62DB!
\[ki = \frac{1}{2}[\Delta r{i^2} + (x{0^2} + y{0^2} + z{0^2}) - (x{i^2} + y{i^2} + z{i^2})]\]
    可以得到以下矩阵：AX=F
    其中A=% MathType!MTEF!2!1!+-
% feaagKart1ev2aaatCvAUfeBSn0BKvguHDwzZbqefSSZmxoarmWu51
% MyVXgatCvAUfeBSjuyZL2yd9gzLbvyNv2CaeHbd9wDYLwzYbItLDha
% ryavP1wzZbItLDhis9wBH5garqqtubsr4rNCHbGeaGqiVu0Je9sqqr
% pepC0xbbL8F4rqqrFfpeea0xe9Lq-Jc9vqaqpepm0xbba9pwe9Q8fs
% 0-yqaqpepae9pg0FirpepeKkFr0xfr-xfr-xb9adbaqaaeGaciGaai
% aabeqaamaaeaqbaeaakeaafaqabeWadaaabaGaamiEaSGaaGimaiaa
% igdaaOqaaiaadMhaliaaicdacaaIXaaakeaacaWG6bWccaaIWaGaaG
% ymaaGcbaGaamiEaSGaaGimaiaaikdaaOqaaiaadMhaliaaicdacaaI
% YaaakeaacaWG6bWccaaIWaGaaGOmaaGcbaGaamiEaSGaaGimaiaaio
% daaOqaaiaadMhaliaaicdacaaIZaaakeaacaWG6bWccaaIWaGaaG4m
% aaaaaaa!5600!
\[\begin{array}{*{20}{c}}
{x01}&{y01}&{z01}\\
{x02}&{y02}&{z02}\\
{x03}&{y03}&{z03}
\end{array}\]，x0i=x0-xi,y01=y0-yi,z0i=z0-zi;
    X=% MathType!MTEF!2!1!+-
% feaagKart1ev2aaatCvAUfeBSn0BKvguHDwzZbqefSSZmxoarmWu51
% MyVXgatCvAUfeBSjuyZL2yd9gzLbvyNv2CaeHbd9wDYLwzYbItLDha
% ryavP1wzZbItLDhis9wBH5garqqtubsr4rNCHbGeaGqiVu0Je9sqqr
% pepC0xbbL8F4rqqrFfpeea0xe9Lq-Jc9vqaqpepm0xbba9pwe9Q8fs
% 0-yqaqpepae9pg0FirpepeKkFr0xfr-xfr-xb9adbaqaaeGaciGaai
% aabeqaamaaeaqbaeaakeaafaqabeWabaaabaGaamiEaaqaaiaadMha
% aeaacaWG6baaaaaa!422B!
\[\begin{array}{*{20}{c}}
x\\
y\\
z
\end{array}\],F=% MathType!MTEF!2!1!+-
% feaagKart1ev2aaatCvAUfeBSn0BKvguHDwzZbqefSSZmxoarmWu51
% MyVXgatCvAUfeBSjuyZL2yd9gzLbvyNv2CaeHbd9wDYLwzYbItLDha
% ryavP1wzZbItLDhis9wBH5garqqtubsr4rNCHbGeaGqiVu0Je9sqqr
% pepC0xbbL8F4rqqrFfpeea0xe9Lq-Jc9vqaqpepm0xbba9pwe9Q8fs
% 0-yqaqpepae9pg0FirpepeKkFr0xfr-xfr-xb9adbaqaaeGaciGaai
% aabeqaamaaeaqbaeaakeaafaqabeWabaaabaGaam4AaSGaaGymaOGa
% ey4kaSIaamOCaSGaaGimaOGaeyyXICTaeyiLdqKaamOCaSGaaGymaa
% GcbaGaam4AaSGaaGOmaOGaey4kaSIaamOCaSGaaGimaOGaeyyXICTa
% eyiLdqKaamOCaSGaaGOmaaGcbaGaam4AaSGaaG4maOGaey4kaSIaam
% OCaSGaaGimaOGaeyyXICTaeyiLdqKaamOCaSGaaG4maaaaaaa!5CCD!
\[\begin{array}{*{20}{c}}
{k1 + r0 \cdot \Delta r1}\\
{k2 + r0 \cdot \Delta r2}\\
{k3 + r0 \cdot \Delta r3}
\end{array}\]。
    利用伪逆法得到X的最小二乘解为：
    % MathType!MTEF!2!1!+-
% feaagKart1ev2aaatCvAUfeBSn0BKvguHDwzZbqefSSZmxoarmWu51
% MyVXgatCvAUfeBSjuyZL2yd9gzLbvyNv2CaeHbd9wDYLwzYbItLDha
% ryavP1wzZbItLDhis9wBH5garqqtubsr4rNCHbGeaGqiVu0Je9sqqr
% pepC0xbbL8F4rqqrFfpeea0xe9Lq-Jc9vqaqpepm0xbba9pwe9Q8fs
% 0-yqaqpepae9pg0FirpepeKkFr0xfr-xfr-xb9adbaqaaeGaciGaai
% aabeqaamaaeaqbaeaakeaacaWGybGaeyypa0Jaaiikaiaadgeadaah
% aaWcbeqaaiaadsfaaaGccaWGbbGaaiykamaaCaaaleqabaGaeyOeI0
% IaaGymaaaakiaadgeadaahaaWcbeqaaiaadsfaaaGccaWGgbaaaa!497A!
\[X = {({A^T}A)^{ - 1}}{A^T}F\]
    令% MathType!MTEF!2!1!+-
% feaagKart1ev2aaatCvAUfeBSn0BKvguHDwzZbqefSSZmxoarmWu51
% MyVXgatCvAUfeBSjuyZL2yd9gzLbvyNv2CaeHbd9wDYLwzYbItLDha
% ryavP1wzZbItLDhis9wBH5garqqtubsr4rNCHbGeaGqiVu0Je9sqqr
% pepC0xbbL8F4rqqrFfpeea0xe9Lq-Jc9vqaqpepm0xbba9pwe9Q8fs
% 0-yqaqpepae9pg0FirpepeKkFr0xfr-xfr-xb9adbaqaaeGaciGaai
% aabeqaamaaeaqbaeaakeaacaGGOaGaamyqamaaCaaaleqabaGaamiv
% aaaakiaadgeacaGGPaWaaWbaaSqabeaacqGHsislcaaIXaaaaOGaam
% yqamaaCaaaleqabaGaamivaaaakiabg2da9iaacUfacaWGHbWccaWG
% PbGaamOAaOGaaiyxaSGaaG4maiaacQcacaaIZaaaaa!4E9D!
\[{({A^T}A)^{ - 1}}{A^T} = [aij]3*3\]
    则方程组解为:% MathType!MTEF!2!1!+-
% feaagKart1ev2aaatCvAUfeBSn0BKvguHDwzZbqefSSZmxoarmWu51
% MyVXgatCvAUfeBSjuyZL2yd9gzLbvyNv2CaeHbd9wDYLwzYbItLDha
% ryavP1wzZbItLDhis9wBH5garqqtubsr4rNCHbGeaGqiVu0Je9sqqr
% pepC0xbbL8F4rqqrFfpeea0xe9Lq-Jc9vqaqpepm0xbba9pwe9Q8fs
% 0-yqaqpepae9pg0FirpepeKkFr0xfr-xfr-xb9adbaqaaeGaciGaai
% aabeqaamaaeaqbaeaakqaabeqaaiaadIhacqGH9aqpcaWGTbWccaaI
% XaGccqGHRaWkcaWGUbWccaaIXaGccqGHflY1caWGYbWccaaIWaaake
% aacaWG4bGaeyypa0JaamyBaSGaaGOmaOGaey4kaSIaamOBaSGaaGOm
% aOGaeyyXICTaamOCaSGaaGimaaGcbaGaamiEaiabg2da9iaad2gali
% aaiodakiabgUcaRiaad6galiaaiodakiabgwSixlaadkhaliaaicda
% aaaa!5E94!
\[\begin{array}{l}
x = m1 + n1 \cdot r0\\
x = m2 + n2 \cdot r0\\
x = m3 + n3 \cdot r0
\end{array}\]
    其中：% MathType!MTEF!2!1!+-
% feaagKart1ev2aaatCvAUfeBSn0BKvguHDwzZbqefSSZmxoarmWu51
% MyVXgatCvAUfeBSjuyZL2yd9gzLbvyNv2CaeHbd9wDYLwzYbItLDha
% ryavP1wzZbItLDhis9wBH5garqqtubsr4rNCHbGeaGqiVu0Je9sqqr
% pepC0xbbL8F4rqqrFfpeea0xe9Lq-Jc9vqaqpepm0xbba9pwe9Q8fs
% 0-yqaqpepae9pg0FirpepeKkFr0xfr-xfr-xb9adbaqaaeGaciGaai
% aabeqaamaaeaqbaeaakqaabeqaaiaad2galiaadMgacqGH9aqpdaae
% WbqaaOGaamyyaSGaamyAaiaadQgakiabgwSixlaadUgaliaadQgaaW
% qaaiaadQgacqGH9aqpcaaIXaaabaGaaG4maaGdcqGHris5aaGcbaGa
% amOBaSGaamyAaiabg2da9maaqahabaGccaWGHbWccaWGPbGaamOAaO
% GaeyyXICTaeyiLdqKaamOCaSGaamOAaaadbaGaamOAaiabg2da9iaa
% igdaaeaacaaIZaaaoiabggHiLdaaaaa!6012!
\[\begin{array}{l}
mi = \sum\limits_{j = 1}^3 {aij \cdot kj} \\
ni = \sum\limits_{j = 1}^3 {aij \cdot \Delta rj} 
\end{array}\]
    并将上式代入原始定位式即可得到一个二次方程，利用求根公式即可求得此解。
    泰勒算法需要一个预估定位，该位置可由chan算法计算得到的定位提供，泰勒算法的基本思想是将时差看作是目标位置的函数，通过对其级数展开并利用关系式构造线性方程组，解算出无人机坐标坐标，再差值向量，然后进一步迭代，直到达到预设阈值之后方可退出迭代，即可得到当前迭代出的坐标。

- 2.代码接口定义
    输入：时差（三站两个时差，四站三个时差），输出：无人机坐标以及坐标图（需要说明的输入输出变量等）
- 3.代码流程
mermaid
graph LR
   输入时差 -->得到定位方程-->方程线性化求解-->得到坐标-->泰勒迭代达到阈值-->跟新当前坐标-->绘图描点坐标 
- 4.预期结果
    输出定位坐标，并且生成无人机坐标图
- 5.实际结果
    与预期一致
- 6.原因分析
- 7.下一步工作
    测试代码与其他方向联调