%%
%%-------------------因频段为2.4Ghz采样点数过多，将导致软件运行时间过长，故将频段调整为2.2Mhz，缩短软件运行时间，若需要
Frequence0          = 2.4e6;                %单位：Hz  信号频率   2.4Mhz
SampleFre           = 6e6;                  %单位：Hz  采样频率
SampleLen           = SampleFre;            %采样点数
%%
%--------------------------------------------产生一个2.4Mhz频段的带宽为22Khz的宽带信号
delta_f = 0:2e2:2.2e4;                      %2.4M频段第一个信道的间隔频率    一个信道22Mhz，在这里模拟为22Khz   此处产生了110个[2400Khz--2422Khz]的信号
SignalData = 0;                             %初始信号为0
F = (Frequence0 + delta_f)';                %2.4M基准频率+间隔频率->2.4M频段第一个信道频率的离散值
%-------------------将2.4M频段第一个信道内所有信号的离散值进行累加
for i = 1:length(delta_f)    
t = 0:1/SampleLen:1/SampleFre*(SampleLen-1);    %时间t
SignalData0 = sin(2*pi*F(i,:)*t);               %信道1内的第i个信号  备注:本程序内，信道1有length(delta_f)个信号
SignalData = SignalData + SignalData0;          %将2.4M频段第一个信道内所有信号进行累加   
end     
figure;plot(t(1:150),SignalData(1:150))
title('2.4Mhz信号时间谱')
%%
%-------------------对2.4Mhz频谱分析
FFT_Data = fft(SignalData);                     %将原始信号SignalData进行频谱分析
Amplitude = abs(FFT_Data);
Amplitude = Amplitude/length(Amplitude);
Amplitude(2:end) = 2*Amplitude(2:end);
Frequence = (0:(length(Amplitude)/2-1))/length(Amplitude)*SampleFre;
figure;plot(Frequence,Amplitude(1:length(Frequence)))
title('2.4Mhz信号频谱')

%%
%-------------------带通滤波,此处取样的是一个带宽为5Khz的子载波
BPF1=load('BPF.mat');                           %采用的是FIR滤波器，通过频段为2400Khz--2405Khz
BPF_Data = filter(BPF1.Num,1,SignalData);       %将2.4M信号SignalData 输入滤波器，输出为经过滤波的信号BPF_Data
figure;plot(t,BPF_Data)                         
title('带通滤波之后的波形')
%%
%-------------------带通滤波之后频谱分析   
FFT_BPF_Data = fft(BPF_Data);                   %将经过滤波的信号BPF_Data进行频谱分析
Amplitude_BPF = abs(FFT_BPF_Data);                                     
Amplitude_BPF = Amplitude_BPF/length(Amplitude_BPF);        
Amplitude_BPF(2:end) = 2*Amplitude_BPF(2:end);              
Frequence = (0:(length(Amplitude_BPF)/2-1))/length(Amplitude_BPF)*SampleFre;
figure;plot(Frequence,Amplitude_BPF(1:length(Frequence)))
title('带通滤波之后的频谱')
