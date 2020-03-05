SampleFre           = 2e8;                  %单位：Hz  采样频率   
SampleLen           = 1e7;                  %采样点数 
%%
%备注：得到的信号无人机为解调过后的信号，主要频率为8Mhz-25Mhz -->采样频率设定为200Mhz
%%
%以大端方式读取指定路径的bin文件
fid=fopen('信号.bin','r+');
signal = fread(fid,'int16','l');            %此处数据类型需要为int16，为int8会出错
%以int16的格式将数据存放在列向量中
fclose(fid);%关闭文件             

t=0:1/SampleFre:(SampleLen-1)/SampleFre;     %时间t
figure;plot(t,signal(1:length(t)))
title('无人机信号时间谱')

%%
%-------------------对基带信号进行频谱分析
FFT_Data = fft(signal);                     %将原始信号signal进行频谱分析
Amplitude = abs(FFT_Data);
Amplitude = Amplitude/length(Amplitude);
Amplitude(2:end) = 2*Amplitude(2:end);
Frequence = (1:(length(t)/2))/length(t)*SampleFre;
figure;plot(Frequence,Amplitude(1:length(Frequence)))
title('无人机信号信号频谱')

%%
%-------------------带通滤波,此处取样的是一个带宽为5Khz的子载波  

%采用的是FIR滤波器，通过频段为10Mhz--10.01Mhz
%将原始信号SignalData 输入滤波器，输出为经过滤波的信号BPF_Data
                                                                           %参数说明:
bpFilt = designfilt('bandpassfir','FilterOrder',10000, ...                 %阶数：FilterOrder-->10000 ;10000阶,阶数越高，效果越好，计算时间越长
         'CutoffFrequency1',10e6,'CutoffFrequency2',10.01e6, ...           %左侧截止频率：CutoffFrequency1-->10Mhz
         'SampleRate',2e8);                                                %右侧截止频率：CutoffFrequency1-->10.01Mhz
fvtool(bpFilt)                                                             %采样频率为:SampleRate-->200Mhz
BPF_Data = filter(bpFilt,signal);                                          

figure;plot(t,BPF_Data)                         
title('带通滤波之后的波形')
%%
%-------------------带通滤波之后频谱分析   
FFT_BPF_Data = fft(BPF_Data);                     %将经过滤波的信号BPF_Data进行频谱分析
Amplitude_BPF = abs(FFT_BPF_Data);                                     
Amplitude_BPF = Amplitude_BPF/length(Amplitude_BPF);        
Amplitude_BPF(2:end) = 2*Amplitude_BPF(2:end);              
Frequence = (0:(length(Amplitude_BPF)/2-1))/length(Amplitude_BPF)*SampleFre;
figure;plot(Frequence,Amplitude_BPF(1:length(Frequence)))
title('带通滤波之后的频谱')
