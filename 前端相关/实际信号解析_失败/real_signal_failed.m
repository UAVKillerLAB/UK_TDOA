SampleFre           = 2e8;                  %单位：Hz  采样频率
SampleLen           = 2e7;                  %采样点数 = 2e8(采样频率) * 5e-9s*2e7(采样时长t=采样周期*原始信号数据个数) = 2e7
%%
%本版本无法实现频谱分析功能

%备注：得到的信号无人机为解调过后的信号，最高频率只有几十Mhz-->采样频率设定为200Mhz
%bin文件内为AD9361采集的原始信号，T=5ns=5e-9,一共有Number=2e7个数据,遂采样时间t =T*Number
%%
%以大端方式读取指定路径的bin文件
fid=fopen('信号.bin','r+');
signal = fread(fid,'int8','l');
%以int16的格式将数据存放在列向量中
fclose(fid);%关闭文件             

t=0:1/SampleFre:(SampleLen-1)/SampleFre;     %时间t
figure;plot(t,signal)
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

