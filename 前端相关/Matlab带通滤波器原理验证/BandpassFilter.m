%%
%%-------------------��Ƶ��Ϊ2.4Ghz�����������࣬�������������ʱ��������ʽ�Ƶ�ε���Ϊ2.2Mhz�������������ʱ�䣬����Ҫ
Frequence0          = 2.4e6;                %��λ��Hz  �ź�Ƶ��   2.4Mhz
SampleFre           = 6e6;                  %��λ��Hz  ����Ƶ��
SampleLen           = SampleFre;            %��������
%%
%--------------------------------------------����һ��2.4MhzƵ�εĴ���Ϊ22Khz�Ŀ���ź�
delta_f = 0:2e2:2.2e4;                      %2.4MƵ�ε�һ���ŵ��ļ��Ƶ��    һ���ŵ�22Mhz��������ģ��Ϊ22Khz   �˴�������110��[2400Khz--2422Khz]���ź�
SignalData = 0;                             %��ʼ�ź�Ϊ0
F = (Frequence0 + delta_f)';                %2.4M��׼Ƶ��+���Ƶ��->2.4MƵ�ε�һ���ŵ�Ƶ�ʵ���ɢֵ
%-------------------��2.4MƵ�ε�һ���ŵ��������źŵ���ɢֵ�����ۼ�
for i = 1:length(delta_f)    
t = 0:1/SampleLen:1/SampleFre*(SampleLen-1);    %ʱ��t
SignalData0 = sin(2*pi*F(i,:)*t);               %�ŵ�1�ڵĵ�i���ź�  ��ע:�������ڣ��ŵ�1��length(delta_f)���ź�
SignalData = SignalData + SignalData0;          %��2.4MƵ�ε�һ���ŵ��������źŽ����ۼ�   
end     
figure;plot(t(1:150),SignalData(1:150))
title('2.4Mhz�ź�ʱ����')
%%
%-------------------��2.4MhzƵ�׷���
FFT_Data = fft(SignalData);                     %��ԭʼ�ź�SignalData����Ƶ�׷���
Amplitude = abs(FFT_Data);
Amplitude = Amplitude/length(Amplitude);
Amplitude(2:end) = 2*Amplitude(2:end);
Frequence = (0:(length(Amplitude)/2-1))/length(Amplitude)*SampleFre;
figure;plot(Frequence,Amplitude(1:length(Frequence)))
title('2.4Mhz�ź�Ƶ��')

%%
%-------------------��ͨ�˲�,�˴�ȡ������һ������Ϊ5Khz�����ز�
BPF1=load('BPF.mat');                           %���õ���FIR�˲�����ͨ��Ƶ��Ϊ2400Khz--2405Khz
BPF_Data = filter(BPF1.Num,1,SignalData);       %��2.4M�ź�SignalData �����˲��������Ϊ�����˲����ź�BPF_Data
figure;plot(t,BPF_Data)                         
title('��ͨ�˲�֮��Ĳ���')
%%
%-------------------��ͨ�˲�֮��Ƶ�׷���   
FFT_BPF_Data = fft(BPF_Data);                   %�������˲����ź�BPF_Data����Ƶ�׷���
Amplitude_BPF = abs(FFT_BPF_Data);                                     
Amplitude_BPF = Amplitude_BPF/length(Amplitude_BPF);        
Amplitude_BPF(2:end) = 2*Amplitude_BPF(2:end);              
Frequence = (0:(length(Amplitude_BPF)/2-1))/length(Amplitude_BPF)*SampleFre;
figure;plot(Frequence,Amplitude_BPF(1:length(Frequence)))
title('��ͨ�˲�֮���Ƶ��')
