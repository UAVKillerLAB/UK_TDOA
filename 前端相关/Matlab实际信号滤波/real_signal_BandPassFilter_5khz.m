SampleFre           = 2e8;                  %��λ��Hz  ����Ƶ��   
SampleLen           = 1e7;                  %�������� 
%%
%��ע���õ����ź����˻�Ϊ���������źţ���ҪƵ��Ϊ8Mhz-25Mhz -->����Ƶ���趨Ϊ200Mhz
%%
%�Դ�˷�ʽ��ȡָ��·����bin�ļ�
fid=fopen('�ź�.bin','r+');
signal = fread(fid,'int16','l');            %�˴�����������ҪΪint16��Ϊint8�����
%��int16�ĸ�ʽ�����ݴ������������
fclose(fid);%�ر��ļ�             
% %%
t=0:1/SampleFre:(SampleLen-1)/SampleFre;     %ʱ��t
figure;plot(t,signal(1:length(t)))
title('���˻��ź�ʱ����')
% 
% %%
% %-------------------�Ի����źŽ���Ƶ�׷���
FFT_Data = fft(signal);                     %��ԭʼ�ź�signal����Ƶ�׷���
Amplitude = abs(FFT_Data);
Amplitude = Amplitude/length(Amplitude);
Amplitude(2:end) = 2*Amplitude(2:end);
Frequence = (1:(length(t)/2))/length(t)*SampleFre;
figure;plot(Frequence,Amplitude(1:length(Frequence)))
title('���˻��ź��ź�Ƶ��')

%%
%-------------------��ͨ�˲�,�˴�ȡ������һ������Ϊ5Khz�����ز�  

%���õ���FIR�˲�����ͨ��Ƶ��Ϊ10Mhz--10.01Mhz
%��ԭʼ�ź�SignalData �����˲��������Ϊ�����˲����ź�BPF_Data
                                                                           %����˵��:
bpFilt = designfilt('bandpassfir','FilterOrder',1300, ...                 %������FilterOrder-->10000 ;10000��,����Խ�ߣ�Ч��Խ�ã�����ʱ��Խ��
         'CutoffFrequency1',10e6,'CutoffFrequency2',10.01e6, ...           %����ֹƵ�ʣ�CutoffFrequency1-->10Mhz
         'SampleRate',2e8);                                                %�Ҳ��ֹƵ�ʣ�CutoffFrequency1-->10.01Mhz
% fvtool(bpFilt)                                                             %����Ƶ��Ϊ:SampleRate-->200Mhz
BPF_Data = filter(bpFilt,signal);

BPF_Data1 = filter(bpFilt,signal);
BPF_Data2 = filter(bpFilt,signal);                                                                            
% BPF1=load('untitled1.mat');                           %���õ���FIR�˲�����ͨ��Ƶ��Ϊ2400Khz--2405Khz
% BPF_Data = filter(BPF1.Num1,1,signal);       %��2.4M�ź�SignalData �����˲��������Ϊ�����˲����ź�BPF_Data
% figure;plot(t,BPF_Data)                         
% title('��ͨ�˲�֮��Ĳ���')
                                                                          

%%
%-------------------��ͨ�˲�֮��Ƶ�׷���   
FFT_BPF_Data = fft(BPF_Data);                     %�������˲����ź�BPF_Data����Ƶ�׷���
Amplitude_BPF = abs(FFT_BPF_Data);                                     
Amplitude_BPF = Amplitude_BPF/length(Amplitude_BPF);        
Amplitude_BPF(2:end) = 2*Amplitude_BPF(2:end);              
Frequence = (0:(length(Amplitude_BPF)/2-1))/length(Amplitude_BPF)*SampleFre;
figure;plot(Frequence,Amplitude_BPF(1:length(Frequence)))
title('��ͨ�˲�֮���Ƶ��')


FFT_BPF_Data = fft(BPF_Data1);                     %�������˲����ź�BPF_Data����Ƶ�׷���
Amplitude_BPF = abs(FFT_BPF_Data);                                     
Amplitude_BPF = Amplitude_BPF/length(Amplitude_BPF);        
Amplitude_BPF(2:end) = 2*Amplitude_BPF(2:end);              
Frequence = (0:(length(Amplitude_BPF)/2-1))/length(Amplitude_BPF)*SampleFre;
figure;plot(Frequence,Amplitude_BPF(1:length(Frequence)))
title('��ͨ�˲�֮���Ƶ��')


FFT_BPF_Data = fft(BPF_Data2);                     %�������˲����ź�BPF_Data����Ƶ�׷���
Amplitude_BPF = abs(FFT_BPF_Data);                                     
Amplitude_BPF = Amplitude_BPF/length(Amplitude_BPF);        
Amplitude_BPF(2:end) = 2*Amplitude_BPF(2:end);              
Frequence = (0:(length(Amplitude_BPF)/2-1))/length(Amplitude_BPF)*SampleFre;
figure;plot(Frequence,Amplitude_BPF(1:length(Frequence)))
title('��ͨ�˲�֮���Ƶ��')



% %test
% bpFilt1 = designfilt('bandpassfir','FilterOrder',1300, ...                 %������FilterOrder-->10000 ;10000��,����Խ�ߣ�Ч��Խ�ã�����ʱ��Խ��
%          'CutoffFrequency1',10.02e6,'CutoffFrequency2',10.03e6, ...           %����ֹƵ�ʣ�CutoffFrequency1-->10Mhz
%          'SampleRate',2e8);                                                %�Ҳ��ֹƵ�ʣ�CutoffFrequency1-->10.01Mhz
% % fvtool(bpFilt)                                                             %����Ƶ��Ϊ:SampleRate-->200Mhz
% BPF_Data1 = filter(bpFilt1,signal);   
% %%
% %-------------------��ͨ�˲�֮��Ƶ�׷���   
% FFT_BPF_Data = fft(BPF_Data1);                     %�������˲����ź�BPF_Data����Ƶ�׷���
% Amplitude_BPF = abs(FFT_BPF_Data);                                     
% Amplitude_BPF = Amplitude_BPF/length(Amplitude_BPF);        
% Amplitude_BPF(2:end) = 2*Amplitude_BPF(2:end);              
% Frequence = (0:(length(Amplitude_BPF)/2-1))/length(Amplitude_BPF)*SampleFre;
% figure;plot(Frequence,Amplitude_BPF(1:length(Frequence)))
% title('��ͨ�˲�֮���Ƶ��2')


% %test
% bpFilt2 = designfilt('bandpassfir','FilterOrder',1300, ...                 %������FilterOrder-->10000 ;10000��,����Խ�ߣ�Ч��Խ�ã�����ʱ��Խ��
%          'CutoffFrequency1',10.04e6,'CutoffFrequency2',10.05e6, ...           %����ֹƵ�ʣ�CutoffFrequency1-->10Mhz
%          'SampleRate',2e8);                                                %�Ҳ��ֹƵ�ʣ�CutoffFrequency1-->10.01Mhz
% % fvtool(bpFilt)                                                             %����Ƶ��Ϊ:SampleRate-->200Mhz
% BPF_Data2 = filter(bpFilt2,signal);   
% %%
% %-------------------��ͨ�˲�֮��Ƶ�׷���   
% FFT_BPF_Data = fft(BPF_Data2);                     %�������˲����ź�BPF_Data����Ƶ�׷���
% Amplitude_BPF = abs(FFT_BPF_Data);                                     
% Amplitude_BPF = Amplitude_BPF/length(Amplitude_BPF);        
% Amplitude_BPF(2:end) = 2*Amplitude_BPF(2:end);              
% Frequence = (0:(length(Amplitude_BPF)/2-1))/length(Amplitude_BPF)*SampleFre;
% figure;plot(Frequence,Amplitude_BPF(1:length(Frequence)))
% title('��ͨ�˲�֮���Ƶ��3')