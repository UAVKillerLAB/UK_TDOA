SampleFre           = 2e8;                  %��λ��Hz  ����Ƶ��
SampleLen           = 2e7;                  %�������� = 2e8(����Ƶ��) * 5e-9s*2e7(����ʱ��t=��������*ԭʼ�ź����ݸ���) = 2e7
%%
%���汾�޷�ʵ��Ƶ�׷�������

%��ע���õ����ź����˻�Ϊ���������źţ����Ƶ��ֻ�м�ʮMhz-->����Ƶ���趨Ϊ200Mhz
%bin�ļ���ΪAD9361�ɼ���ԭʼ�źţ�T=5ns=5e-9,һ����Number=2e7������,�����ʱ��t =T*Number
%%
%�Դ�˷�ʽ��ȡָ��·����bin�ļ�
fid=fopen('�ź�.bin','r+');
signal = fread(fid,'int8','l');
%��int16�ĸ�ʽ�����ݴ������������
fclose(fid);%�ر��ļ�             

t=0:1/SampleFre:(SampleLen-1)/SampleFre;     %ʱ��t
figure;plot(t,signal)
title('���˻��ź�ʱ����')

%%
%-------------------�Ի����źŽ���Ƶ�׷���
FFT_Data = fft(signal);                     %��ԭʼ�ź�signal����Ƶ�׷���
Amplitude = abs(FFT_Data);
Amplitude = Amplitude/length(Amplitude);
Amplitude(2:end) = 2*Amplitude(2:end);
Frequence = (1:(length(t)/2))/length(t)*SampleFre;
figure;plot(Frequence,Amplitude(1:length(Frequence)))
title('���˻��ź��ź�Ƶ��')

