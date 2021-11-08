"""
Created on Tue Oct 26 22:37:07 2021

@author: Elvin Flores
"""

#LIBRERIAS DEL PROGRAMA AUDIO

from scipy.io import wavfile
from matplotlib import pyplot as plt
from winsound import*
import numpy as np
import time
import pyaudio
import wave
#muestreo1: Velocidad de muestreo
# cancion: Señal de canción
muestreo1,Cancion=wavfile.read('Audios/Senal.wav') # Lectura de señal
t1=np.arange(len(Cancion))/float(muestreo1)
cancion=Cancion/(2.**15) # Normalización 2 a la 15

muestreo2,Ruido=wavfile.read('Audios/Ruido_lab.wav') # Lectura de señal
t1=np.arange(len(Ruido))/float(muestreo2)
ruido=Ruido/(2.**15) # Normalización 2 a la 15

target=cancion+ruido
tar=target*(2.**15)
tar=np.array(tar,dtype=np.int16) # Se convierte out (cadena) a un array
wavfile.write('Audios/Target.wav',muestreo1,tar)

# PlaySound(r'Audios/Target.wav',SND_FILENAME|SND_ASYNC)

# wavfile.write('Target.wav',muestreo1,Target*(2.**5))

time.sleep(30)

# PlaySound(r'Target.wav',SND_FILENAME|SND_ASYNC)
#========== PRIMER RED NEURONAL ==========#
w=[-0.5,0.5,0.5] # Vector de valores alatrorios (1 Neurona-3 Entradas) (Sonido, Delay, Delay de Delay)
b=-0.3 # Valor de polarización (aleatorio)

patron=np.zeros((3,1))

out_1=np.zeros((len(ruido),1))

alfa=0.1 # Razon de aprendizaje

for t in range(len(ruido)): # Todo el tiempo de duración de la señal
    if(t==0):
        patron[0]=ruido[t]
    elif(t==1):
        patron[0]=ruido[t]
        patron[1]=ruido[t-1] # Delay
    else:
        patron[0]=ruido[t]
        patron[1]=ruido[t-1] # Delay
        patron[2]=ruido[t-2] # Delay del Delay
    a=np.dot(w,patron)+b # Salida de la neurona
    error=target[t]-a # Calculo de error
    out_1[t]=error # ALmacenamiento de error
    w=w+(2*alfa*error*patron.T)
    b=b+(2*alfa*error) # Neurona comienza a aprender

out_1=out_1*(2.**15)

out_1=np.array(out_1,dtype=np.int16) # Se convierte out (cadena) a un array
wavfile.write('Audios/salida_Red1.wav',muestreo1,out_1)  
# PlaySound(r'salida_Red1.wav',SND_FILENAME|SND_ASYNC)
#========== SEGUNDA RED NEURONAL ==========#
w=[-0.5,0.5,0.5]
b=-0.3 # Valor de polarización (aleatorio)

patron=np.zeros((3,1))

out_2=np.zeros((len(ruido),1))
alfa=0.1 # Razon de aprendizaje
for t in range(len(ruido)): # Todo el tiempo de duración de la señal
    if(t==0):
        patron[0]=ruido[t]
    elif(t==1):
        patron[0]=ruido[t]
        patron[1]=out_2[t-1] # Delay
    else:
        patron[0]=ruido[t]
        patron[1]=out_2[t-1] # Delay
        patron[2]=out_2[t-2] # Delay del Delay
    a=np.dot(w,patron)+b # Salida de la neurona
    error=target[t]-a # Calculo de error
    out_2[t]=error # ALmacenamiento de error
    w=w+(2*alfa*error*patron.T)
    b=b+(2*alfa*error) # Neurona comienza a aprender

out_2=out_2*(2.**15)
out_2=np.array(out_2,dtype=np.int16) # Se convierte out (cadena) a un array
wavfile.write('Audios/salida_Red2.wav',muestreo1,out_2)  
# PlaySound(r'salida_Red2.wav',SND_FILENAME|SND_ASYNC)

fig, axs = plt.subplots(4)
axs[0].plot(Cancion)
axs[0].set_title('Señal de la canción')
axs[0].axis('off')
axs[1].plot(target, 'tab:orange')
axs[1].set_title('Señal de la Canción con ruido')
axs[1].axis('off')
axs[2].plot(out_1, 'tab:green')
axs[2].set_title('Resultado Red 1')
axs[2].axis('off')
axs[3].plot(out_2, 'tab:red')
axs[3].set_title('Resultado Red 2')
axs[3].axis('off')

