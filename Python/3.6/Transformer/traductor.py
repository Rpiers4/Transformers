#https://www.youtube.com/watch?v=p2sTJYoIwj0

#from subprocess import check_output
#check_output("pip install tensorflow", shell=True)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
from keras_transformer import get_model, decode
from pickle import load
np.random.seed(0)

def construye_diccionario(token_list):
    token_dict = {
        '<PAD>' : 0,
        '<START>' : 1,
        '<END>' : 2
    }
    for tokens in token_list:
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    return token_dict

def traductor3000(frase):
    tokens_frase= [tokens + ['<END>','<PAD>'] for tokens in [frase.split(' ')]]
    tr_entrada = [list(map(lambda x: diccionario_entrada[x], tokens)) for tokens in tokens_frase][0]
    salida_decodificada = decode(
        modelo,
        tr_entrada,
        start_token = diccionario_salida['<START>'],
        end_token = diccionario_salida['<END>'],
        pad_token = diccionario_salida['<PAD>']
    )
    print('Frase original: {}'.format(frase))
    print('Traduccion: {}'.format(' '.join(map(lambda x: diccionario_salida_inverso[x], salida_decodificada[1:-1]))))
   

    
#Leer set te de entrenamiento.

filename='english-spanish.pkl'
dataset = load(open(filename,'rb'))
#print(dataset[120000,0])
#print(dataset[120000,1])

#Tokenizar
tokens_entrada=[]
for sentence in dataset[:,0]:
    tokens_entrada.append(sentence.split(' '))
#print(tokens_entrada[120000])

tokens_salida=[]
for sentence in dataset[:,1]:
    tokens_salida.append(sentence.split(' '))
#print(tokens_salida[120000])

diccionario_entrada = construye_diccionario(tokens_entrada)
diccionario_salida = construye_diccionario(tokens_salida)
diccionario_salida_inverso = {v:k for k,v in diccionario_salida.items()}

#print('diccionario_entrada',len(diccionario_entrada))
#print('diccionario_salida',len(diccionario_salida))
#print('diccionario_salida_inverso',len(diccionario_salida_inverso))


#Agregar START, END  y PAD a cada prase del set de entrenamiento

tokens_codificador = [['<START>'] + tokens + ['<END>'] for tokens in tokens_entrada]
tokens_decodificador = [['<START>'] + tokens + ['<END>'] for tokens in tokens_salida]
tokens_resultado = [tokens + ['<END>'] for tokens in tokens_salida]
#buscamos la longitud max de frase
entrada_max_len = max(map(len, tokens_codificador))
salida_max_len = max(map(len, tokens_decodificador))
# añadimos PAD a las frases mas cortas que la frase mas larga.
tokens_codificador = [tokens + ['<PAD>'] * (entrada_max_len-len(tokens)) for tokens in tokens_codificador]
tokens_decodificador = [tokens + ['<PAD>'] * (salida_max_len-len(tokens)) for tokens in tokens_decodificador]
tokens_resultado = [tokens + ['<PAD>'] * (salida_max_len-len(tokens)) for tokens in tokens_resultado]

#print(tokens_codificador[120000])

entrada_codificador = [list(map(lambda x : diccionario_entrada[x], tokens)) for tokens in tokens_codificador]
entrada_decodificador = [list(map(lambda x : diccionario_salida[x], tokens)) for tokens in tokens_decodificador]
salida_decodificador = [list(map(lambda x : diccionario_salida[x], tokens)) for tokens in tokens_resultado]

#print(entrada_codificador[120000])


#Crear la red transformer
modelo = get_model(
#numero de lapalbras que esta utilizando el modelo
    token_num=max(len(diccionario_entrada),len(diccionario_salida)),
    embed_dim= 32,
    encoder_num=2,
    decoder_num=2,
    head_num=4,
    hidden_dim=128,
    dropout_rate = 0.05,
    use_same_embed = False,
)
modelo.compile('adam', 'sparse_categorical_crossentropy')
#modelo.summary()
modelo.load_weights('translator_preentrenado.h5')

#ENTRENAMIENTO:

#Arreglo con las dos entradas codificadas en español e inglés.
x = [np.array(entrada_codificador), np.array(entrada_decodificador)]
#salida.
y = np.array(salida_decodificador)

#modelo.fit(x,y, epochs=15, batch_size=32)


traductor3000('im in mother lover')
