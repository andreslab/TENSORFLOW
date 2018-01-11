import tensorflow as tf
import numpy as np

#tensor lleno de zeros
#zero_tsr = tf.zeros([row_dim, col_dim])

#tensor lleno de unos
#ones_tsr = tf.ones([row_dim, col_dim]) 

#tensor lleno constantes
#filled_tsr = tf.fill([row_dim, col_dim])

#tensor de una constante existente
#cosntant_tsr = tf.constant([1,2,3])

#tensores linspace
#linear_tsr = tf.linspace(start=0, stop=1, start=3)
#result: [0.0, 0.5, 1.0]

#tensores con range
#integer_seq_tsr = tf.range(start=6, limit= 15, delta=3)
#print(integer_seq_tsr)
#result: [6,9,12]

#numeros aleatorios
#randunif_tsr = tf.random_uniform([row_dim, col_dim], minval=0, maxval=None1)

#desordenar valores del tensor
#shuffle_output = tf.random_shuffle(input_tensor)

#recortar tensor
#cropped_output = tf.random_crop(input_tensor, crop_size)

#__________________________________________________


#encapsulamos un tensor en una variable
#myvar = tf.Variable(tf.zeros([row_dim, col_dim])

#podemos convertir cualquier matriz de numpy a un tensor con la funcion
#convert_to_tensor()

#las vairblaes son parametros del algoritmo y tensorflow las puede manipular para optimizar un algoritmo
#los placeholder son objetos que le permiten alimentar datos de un tipo y forma y dependen de los resutlados del gráfico computacional

#declarar e inicializar una variable

"""myvar = tf.Variable(tf.zeros([2,3]))
sess = tf.Session()
initializa_op = tf.global_variables_initializer()
sess.run(initializa_op)"""

#los placeholder obtinene sus datos de un argumento feed_dict en la sesion

"""sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[2,2])
y = tf.identity(x)
x_vals = np.random.rand(2,2)
sess.run(y, feed_dict={x:x_vals})"""
#note that sess.run(x, feed_dist={x:x_vals}) will result in a self referencing error


#tenemos que inficarle a tensorflow cuando inicializar las variables.
#existe uns funcion general que inicializa todas global_variables_iniializer()


#initializer_op= tf.global_variables_initializer()

#inicilizar solo una variable
"""
sess = tf.Session()
first_var = tf.Variable(tf.zeros([2,3]))
sess.run(first_var.initializer)
#inicializar una variable despues de otra
second_var = tf.Variable(tf.zeros_like(first_var))
#depende de la first_var
sess.run(second_var.initializer)"""


"""
#operaciones con matrices
sess = tf.Session()
#podemos crear una matriz diagonal a partir de una matriz o lista unidimensional con diag()

identity_matrix = tf.diag([1.0,1.0,1.0])
A = tf.truncated_normal([2,3])
B = tf.fill([2,3], 5.0)
C = tf.random_uniform([3,2])
D = tf.convert_to_tensor(np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]))
print(sess.run(identity_matrix))
print(sess.run(A))
print(sess.run(B))
print(sess.run(C))
print(sess.run(D))

#sumar o restar 
print("SUMA")
print(sess.run(A+B))
print("RESTA")
print(sess.run(B-B))
#multiplicacion
print("MULTIPLICACION")
print(sess.run(tf.matmul(B, identity_matrix)))
#transponer los argumentos
print("TRANSPONER ARGUMENTOS")
print(sess.run(tf.transpose(C)))
#determinante
print("MATRIZ DETERMINANTE")
print(sess.run(tf.matrix_determinant(D)))
#inversa
print("INVERSA")
print(sess.run(tf.matrix_inverse(D)))
#descomposicion de Cholesky
print(sess.run(tf.cholesky(identity_matrix)))
#para valores y vectores propios use, genera autovalores en la primer fila y los vectores seguientes en  los vectores restantes
print(sess.run(tf.self_adjoint_eig(D)))
"""

#estas han sido opraciones matriciales que le agregamos a al grafico


#______________________________________________________

sess = tf.Session()

#print(sess.run(tf.div(3,4))) #division con resultados enteros

#print(sess.run(tf.truediv(3,4))) #division con resultados float

#division de de floatantes con resultado en entero
#print(sess.run(tf.floordiv(22.0, 5.0))) #4.0

#devuelve el residuo en flotante al dividir flotantes
#print(sess.run(tf.mod(22.0, 5.0))) #2.0

#producto cruzado entre 2 tensores con cross(), solo sirve para vectores tridimensionales
#print(sess.run(tf.cross([1., 0., 0.], [0., 1.,0.])))

#funciones extras
"""
abs() Absolute value of one input tensor
ceil() Ceiling function of one input tensor
cos() Cosine function of one input tensor
exp() Base e exponential of one input tensor
floor() Floor function of one input tensor
inv() Multiplicative inverse (1/x) of one input tensor
log() Natural logarithm of one input tensor
maximum() Element-wise max of two tensors
minimum() Element-wise min of two tensors
neg() Negative of one input tensor
pow() The first tensor raised to the second tensor element-wise
round() Rounds one input tensor
rsqrt() One over the square root of one tensor
sign() Returns -1, 0, or 1, depending on the sign of the tensor
sin() Sine function of one input tensor
sqrt() Square root of one input tensor
square() Square of one input tensor
"""

#funciones especiales

"""
digamma() Psi function, the derivative of the lgamma() function
erf() Gaussian error function, element-wise, of one tensor
erfc() Complimentary error function of one tensor
igamma() Lower regularized incomplete gamma function
igammac() Upper regularized incomplete gamma function
lbeta() Natural logarithm of the absolute value of the beta function
lgamma() Natural logarithm of the absolute value of the gamma function
squared_difference() Computes the square of the differences between two tensors
"""

#funcion tangente (tax(pi/4)=1)
print(sess.run(tf.div(tf.sin(3.1416/4.), tf.cos(3.1416/4.))))

#agregamos funciones personalizadas: 3x^2 - x + 10
def custom_polynomial(value):
    return(tf.subtract(3 * tf.square(value), value) + 10)
print(sess.run(custom_polynomial(11)))


#________________________________________________________________________
#Implementing Activation Functions pag 16

