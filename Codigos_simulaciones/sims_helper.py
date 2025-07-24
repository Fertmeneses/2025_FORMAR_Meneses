import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from PIL import Image
from scipy.ndimage import rotate


def crear_AAO(
    M, 
    N, 
    r=0,
    dcc=1,
    rot_deg=0, 
    forma='circ',
    orden_red='hexa',
    seed=0,
    ruido=0,
    dcc_desorden=0,
    r_desorden=0,
    save_path=None
    ):
    """
    Crea una matriz MxN que simula una red 2D de poros regularmente distribuidos sobre una estructura cristalina.
    Cada poro puede ser circular o cuadrado, con un tamaño promedio {r} y un espaciado {dcc} respecto al centro de sus vecinos más cercanos. 
    Se permite un grado de desorden en el tamaño y la posición de los poros. La red puede ser de tipo cuadrada o hexagonal, y se puede aplicar una rotación global.
    Opcionalmente se puede agregar ruido y exportar la matriz como imagen. Los poros se representan con valor 0 y la alúmina con valor 255.

    --- Entrada ---
    {M} [int>0]: Número de píxeles en dirección X (filas) [píxeles].
    {N} [int>0]: Número de píxeles en dirección Y (columnas) [píxeles].
    {r} [int>=0]: Radio del poro circular, o lado del poro cuadrado [píxeles].
    {dcc} [int>0]: Distancia entre centros de poros vecinos [píxeles].
    {rot_deg} [float]: Ángulo de rotación global de la red [grados].
    {forma} [string]: Forma del poro, elegir entre 'circ' para círculo o 'cuad' para cuadrado.
    {orden_red} [string]: Orden de la red cristalina, elegir entre 'cuad' para red cuadrada o 'hexa' para red hexagonal.
    {seed} [int]: Semilla para iniciación aleatoria de la matriz.
    {ruido} [float>=0]: Intensidad del ruido aleatorio.
    {dcc_desorden} [int>=0]: Variación aleatoria máxima del parámetro {dcc} [píxeles]. 
    Nota: el valor máximo de {dcc_desorden} es {dcc}, cualquier valor superior será convertido a {dcc}.
    {r_desorden} [int>=0]: Variación aleatoria máxima del parámetro {r} [píxeles]. 
    Nota: el valor máximo de {r_desorden} es {r}, cualquier valor superior será convertido a {r}.
    {save_path} [string o None]: Provee el directorio para guardar la matriz simulada como imagen,
    en formato 'jpg'. Si {save_path}=None, entonces la imagen no se guarda. 

    --- Salida | Variables ---
    {AAO} [numpy.ndarray]: Matriz MxN con poros distribuidos regularmente con valores de pixeles entre 0 y 255. 

    --- Salida | Archivos ---
    [Opcional] Exporta una imagen '.jpg' de la matriz {AAO}.
    """
    np.random.seed(seed) # Semilla aleatoria
    
    # Inicializar matriz agrandada para realizar un dibujo simétrico:
    larger_M = 5*M  # Alargar M temporalmente
    larger_N = 5*N  # Alargar N temporalmente
    AAO = np.zeros((larger_M,larger_N),dtype=int) # Matriz AAO inicializada sin poros

    # Chequear parámetros de desorden, ajustar si es necesario:
    r_desorden = min(r_desorden,r) # Variación aleatoria máxima del radio [píxeles]
    dcc_desorden = min(dcc_desorden,dcc) # Variación aleatoria máxima de la distancia entre poros [píxeles] 

    # Chequear que hacer que dcc sea almenos 2*r, ajustar si es necesario:
    dcc = max(dcc+1,2*r+2) 
    
    # Chequeo para evitar solapamientos:
    dx_min = dcc-1-2*dcc_desorden   # distancia mínima entre centros con desorden
    diam_max = 2*(r + r_desorden)  # diámetro máximo posible
    if diam_max >= dx_min:
        print('⚠️ Warning: Poros potencialmente solapados. Reducir r_desorden o dcc_desorden.')

    # Asigna el orden de la red:    
    dx, dy, f = asignar_orden_red(dcc,orden_red)
    
    # Rellenar píxeles correspondientes a poros:  
    for cx in np.arange(M,larger_M-M,dx).astype(int):
        for i,cy in enumerate(np.arange(N,larger_N-N,dy).astype(int)):
            # Asigna las distancias entre poros (puede incluir variaciones aleatorias):
            cx_ = cx+f(i)+np.random.randint(-dcc_desorden,dcc_desorden+1) # Distancia X distorsionada [píxeles]
            cy_ = cy+np.random.randint(-dcc_desorden,dcc_desorden+1) # Distancia Y distorsionada [píxeles]
            # Asigna el radio del poro (puede incluir variaciones aleatorias):
            R = r+np.random.randint(-r_desorden,r_desorden+1) # Radio distorsionado [píxeles] 
            # Para cada píxel de la fila x, calcular la altura del poro y actualizar AAO:
            for x in range(-R, R+1):    
                # Determinar altura del poro:
                y_max = determinar_y_max(R,x,forma) # [píxeles] 
                # Actualizar matriz:
                if y_max==0:
                    AAO[cx_+x,cy_] = 1
                else:
                    AAO[x+cx_,-y_max+cy_:y_max+cy_+1] = 1
                    
    # Rotar la matriz completa (antes del recorte):
    AAO = rotate(AAO, angle=rot_deg, reshape=False, order=3, mode='nearest')                
                    
    # Recortar matriz al tamaño original, con corte aleatorio:
    v_i = np.random.randint((2*M-dcc-r),2*M) # Posición X del primer píxel
    v_j = np.random.randint((2*N-dcc-r),2*N) # Posición Y del primer píxel
    AAO = AAO[v_i:M+v_i,v_j:N+v_j].astype(float) # Matrix recortada (MxN)

    # Agrega ruido si es necesario:
    if ruido>0:
        matriz_ruido = ruido/2-np.random.random((M,N))*ruido # Genera la matriz de ruido
        AAO += matriz_ruido # Actualiza la matriz
        AAO = np.clip(AAO,0,1)

    # Determinar porosidad, fracción entre 0 (todo alúmina) y 1 (todo poros):
    P = np.sum(AAO)/AAO.size

    # Asigno valores de píxeles entre 0 y 255:
    AAO = np.clip((1-AAO)*255,0,255)
    
    # Guarda la matriz AAO si es requerido:
    if save_path is not None:
        # Generar el nombre del archivo:
        nombre = generar_nombre_archivo_AAO(M,N,r,dcc,P,forma,orden_red,rot_deg,seed,ruido,r_desorden,dcc_desorden)        
        # Prepara la imagen para guardar en formato de escala de grises 0-255:
        im = Image.fromarray(np.uint8(AAO), mode='L') # Generar imagen
        # Guarda la imagen:
        im.save(save_path+nombre+".jpg")

    return AAO

def mostrar_AAO(
    AAO,
    cmap='grey',
    figsize=(4,4)
    ):
    """
    Grafica una matriz AAO de dimensiones MxN arbitrarias.
    Rango entre 0 y 255, representando poros y alumina respectivamente.

    --- Entrada ---
    {AAO} [Numpy array]: Matriz MxN en la cual cada elemento representa un
    valor para el píxel.
    {cmap} [Matplotlib color]: Mapa de color para los píxeles.
    {figsize} [Tupla]: Valores de ancho y alto para el tamaño de la figura

    --- Salida | Gráficos ---
    Matriz MxN graficada con la función matplotlib.pyplot.imshow.
    """   
    # Crear figura y graficar:    
    fig = plt.figure(figsize=figsize) # Crear figura (objeto)
    ax = plt.imshow(AAO,cmap=cmap, vmin=0, vmax=255) # Graficar   
    
    # Título y ejes:
    plt.title(f'Matrix {AAO.shape[0]}x{AAO.shape[1]}')
    plt.xlabel('Distancia X [px]')
    plt.ylabel('Distancia Y [px]')
    plt.show()   

def asigna_brillo_contraste(AAO, orden_bc, B=0, C=0):
    """
    Modifica los valores de píxeles de la matriz AAO aplicando transformaciones
    de brillo (suma) y contraste (multiplicación) en el orden especificado.

    --- Entrada ---
    {AAO} [numpy.ndarray]: Matriz MxN con poros distribuidos regularmente con valores de pixeles entre 0 y 255.
    {orden_bc} [String]: Orden en el que se aplica el brillo y el contraste. Elegir entre 'BC', 'CB', 'B' o 'C'.
    {B} [float]: Brillo. Elegir valor entre -255 y 255.
    {C} [float]: Contraste. Valores negativos y cero generan una imagen sin contraste. 

    --- Salida ---
    {AAO} [numpy.ndarray]: Matriz AAO con transformaciones de brillo y/o contraste aplicados.
    
    """
    if orden_bc == 'BC':
        AAO = np.clip(AAO+B,0,255)
        AAO = np.clip(AAO*C,0,255)
    elif orden_bc == 'CB':
        AAO = np.clip(AAO*C,0,255)
        AAO = np.clip(AAO+B,0,255)
    elif orden_bc == 'B':
        AAO = np.clip(AAO+B,0,255)
    elif orden_bc == 'C':
        AAO = np.clip(AAO*C,0,255)
    else:
        raise ValueError(f'El parámetro orden_bc={orden_bc} es incorrecto. Elegir entre "BC", "CB", "B" o "C". ')

    return AAO

def asignar_orden_red(dcc,orden_red):
    """
    Asigna el orden de la red de poros y determina los parámetros geométricos.

    --- Entrada ---
    {dcc} [int>=0]: Distancia centro a centro poros más cercanos [píxeles].
    {orden_red} [string]: Orden de la red cristalina, elegir entre 'cuad' para red cuadrada o 'hexa' para red hexagonal.

    --- Salida ---
    {dx} [int]: Distancia en X entre centros de poros vecinos [píxeles].
    {dy} [int]: Distancia en Y entre centros de poros vecinos [píxeles].
    {f} [function]: Función que desplaza los poros en X en las filas pares para la red hexagonal.
    """   
    # Definimos los desplazamientos en X e Y, y la función según el orden de la red cristalina:
    if orden_red=='cuad':
        dx = dcc 
        dy = dcc 
        f = lambda i: 0 
    elif orden_red=='hexa':
        dx = dcc
        dy = int(round(np.sqrt(3)/2 * dcc))  
        f = lambda i: int(i % 2 == 0)*int(dx/2) 
    else: 
        raise ValueError(f'El parámetro orden_red={orden_red} ingresado es incorrecto. Elegir entre "cuad" o "hexa".')

    return dx, dy, f

def determinar_y_max(R,x,forma):
    """
    Determina el valor máximo de Y para un radio R y una coordenada X dada, según la forma del poro.

    --- Entrada ---
    {R} [int>=0]: Radio del poro circular, o lado del poro cuadrado [píxeles].
    {x} [int]: Coordenada X del poro [píxeles].
    {forma} [string]: Forma del poro, elegir entre 'circ' para círculo o 'cuad' para cuadrado.

    --- Salida ---
    {y_max} [int]: Valor máximo en Y para el poro en la coordenada X, según la forma especificada [píxeles].
    """
    if forma=='circ':
        y_max = int(round(np.sqrt(R**2-x**2))) 
    elif forma=='cuad':
        y_max = R
    else:
        raise ValueError(f'El parámetro forma={forma} ingresado es incorrecto. Elegir entre "circ" o "cuad".')
    return y_max

def generar_nombre_archivo_AAO(
    M,N,r,dcc,P,forma,orden_red,rot_deg,seed,ruido,r_desorden,dcc_desorden):
    """
    Genera un nombre de archivo para guardar la matriz AAO, incluyendo todos los parámetros relevantes.

    --- Entrada ---
    {M} [int>0]: Número de píxeles en dirección X (filas) [píxeles].
    {N} [int>0]: Número de píxeles en dirección Y (columnas) [píxeles].
    {r} [int>=0]: Radio del poro circular, o lado del poro cuadrado [píxeles].
    {dcc} [int>=0]: Distancia fija entre centros de poros más cercanos [píxeles].
    {P} [float]: Porosidad de la muestra. Fracción entre 0 (todo alúmina) y 1 (todo poros).
    {forma} [string]: Forma del poro, elegir entre 'circ' para círculo o 'cuad' para cuadrado.    
    {orden_red} [string]: Orden de la red cristalina, elegir entre 'cuad' para red cuadrada o 'hexa' para red hexagonal.
    {rot_deg} [float]: Ángulo de rotación global de la red [grados].
    {seed} [int]: Semilla para iniciación aleatoria de la matriz.
    {ruido} [float>=0]: Intensidad del ruido aleatorio.
    {r_desorden} [int>=0]: Variación aleatoria máxima del parámetro {r} [píxeles].
    {dcc_desorden} [int>=0]: Variación aleatoria máxima del parámetro {dcc} [píxeles]. 

    --- Salida ---
    {nombre} [String]: nombre del archivo con la matriz AAO.
    """
    # Generar el nombre del archivo:
    nombre = f'Sim_AAO{M}x{N}_r{r}_c{dcc}_P{P:.2f}_poro{forma}_orden{orden_red}' # Parámetros básicos
    if rot_deg:
        nombre += f'_rot{rot_deg:.0f}' # Agrega el parámetro {rot_deg} si es distinto de 0
    if ruido:
        nombre += f'_ruido{ruido:.2f}' # Agrega el parámetro {ruido} si es distinto de 0
    if r_desorden:
        nombre += f'_rdesorden{r_desorden:.0f}' # Agrega el parámetro {r_desorden} si es distinto de 0
    if dcc_desorden:
        nombre += f'_cdesorden{dcc_desorden:.0f}' # Agrega el parámetro {c_desorden} si es distinto de 0
    nombre += f'_seed{seed}'

    return nombre