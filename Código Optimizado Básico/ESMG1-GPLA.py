import time 
import numpy
import scipy.integrate
import os
import random
from numba import jit

@jit(nopython = True, parallel = False)
def cargarposicion(nparticulas, longitud_malla, aP, x) -> bool:
    espacio_particulas = longitud_malla/nparticulas
    for i in range(nparticulas):
        x[i] = espacio_particulas*(i + 0.5)
        x[i] += aP*numpy.cos(x[i])
    return True

@jit(nopython = False, parallel = False)
def cargarvelocidad(nparticulas, num_especies, c, v, vh, vth, longitud_malla, vP) -> bool:
    if numpy.sum(vth) != 0: # Plasma Frío: Distrib. Delta de Dirac, vth -> 0
        lim_inf = 0
        lim_sup = 0
        xr = random.sample(range(nparticulas), nparticulas)
        for i in range(num_especies):
            lim_sup += numpy.int32(c[i]*nparticulas)
            for j in range(lim_inf, lim_sup):
                signo = numpy.power(-1, numpy.int32(numpy.round(numpy.random.random())))
                u = 1-numpy.random.uniform(0, 0.9)**2
                v[xr[j]] = signo*vth[i]*numpy.sqrt(numpy.abs(2*numpy.log(1/u))) + vh[i]
            lim_inf = lim_sup
    v += vP*numpy.sin(2.0*numpy.pi*x/longitud_malla)        # Se agrega la perturbacion
    return True

@jit(nopython = True, parallel = False)
def particula_cf(nparticulas, dimension, x) -> bool: # Condiciones de frontera periódicas
    for i in range(nparticulas):
        if (x[i] < 0.0): # Si la partícula está a la izq., se mueve al lugar correspondiente a la d.
            x[i] += dimension
        elif (x[i] >= dimension):
            x[i] -= dimension
    return True

@jit(nopython = True, parallel = False)
def densidad(nparticulas, npuntos_malla, x, dx, qe):
    re = qe/dx                              # Factor de ponderacion de carga
    rhoe = numpy.zeros(npuntos_malla + 1)
    for i in range(nparticulas):            # Mapa de cargas sobre la malla
        cuadrante = x[i]/dx                 # xparticula/dx: posición de la partícula relativa a celda
        posicion = int(cuadrante)           # índice de la malla fija
        f2 = cuadrante - posicion           # factor de ponderación |xmalla - xparticula|/dx
        f1 = 1.0 - f2
        rhoe[posicion] += re*f1
        rhoe[posicion + 1] += re*f2
    rhoe[0] += rhoe[npuntos_malla]          # Condiciones de frontera periodica
    rhoe[npuntos_malla] = rhoe[0]
    return rhoe

@jit(nopython = False, parallel = False)
def campo(rhoe, rhoi, dx, npuntos_malla):
    rhot = rhoe + rhoi                      # Densidad total de carga
    xintegrar = dx * numpy.arange(npuntos_malla + 1)
    Ex = scipy.integrate.cumulative_trapezoid(rhot, xintegrar, initial = xintegrar[0])
    edc = -numpy.mean(Ex)                   # Remueve desplazamiento DC
    Ex[0:npuntos_malla] += edc
    Ex[npuntos_malla] = Ex[0]
    return Ex

# - Si no hay campo magnético, el movimiento solo es a lo largo de x.
@jit(nopython = True, parallel = False)
def movimientoparticulas1(nparticulas, x, v, Ex, dt, dx, carga_masa) -> bool:
    for i in range(nparticulas):
        cuadrante = x[i]/dx                 # Mismos pasos que en densidad(...)
        posicion = int(cuadrante)
        f2 = cuadrante - posicion
        f1 = 1.0 - f2
        contribucion_campo_electrico = f1*Ex[posicion] + f2*Ex[posicion + 1]
        v[i] += carga_masa*dt*contribucion_campo_electrico
    x += dt*v                               # Actualizar posiciones (2do paso del Leap Frog)
    return True

# - Caso contrario: hay un campo magnético aplicado.
@jit(nopython = True, parallel = False)
def movimientoparticulas2(nparticulas, x, y, v, vx, vy, Ex, dt, dx, carga_masa, Bo) -> bool:
    v_menosx = 0.0 # Todas las componentes en z serán cero, no tiene sentido usar vectores 3D :)
    v_menosy = 0.0
    v_primax = 0.0
    v_primay = 0.0
    v_masx = 0.0
    v_masy = 0.0
    t = carga_masa*Bo*dt*0.5 # En z
    s = 2*t/(1 + t**2)
    for i in range(nparticulas):
        cuadrante = x[i]/dx
        posicion = int(cuadrante)
        f2 = cuadrante - posicion
        f1 = 1.0 - f2
        contribucion_campo_electrico = f1*Ex[posicion] + f2*Ex[posicion + 1]
        vmenosx = vx[i] + carga_masa*dt*contribucion_campo_electrico*0.5 # Metodo de Boris
        vmenosy = vy[i]
        vprimax = vmenosx + vmenosy*t
        vprimay = vmenosy - vmenosx*t
        vmasx = vmenosx + vprimay*s
        vmasy = vmenosy - vprimax*s
        vx[i] = vmasx + carga_masa*dt*contribucion_campo_electrico*0.5
        vy[i] = vmasy
    x += dt*vx
    y += dt*vy
    return True

#n_especies = 1 # Plasma Frío
#c_n = numpy.array([1.], dtype='double')
#vh = numpy.array([1.], dtype='double')
#vth = numpy.array([0.], dtype='double')

#n_especies = 1 # Flujo bajo la acción de un campo uniforme
#c_n = numpy.array([1.], dtype='double')
#vh = numpy.array([8.], dtype='double')
#vth = numpy.array([1], dtype='double')

#n_especies = 2 # Two Stream
#c_n = numpy.array([.5, .5], dtype='double')
#vh = numpy.array([8., -8.], dtype='double')
#vth = numpy.array([1, 1], dtype='double')

#n_especies = 2 # Bump-On-Tail
#c_n = numpy.array([.9, .1], dtype='double')
#vh = numpy.array([8., -8.*c_n[1]/(c_n[0]-c_n[1])], dtype='double')
#vth = numpy.array([1, 1], dtype='double')

# Movidas: a, ca = 0.9 (> 0.5), va = vh, na = n*ca
# Quietas: b, cb = 0.1 (< 0.5), vb = vh*ca/(ca-cb), nb = n*cb

inicio = time.time()

c_n = numpy.array([0.5, 0.5], dtype = 'double')
vh = numpy.array([-5.0, 5.0], dtype = 'double')
vth = numpy.array([1.0, 1.0], dtype = 'double')

# Parámetros de simulación
nparticulas         = 20000
npuntos_malla       = 1024
npasos_temporales   = 1200
longitud_malla      = 32*numpy.pi
Bo                  = 0.0
aP                  = 0.0
vP                  = 0.5

n_especies          = len(c_n)
x                   = numpy.empty(nparticulas)
y                   = numpy.empty(nparticulas)
v                   = numpy.zeros(nparticulas)
Ex                  = numpy.empty(npuntos_malla + 1)    # Campo electrico
rhoe                = numpy.empty(npuntos_malla + 1)    # Arreglo de Mallas: Densidad electronica
vx                  = numpy.empty(nparticulas)
vy                  = numpy.empty(nparticulas)

carga_masa = -1.0
rhoi = 1.0
dx = longitud_malla/npuntos_malla
dt = 0.1

# Creacion de la carpeta para cada parámetro.

carpetas = [name for name in os.listdir(".") if os.path.isdir(name)]
for i in range(len(carpetas)+1):
    if not os.path.exists("Run " + str(i+1)):
        os.makedirs("Run " + str(i+1))
        ruta = "Run " + str(i+1)
        break

# Configuracion inicial de la distribucion de particulas y campos:
cargarposicion(nparticulas, longitud_malla, aP, x)      # Se cargan las particulas sobre la malla
q = -rhoi*longitud_malla/nparticulas
masa = q/carga_masa                                     # Carga
cargarvelocidad(nparticulas, n_especies, c_n, v, vh, vth, longitud_malla, vP)
x += 0.5*dt*v                                           # Primer avanzo a medio intervalo para x
particula_cf(nparticulas, longitud_malla, x)            # Se mantienen las particulas dentro / malla
rhoe = densidad(nparticulas, npuntos_malla, x, dx, q)   # Se calcula la densidad inicial
Ex = campo(rhoe, rhoi, dx, npuntos_malla)               # Se calcula el campo inicial

# Ciclo Principal: CIC

# Se crean los archivos base, estos son los mínimos requeridos
with open(ruta + '/posx.csv', 'w') as posx,\
       open(ruta + '/campoelectrico.csv', 'w') as E, \
       open(ruta + '/densidadcarga.csv', 'w') as Rho:
    numpy.savetxt(posx, [x], fmt = '%10.16f',delimiter=',')
    numpy.savetxt(E, [Ex], fmt = '%10.16f',delimiter=',')
    numpy.savetxt(Rho, [rhoe], fmt = '%10.16f',delimiter=',')
posx = open(ruta + '/posx.csv', 'a')
E = open(ruta + '/campoelectrico.csv', 'a')
Rho = open(ruta + '/densidadcarga.csv', 'a')
if Bo == 0:
    # Se registra la magnitud de la velocidad
    with open(ruta + '/vel.csv', 'w') as vel:
        numpy.savetxt(vel, [v], fmt = '%10.16f', delimiter=',')
    vel = open(ruta + '/vel.csv', 'a')
    for itiempo in range(1, npasos_temporales + 1):
        print('Paso: ', itiempo)
        movimientoparticulas1(nparticulas, x, v, Ex, dt, dx, carga_masa)
        particula_cf(nparticulas, longitud_malla, x)
        rhoe = densidad(nparticulas, npuntos_malla, x, dx, q)
        Ex = campo(rhoe, rhoi, dx, npuntos_malla)
        numpy.savetxt(posx, [x], fmt = '%10.16f',delimiter=',')
        numpy.savetxt(E, [Ex], fmt = '%10.16f',delimiter=',')
        numpy.savetxt(Rho, [rhoe], fmt = '%10.16f',delimiter=',')
        numpy.savetxt(vel, [v], fmt = '%10.16f', delimiter=',')
    vel.close()
# Si hay B, se abren componentes en X y Y de la velocidad, así como comp. Y de la pos :)
# Calcular la magnitud de la velocidad se puede hacer luego para ahorrar tiempo de cómputo
else:
    vx = v*numpy.cos(2*numpy.pi*x/longitud_malla) # Componentes de la velocidad
    vy = v*numpy.sin(2*numpy.pi*x/longitud_malla)
    y = dt*vy
    with open(ruta + '/velx.csv', 'w') as velx, \
           open(ruta + '/vely.csv', 'w') as vely, \
           open(ruta + '/posy.csv', 'w') as posy:
        numpy.savetxt(velx, [vx], fmt = '%10.16f', delimiter=',')
        numpy.savetxt(vely, [vy], fmt = '%10.16f', delimiter=',')
        numpy.savetxt(posy, [y], fmt = '%10.16f', delimiter=',')
    velx = open(ruta + '/velx.csv', 'a')
    vely = open(ruta + '/vely.csv', 'a')
    posy = open(ruta + '/posy.csv', 'a')
    for itiempo in range(1, npasos_temporales + 1):
        print('Paso:', itiempo)
        movimientoparticulas2(nparticulas, x, y, v, vx, vy, Ex, dt, dx, carga_masa, Bo)
        particula_cf(nparticulas, longitud_malla, x)
        rhoe = densidad(nparticulas, npuntos_malla, x, dx, q)
        Ex = campo(rhoe, rhoi, dx, npuntos_malla)
        numpy.savetxt(posx, [x], fmt = '%10.16f',delimiter=',')
        numpy.savetxt(E, [Ex], fmt = '%10.16f',delimiter=',')
        numpy.savetxt(Rho, [rhoe], fmt = '%10.16f',delimiter=',')
        numpy.savetxt(velx, [vx], fmt = '%10.16f', delimiter=',')
        numpy.savetxt(vely, [vy], fmt = '%10.16f', delimiter=',')
        numpy.savetxt(posy, [y], fmt = '%10.16f', delimiter=',')
    velx.close()
    vely.close()
    posy.close()
posx.close()
E.close()
Rho.close()

t = time.time() - inicio
print('Tiempo total: ', t)

with open(ruta + '/descripcion.txt', 'w') as mainArchivo:
    mainArchivo.write(" Descripcion de simulacion \"" + ruta + "\" \n")
    mainArchivo.write(" - Fenomeno:             ")
    if numpy.sum(vth) == 0:
        mainArchivo.write("Plasma frio\n")
    elif n_especies == 1:
        mainArchivo.write("Flujo normal (1 especie)\n")
    elif n_especies == 2:
        if c_n[0] == 0.5:
            mainArchivo.write("Two-Stream\n")
        else:
            mainArchivo.write("Bump-On-Tail\n")
    else:
        mainArchivo.write("Personalizado\n")
    mainArchivo.write(" - Numero de especies:   " + str(n_especies) + "\n")
    mainArchivo.write(" - Coeficient(es) ci:    ")
    numpy.savetxt(mainArchivo, [c_n], '%1.5f', delimiter=',')
    mainArchivo.write(" - Velocidad(es) vhi:    ")
    numpy.savetxt(mainArchivo, [vh], '%1.5f', delimiter=',')
    mainArchivo.write(" - Ancho(s) v. vthi:     ")
    numpy.savetxt(mainArchivo, [vth], '%1.5f', delimiter=',')
    mainArchivo.write(" ###################################################\n")
    mainArchivo.write(" - Particulas:           " + str(nparticulas) + "\n")
    mainArchivo.write(" - Puntos de malla (SP): " + str(npuntos_malla) + "\n")
    mainArchivo.write(" - Pasos temporales:     " + str(npasos_temporales) + "\n")
    mainArchivo.write(" - Longitud de malla:    " + str(longitud_malla) + "\n")
    mainArchivo.write(" - Campo B (Bo, dir z):  " + str(Bo) + "\n")
    mainArchivo.write(" - Amp. Perturb. Pos.:   " + str(aP) + "\n")
    mainArchivo.write(" - Amp. Perturb. Vel.:   " + str(vP) + "\n")
    mainArchivo.write(" ###################################################\n")
    if t > 300:
        mainArchivo.write(" - Tiempo de simulacion (min): " + str(t/60.0) +" \n")
    else:
        mainArchivo.write(" - Tiempo de simulacion (seg): " + str(t) +" \n")
os.system("graficas.pyw " + ruta)
