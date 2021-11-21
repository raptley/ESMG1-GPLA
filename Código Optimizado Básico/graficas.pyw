import numpy
import sys
import matplotlib.pyplot
import csv
import os
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.fftpack import fft2
from scipy.fftpack import fftshift as shift

# Se pasa como parámetro la ruta de ejecución. Alternativamente, se puede elegir desde explorador.
ruta = "Run 4"
if len(sys.argv) > 1:
    ruta = sys.argv[1] + " " + sys.argv[2]

with open(ruta + '/descripcion.txt') as datos:
    lineas = datos.readlines()
nparticulas         = int(lineas[7][25:len(lineas[7])-1])
npuntos_malla       = int(lineas[8][25:len(lineas[8])-1])
npasos_temporales   = int(lineas[9][25:len(lineas[9])-1])
longitud_malla      = float(lineas[10][25:len(lineas[10])-1])
Bo                  = float(lineas[11][25:len(lineas[11])-1])
dx = longitud_malla/npuntos_malla

# Carga Campo Eléctrico (lista de npuntos_malla + 1 columnas y npasos_tamporales + 1 filas)
with open(ruta + '/campoelectrico.csv') as archivoE:
    E = list(csv.reader(archivoE, delimiter = ','))
x = numpy.linspace(0, longitud_malla, npuntos_malla + 1)

# Calcula Energías Potencial, Cinética y Total. This kills the man.
if Bo == 0:
    with open(ruta + '/vel.csv') as archivoVel, \
    open(ruta + '/posx.csv') as archivoPos:
        velx = list(csv.reader(archivoVel, delimiter = ','))
        posx = list(csv.reader(archivoPos, delimiter = ','))
else:
    with open(ruta + '/velx.csv') as archivoVelx, \
    open(ruta + '/posx.csv') as archivoPosx:
        velx = list(csv.reader(archivoVelx, delimiter = ','))
        posx = list(csv.reader(archivoPosx, delimiter = ','))

if not os.path.exists(ruta + "/Energias"):
    os.makedirs(ruta + "/Energias")
    Ek = open(ruta + '/Energias/Ek.csv', 'w')
    Ep = open(ruta + '/Energias/Ep.csv', 'w')
    Et = open(ruta + '/Energias/Et.csv', 'w')
    for i in range(len(velx)):
        numpy.savetxt(Ek, 
                      [0.5*longitud_malla*numpy.sum(numpy.asarray(velx[i], dtype = 'double')**2)/nparticulas], 
                      fmt = '%10.16f', delimiter = ',')
        numpy.savetxt(Ep, 
                      [0.5*dx*numpy.sum(numpy.asarray(E[i], dtype = 'double')**2)], 
                      fmt = '%10.16f', delimiter = ',')
        numpy.savetxt(Et, 
                      [0.5*longitud_malla*numpy.sum(numpy.asarray(velx[i], dtype = 'double')**2)/nparticulas +
                      0.5*dx*numpy.sum(numpy.asarray(E[i], dtype = 'double')**2)], 
                      fmt = '%10.16f', delimiter = ',')
    Ek.close()
    Ep.close()
    Et.close()
with open(ruta + '/Energias/Ek.csv') as archivoEk, \
        open(ruta + '/Energias/Ep.csv') as archivoEp, \
        open(ruta + '/Energias/Et.csv') as archivoEt:
    Ek = list(csv.reader(archivoEk, delimiter = ','))
    Ep = list(csv.reader(archivoEp, delimiter = ','))
    Et = list(csv.reader(archivoEt, delimiter = ','))
t = 0.1*numpy.arange(npasos_temporales + 1)

# Espacio de fase
if not os.path.exists(ruta + "/PlanosFase"):
    os.makedirs(ruta + "/PlanosFase")
    vsup = open(ruta + '/PlanosFase/v1.csv', 'w')
    vinf = open(ruta + '/PlanosFase/v2.csv', 'w')
    xsup = open(ruta + '/PlanosFase/x1.csv', 'w')
    xinf = open(ruta + '/PlanosFase/x2.csv', 'w')
    v1 = numpy.zeros(nparticulas)
    v2 = numpy.zeros(nparticulas)
    x1 = numpy.zeros(nparticulas)
    x2 = numpy.zeros(nparticulas)
    for i in range(len(velx)):
        vx_aux = numpy.asarray(velx[i], dtype = 'double')
        x_aux = numpy.asarray(posx[i], dtype = 'double')
        for j in range(nparticulas):
            if (vx_aux[j-1] > vx_aux[j]):
                v1[j] = vx_aux[j]
                x1[j] = x_aux[j]
            elif(vx_aux[j-1] <= vx_aux[j]):
                v2[j] = vx_aux[j]
                x2[j] = x_aux[j]
        numpy.savetxt(vsup, [v1], fmt = '%10.16f', delimiter = ',')
        numpy.savetxt(vinf, [v2], fmt = '%10.16f', delimiter = ',')
        numpy.savetxt(xsup, [x1], fmt = '%10.16f', delimiter = ',')
        numpy.savetxt(xinf, [x2], fmt = '%10.16f', delimiter = ',')
        v1[:] = 0
        v2[:] = 0
        x1[:] = 0
        x2[:] = 0
    vsup.close()
    vinf.close()
    xsup.close()
    xinf.close()
    with open(ruta + '/PlanosFase/v1.csv') as archivoVsup, \
        open(ruta + '/PlanosFase/v2.csv') as archivoVinf, \
        open(ruta + '/PlanosFase/x1.csv') as archivoXsup, \
        open(ruta + '/PlanosFase/x2.csv') as archivoXinf:
        vsup = list(csv.reader(archivoVsup, delimiter = ','))
        vinf = list(csv.reader(archivoVinf, delimiter = ','))
        xsup = list(csv.reader(archivoXsup, delimiter = ','))
        xinf = list(csv.reader(archivoXinf, delimiter = ','))
    for i in range(len(velx)): # Remover ceros: hacen peso y no se necesita graficarlos.
        vsup[i] = [elemento for elemento in vsup[i] if float(elemento) != 0]
        xsup[i] = [elemento for elemento in xsup[i] if float(elemento) != 0]
        vinf[i] = [elemento for elemento in vinf[i] if float(elemento) != 0]
        xinf[i] = [elemento for elemento in xinf[i] if float(elemento) != 0]
else:
    with open(ruta + '/PlanosFase/v1.csv') as archivoVsup, \
        open(ruta + '/PlanosFase/v2.csv') as archivoVinf, \
        open(ruta + '/PlanosFase/x1.csv') as archivoXsup, \
        open(ruta + '/PlanosFase/x2.csv') as archivoXinf:
        vsup = list(csv.reader(archivoVsup, delimiter = ','))
        vinf = list(csv.reader(archivoVinf, delimiter = ','))
        xsup = list(csv.reader(archivoXsup, delimiter = ','))
        xinf = list(csv.reader(archivoXinf, delimiter = ','))

del posx
del velx

# Relación de Dispersión
# Reemplazar nextpow2(longitud_malla) con 2**int(numpy.log2(longitud_malla)+1)
omega_min = 2*numpy.pi/(0.1)/2/(npasos_temporales/2)
omega_max = omega_min*(npasos_temporales/2)
k_min = 2*numpy.pi/(npuntos_malla)
k_max = k_min*((npuntos_malla/2)-1)
k_t = numpy.linspace(0, k_max, nparticulas)
k_simulada = numpy.linspace(-k_max, k_max, 2**int(numpy.log2(npuntos_malla)+1))
omega_t = numpy.linspace(0, longitud_malla, npuntos_malla + 1)
omega_simulada = numpy.linspace(-omega_max, omega_max, 2**int(numpy.log2(npuntos_malla) + 1))
K, W = numpy.meshgrid(k_simulada, omega_simulada)

if not os.path.exists(ruta + "/RelDisp"):
    os.makedirs(ruta + "/RelDisp")
    n2pmalla = 2**int(numpy.log2(npuntos_malla) + 1)
    with open(ruta + "/RelDisp/Ewk.csv", 'w') as archivoEwk:
        Ewk = shift(abs(fft2(numpy.array(numpy.array(E), dtype = 'double'), 
                                        (n2pmalla,n2pmalla))/longitud_malla))
        for i in range(n2pmalla):
            numpy.savetxt(archivoEwk, 
                        [Ewk[i]], 
                        fmt = '%10.16f', delimiter = ',')

with open(ruta + '/RelDisp/Ewk.csv') as archivoEwk:
        Ewk = numpy.array(numpy.array(list(csv.reader(archivoEwk, 
                                                      delimiter = ','))), 
                          dtype = 'double')

FigPrincipal, Ejes = matplotlib.pyplot.subplots(2, 2)
matplotlib.pyplot.subplots_adjust(left = 0.15, bottom = 0.2)

Ejes[0, 0].set_xlim(0, longitud_malla)
ymaxEx = numpy.max(numpy.asarray(E[npasos_temporales], dtype = 'double'))
yminEx = numpy.min(numpy.asarray(E[npasos_temporales], dtype = 'double'))
ymaxEx = numpy.round(ymaxEx, -int(f'{ymaxEx:e}'.split('e')[-1]))
yminEx = numpy.round(yminEx, -int(f'{yminEx:e}'.split('e')[-1]))
Ejes[0, 0].set_ylim(numpy.max([abs(ymaxEx), abs(yminEx)]), 
                    -numpy.max([abs(ymaxEx), abs(yminEx)]))
Ejes[0, 0].grid(True)
Ejes[0, 0].set_xlabel("Separacion SP (m)")
Ejes[0, 0].set_ylabel("Campo Electrico (Vm/s)")

Ejes[1, 0].set_xlim(0, longitud_malla)
ymaxvx = 1.5*numpy.ceil(numpy.max(numpy.asarray(vsup[0], dtype = 'double')))+1
yminvx = 1.5*numpy.floor(numpy.min(numpy.asarray(vinf[0], dtype = 'double')))-1
Ejes[1, 0].set_ylim(ymaxvx, yminvx)
Ejes[1, 0].set_xlabel("x")
Ejes[1, 0].set_ylabel("v")

Ejes[0, 1].set_xlim(0, npasos_temporales*0.1)
Ejes[0, 1].set_xlabel("Tiempo (s)")
Ejes[0, 1].set_ylabel("Energia (J)")

l, = Ejes[0, 0].plot(x, numpy.asarray(E[0], dtype = 'double'), linewidth = 2, color = 'black')
Ejes[1, 0].scatter(numpy.asarray(xsup[0], dtype = 'double'), numpy.asarray(vsup[0], dtype = 'double'), 
           s = 0.1, color = 'red')
Ejes[1, 0].scatter(numpy.asarray(xinf[0], dtype = 'double'), numpy.asarray(vinf[0], dtype = 'double'), 
           s = 0.1, color = 'blue')
Ejes[0, 1].plot(t, numpy.asarray(Ek, dtype = 'double'), color = 'blue', label='E. Cinetica')
Ejes[0, 1].plot(t, numpy.asarray(Ep, dtype = 'double'), color = 'red', label='E. Potencial')
Ejes[0, 1].plot(t, numpy.asarray(Et, dtype = 'double'), color = 'black', label='E. Total')
Ejes[0, 1].legend(loc = 4)
m = Ejes[1, 1].contourf(K, W, Ewk, 8, alpha = .75, cmap = 'jet')
Ejes[1, 1].set_xlim(-0.1, 0.1)
Ejes[1, 1].set_ylim(-2.5, 2.5)
FigPrincipal.colorbar(m, ax = Ejes[1, 1])
Ejes[1, 1].set_xlabel('k')
Ejes[1, 1].set_ylabel('$\omega$')

Ejes[0, 0].margins(x = 0)
                                    #[left, bottom, width, height]
EjeTiempo = matplotlib.pyplot.axes([0.15, 0.075, 0.75, 0.03], facecolor = 'white') # Objeto Axes
SliderTiempo = Slider(EjeTiempo, 't', 0, int(npasos_temporales/10), valinit = 0, valstep = 0.1)

def ActualizarSliderTiempo(val):
    l.set_ydata(numpy.asarray(E[int(10*SliderTiempo.val)], dtype = 'double'))
    Ejes[1, 0].cla()
    Ejes[1, 0].scatter(numpy.asarray(xsup[int(10*SliderTiempo.val)], dtype = 'double'), 
                       numpy.asarray(vsup[int(10*SliderTiempo.val)], dtype = 'double'), 
           s = 0.1, color = 'red')
    Ejes[1, 0].scatter(numpy.asarray(xinf[int(10*SliderTiempo.val)], dtype = 'double'), 
                       numpy.asarray(vinf[int(10*SliderTiempo.val)], dtype = 'double'), 
           s = 0.1, color = 'blue')
    Ejes[1, 0].set_ylim(ymaxvx, yminvx)
    Ejes[1, 0].set_xlabel("x")
    Ejes[1, 0].set_ylabel("v")
    FigPrincipal.suptitle('Graficas a t = ' + str(numpy.round(SliderTiempo.val, 1)) + 's')
    FigPrincipal.canvas.draw_idle()

SliderTiempo.on_changed(ActualizarSliderTiempo)

Avanzar = matplotlib.pyplot.axes([0.85, 0.01, 0.05, 0.05])
BtnAvanzar = Button(Avanzar, '>', color = 'white', hovercolor = '0.975')

def PressBtnAvanzar(event):
    if SliderTiempo.val + 0.1 <= npasos_temporales/10:
        FigPrincipal.suptitle('Graficas a t = ' + str(numpy.round(SliderTiempo.val, 1)) + 's')
        SliderTiempo.set_val(SliderTiempo.val + 0.1)

BtnAvanzar.on_clicked(PressBtnAvanzar)

Retroceder = matplotlib.pyplot.axes([0.15, 0.01, 0.05, 0.05])
BtnRetroceder = Button(Retroceder, '<', color = 'white', hovercolor = '0.975')

def PressBtnRetroceder(event):
    if SliderTiempo.val - 0.1 >= 0:
        FigPrincipal.suptitle('Graficas a t = ' + str(numpy.round(SliderTiempo.val, 1)) + 's')
        SliderTiempo.set_val(SliderTiempo.val - 0.1)

BtnRetroceder.on_clicked(PressBtnRetroceder)

Pausar = matplotlib.pyplot.axes([0.5, 0.01, 0.05, 0.05])
BtnPausar = Button(Pausar, 'v', color = 'white', hovercolor = '0.975')
BtnPausar.label.set_rotation(90)

def PressBtnPausar(event):
    if BtnPausar.label.get_text() == 'v':
        if SliderTiempo.val != npasos_temporales/10:
            BtnPausar.label.set_text('=')
            while BtnPausar.label.get_text() == '=':
                if SliderTiempo.val + 0.1 <= npasos_temporales/10:
                    FigPrincipal.suptitle('Graficas a t = ' + str(numpy.round(SliderTiempo.val, 1)) + 's')
                    SliderTiempo.set_val(SliderTiempo.val + 0.1)
                else:
                    BtnPausar.label.set_text('v')
                    break
                matplotlib.pyplot.pause(0.05)
    else:
        BtnPausar.label.set_text('v')

BtnPausar.on_clicked(PressBtnPausar)

matplotlib.pyplot.subplots_adjust(left=0.15,
                    bottom=0.2, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.4)
FigPrincipal.suptitle('Graficas a t = 0 s')
matplotlib.pyplot.show()
