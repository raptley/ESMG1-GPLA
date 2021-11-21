from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter.ttk import Notebook
from tkinter.ttk import Progressbar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import ImageTk, Image
import time
import numpy
import scipy.integrate
import os
import sys
import random
import csv
from scipy.fftpack import fft2
from scipy.fftpack import fftshift as shift
from numba import jit
import warnings
warnings.filterwarnings("ignore")

# Funciones para simular
@jit(nopython = True, parallel = False)
def cargarposicion(nparticulas, longitud_malla, aP, x) -> bool:
    espacio_particulas = longitud_malla/nparticulas
    for i in range(nparticulas):
        x[i] = espacio_particulas*(i + 0.5)
        x[i] += aP*numpy.cos(x[i])
    return True

@jit(nopython = False, parallel = False)
def cargarvelocidad(nparticulas, num_especies, c, v, vh, vth, longitud_malla, vP) -> bool:
    if numpy.sum(vth) != 0:
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
    v += vP*numpy.sin(2.0*numpy.pi*v/longitud_malla)
    return True

@jit(nopython = True, parallel = False)
def particula_cf(nparticulas, dimension, x) -> bool:
    for i in range(nparticulas):
        if (x[i] < 0.0):
            x[i] += dimension
        elif (x[i] >= dimension):
            x[i] -= dimension
    return True

@jit(nopython = True, parallel = False)
def densidad(nparticulas, npuntos_malla, x, dx, qe):
    re = qe/dx
    rhoe = numpy.zeros(npuntos_malla + 1)
    for i in range(nparticulas):
        cuadrante = x[i]/dx
        posicion = int(cuadrante)
        f2 = cuadrante - posicion
        f1 = 1.0 - f2
        rhoe[posicion] += re*f1
        rhoe[posicion + 1] += re*f2
    rhoe[0] += rhoe[npuntos_malla]
    rhoe[npuntos_malla] = rhoe[0]
    return rhoe

@jit(nopython = False, parallel = False)
def campo(rhoe, rhoi, dx, npuntos_malla):
    rhot = rhoe + rhoi
    xintegrar = dx * numpy.arange(npuntos_malla + 1)
    Ex = scipy.integrate.cumulative_trapezoid(rhot, xintegrar, initial = xintegrar[0])
    edc = -numpy.mean(Ex)
    Ex[0:npuntos_malla] += edc
    Ex[npuntos_malla] = Ex[0]
    return Ex

@jit(nopython = True, parallel = False)
def movimientoparticulas1(nparticulas, x, v, Ex, dt, dx, carga_masa) -> bool:
    for i in range(nparticulas):
        cuadrante = x[i]/dx
        posicion = int(cuadrante)
        f2 = cuadrante - posicion
        f1 = 1.0 - f2
        contribucion_campo_electrico = f1*Ex[posicion] + f2*Ex[posicion + 1]
        v[i] += carga_masa*dt*contribucion_campo_electrico
    x += dt*v
    return True

@jit(nopython = True, parallel = False)
def movimientoparticulas2(nparticulas, x, y, v, vx, vy, Ex, dt, dx, carga_masa, Bo) -> bool:
    v_menosx = 0.0
    v_menosy = 0.0
    v_primax = 0.0
    v_primay = 0.0
    v_masx = 0.0
    v_masy = 0.0
    t = carga_masa*Bo*dt*0.5
    s = 2*t/(1 + t**2)
    for i in range(nparticulas):
        cuadrante = x[i]/dx
        posicion = int(cuadrante)
        f2 = cuadrante - posicion
        f1 = 1.0 - f2
        contribucion_campo_electrico = f1*Ex[posicion] + f2*Ex[posicion + 1]
        vmenosx = vx[i] + carga_masa*dt*contribucion_campo_electrico*0.5
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

# Constantes
ancho = 20
alto = 2
espacio = "     "

##### Eventos #####
def NuevaSimul():
    python = sys.executable
    os.execl(python, python, * sys.argv)

def AbrirSimul():
    global directorio_open, apertura
    try:
        x = filedialog.askdirectory(initialdir = "simulaciones", title = "Seleccione una carpeta Run")
        directorio_open = x
        with open(x + "/descripcion.txt") as datos:
            lineas = datos.readlines()
        root.title(x.split("/")[-1] + ", ESMG1-GPLA (GUI)")
        var1.set(lineas[3][25:len(lineas[3])-1])
        var2.set(lineas[4][25:len(lineas[4])-1])
        var3.set(lineas[5][25:len(lineas[5])-1])
        var4.set(lineas[7][25:len(lineas[7])-1])
        var5.set(lineas[8][25:len(lineas[8])-1])
        var6.set(lineas[9][25:len(lineas[9])-1])
        var7.set(lineas[10][25:len(lineas[10])-1])
        var8.set(lineas[11][25:len(lineas[11])-1])
        var9.set(lineas[12][25:len(lineas[12])-1])
        var10.set(lineas[13][25:len(lineas[13])-1])
        pivar.set(0)
        apertura = True
        ingci["state"] = "disabled"
        ingvh["state"] = "disabled"
        ingvth["state"] = "disabled"
        ingnp["state"] = "disabled"
        ingnm["state"] = "disabled"
        ingst["state"] = "disabled"
        inglm["state"] = "disabled"
        checkpi["state"] = "disabled"
        ingBo["state"] = "disabled"
        ingxP["state"] = "disabled"
        ingvP["state"] = "disabled"
        btnSimular["state"] = "active"
        btnSimular["text"] = "Graficar..."
    except:
        messagebox.showerror("Error en apertura de carpeta", 
                         "Se eliminó la carpeta de simulaciones, \no no se eligió una carpeta Run. Intente de nuevo.")

def actualizar():
    graficar = False
    try:
        ci = numpy.fromstring(ingci.get(), dtype="double", sep = ",")
        vh = numpy.fromstring(ingvh.get(), dtype="double", sep = ",")
        vth = numpy.fromstring(ingvth.get(), dtype="double", sep = ",")
        if ((len(ci) == len(vh) and len(ci) == len(vth)) or numpy.sum(vth) == 0) and ingci.get() != "" and ingvh.get() != "" and ingvth.get() != "":
            graficar = True
    except:
        ax.cla()
        canvas.draw()
    if graficar:
        if numpy.sum(vth) != 0:
            vmin = numpy.min(vh)-3*numpy.max(vth)
            vmax = numpy.max(vh)+3*numpy.max(vth)
            v = numpy.linspace(vmin, vmax, 1000)
            y = numpy.zeros(1000)
            for i in range(len(ci)):
                for j in range(1000):
                    y[j] += (ci[i])*numpy.exp(-((v[j]-vh[i])**2/(vth[i]**2)))
            ax.cla()
            ax.set_ylim(0, 1.1)
            ax.set_xlim(vmin, vmax)
            ax.set_xlabel("v")
            ax.set_ylabel("f(v)")
            ax.grid()
            ax.plot(v, y, color = listaColor[colorFDistVariable.get()])
            canvas.draw()
        else:
            v = numpy.linspace(-2, 2, 10000)
            ax.cla()
            ax.set_ylim(0, 1.1)
            ax.set_xlim(-2, 2)
            ax.set_xlabel("v")
            ax.set_ylabel("f(v)")
            ax.grid()
            ax.plot(v, numpy.exp(-100000*v**2), color = listaColor[colorFDistVariable.get()])
            canvas.draw()
    return graficar

def callback(var):
   actualizar()

def nada(var):
    pass

def validar(input):
    if input.isdigit():
        return True
    elif input == "":
        return True
    elif input[-1] == "." or input[-1] == "," or input[-1] == "-" or input[-1].isdigit() or input[-1] == " ":
        return True
    else:
        return False

def cierre():
    if messagebox.askokcancel("Salir", "Hay una simulación en proceso.\nSi sale, se perderán los datos.\n¿Desea salir de todos modos?"):
        root.quit()
        root.destroy()

def runSimul():
    global sliderTiempo, graficadoCE, axCE, npasos_temporales, ruta, canvasCE, colorCEVariable
    global sliderTiempo2, graficadoDiagFase, axDiagFase, canvasDiagFase, colorDiagFaseVariableA, colorDiagFaseVariableB, listaColor
    global axEnergias, canvasEnergias, colorEkVariable, colorEpVariable, colorEtVariable
    global cadena1, cadena2, cadena3
    global graficadoEnergias
    global directorio_open, apertura
    if btnSimular["text"] == "Simular...":
        exitns = False
        try:
            c_n = numpy.fromstring(ingci.get(), dtype="double", sep = ",")
            vh = numpy.fromstring(ingvh.get(), dtype="double", sep = ",")
            vth = numpy.fromstring(ingvth.get(), dtype="double", sep = ",")
            nparticulas = int(ingnp.get())
            npuntos_malla = int(ingnm.get())
            npasos_temporales = int(ingst.get())
            longitud_malla = float(inglm.get())
            if pivar.get() == 1:
                longitud_malla = longitud_malla*numpy.pi
            Bo = float(ingBo.get())
            aP = float(ingxP.get())
            vP = float(ingvP.get())
            exitns = True
        except:
            messagebox.showerror("Error", "Uno o más parámetros ingresados no válidos. Inténtelo de nuevo.\nSi necesita ayuda, siga uno de los ejemplos.")
        if exitns and actualizar():
            n_especies = len(c_n)
            btnSimular["state"] = "disabled"
            root.update()
            # Ciclo de simulación
            inicio = time.time()
            labelBarra["text"] = "Iniciando..."
            barra["mode"] = "indeterminate"
            barra.start()
            root.update()
            carga_masa = -1.0
            x                   = numpy.empty(nparticulas)
            y                   = numpy.empty(nparticulas)
            v                   = numpy.zeros(nparticulas)
            Ex                  = numpy.empty(npuntos_malla + 1)
            rhoe                = numpy.empty(npuntos_malla + 1)
            rhoi                = numpy.empty(npuntos_malla + 1)
            vx                  = numpy.empty(nparticulas)
            vy                  = numpy.empty(nparticulas)
            rhoi = 1.0
            dx = longitud_malla/npuntos_malla
            dt = 0.1
            k = 1
            labelBarra["text"] = "Creando carpeta..."
            root.update()
            while True:
                if not os.path.exists("simulaciones/Run " + str(k)):
                    os.makedirs("simulaciones/Run " + str(k))
                    ruta = "simulaciones/Run " + str(k)
                    break
                k+=1
            cargarposicion(nparticulas, longitud_malla, aP, x)
            q = -rhoi*longitud_malla/nparticulas
            masa = q/carga_masa
            cargarvelocidad(nparticulas, n_especies, c_n, v, vh, vth, longitud_malla, vP)
            x += 0.5*dt*v
            particula_cf(nparticulas, longitud_malla, x)
            rhoe = densidad(nparticulas, npuntos_malla, x, dx, q)
            Ex = campo(rhoe, rhoi, dx, npuntos_malla)
            labelBarra["text"] = "Creando archivos..."
            root.update()
            with open(ruta + '/posx.csv', 'w') as posx,\
                   open(ruta + '/campoelectrico.csv', 'w') as E, \
                   open(ruta + '/densidadcarga.csv', 'w') as Rho:
                numpy.savetxt(posx, [x], fmt = '%10.16f',delimiter=',')
                numpy.savetxt(E, [Ex], fmt = '%10.16f',delimiter=',')
                numpy.savetxt(Rho, [rhoe], fmt = '%10.16f',delimiter=',')
            posx = open(ruta + '/posx.csv', 'a')
            E = open(ruta + '/campoelectrico.csv', 'a')
            Rho = open(ruta + '/densidadcarga.csv', 'a')
            barra.stop()
            barra["mode"] = "determinate"
            barra.step(0)
            labelBarra["text"] = "Iniciando pasos..."
            root.update()
            if Bo == 0:
                with open(ruta + '/vel.csv', 'w') as vel:
                    numpy.savetxt(vel, [v], fmt = '%10.16f', delimiter=',')
                vel = open(ruta + '/vel.csv', 'a')
                for itiempo in range(1, npasos_temporales + 1):
                    labelBarra["text"] = "Paso temporal " + str(itiempo) + " de " + str(npasos_temporales)
                    barra.step(100/(npasos_temporales-1))
                    root.update()
                    movimientoparticulas1(nparticulas, x, v, Ex, dt, dx, carga_masa)
                    particula_cf(nparticulas, longitud_malla, x)
                    rhoe = densidad(nparticulas, npuntos_malla, x, dx, q)
                    Ex = campo(rhoe, rhoi, dx, npuntos_malla)
                    numpy.savetxt(posx, [x], fmt = '%10.16f',delimiter=',')
                    numpy.savetxt(E, [Ex], fmt = '%10.16f',delimiter=',')
                    numpy.savetxt(Rho, [rhoe], fmt = '%10.16f',delimiter=',')
                    numpy.savetxt(vel, [v], fmt = '%10.16f', delimiter=',')
                vel.close()
            else:
                vx = v*numpy.cos(2*numpy.pi*x/longitud_malla)
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
                    labelBarra["text"] = "Paso temporal " + str(itiempo) + " de " + str(npasos_temporales)
                    barra.step(100/(npasos_temporales-1))
                    root.update()
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
            labelBarra["text"] = "Escribiendo datos..."
            barra["mode"] = "indeterminate"
            barra.start()
            root.update()
            with open(ruta + '/descripcion.txt', 'w') as mainArchivo:
                mainArchivo.write(" Descripcion de simulacion \"" + ruta[13:] + "\" \n")
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
            labelBarra["text"] = "Terminado en " + str(t) + " segundos."
            barra.stop()
            barra["mode"] = "determinate"
            barra.step(100)
            root.update()
            root.title(ruta[13:] + ", ESMG1-GPLA (GUI)")
            btnSimular["state"] = "active"
            btnSimular["text"] = "Graficar..."
            ingci["state"] = "disabled"
            ingvh["state"] = "disabled"
            ingvth["state"] = "disabled"
            ingnp["state"] = "disabled"
            ingnm["state"] = "disabled"
            ingst["state"] = "disabled"
            inglm["state"] = "disabled"
            checkpi["state"] = "disabled"
            ingBo["state"] = "disabled"
            ingxP["state"] = "disabled"
            ingvP["state"] = "disabled"
    else:
        labelBarra["text"] = "Generando gráficas..."
        root.update()
        btnSimular["state"] = "disabled"
        if apertura:
            ruta = directorio_open
            apertura = False
        # Campo eléctrico
        labelBarra["text"] = "Generando gráficas: Campo Eléctrico..."
        npasos_temporales = int(ingst.get())
        necesario_graficar = True
        if os.path.exists(ruta + "/CElec"):
            archivos = [name for name in os.listdir(ruta + "/CElec") if os.path.isfile(os.path.join(ruta + "/CElec", name))]
            if len(archivos) == npasos_temporales + 2:
                necesario_graficar = False
                graficadoCE = True
        with open(ruta + "/campoelectrico.csv") as lectorE:
            E = list(csv.reader(lectorE))
        figCElec = Figure(figsize=(5, 4), dpi = 110)
        axCE = plt.Axes(figCElec, [0., 0., 1., 1.])
        axCE.set_axis_off()
        frameFigCElec = Frame(pestanas, width = 800, height = 800)
        canvasCE = FigureCanvasTkAgg(figCElec, master = frameFigCElec)
        canvasCE.get_tk_widget().grid(row = 0, column = 0)
        pestanas.add(frameFigCElec, text = "Campo Eléctrico")
        sliderTiempo = Scale(frameFigCElec, from_ = 0, to = npasos_temporales/10, 
                             resolution = 0.1, orient = HORIZONTAL, length = 450, command = escalaTiempo)
        sliderTiempo.place(x=90, y=400)
        figCElec.add_axes(axCE)
        colorCEVariable = StringVar(frameFigCElec)
        colorCEVariable.trace("w", cambiocolorCE)
        colorCEVariable.set(list(listaColor.keys())[0])
        listaColorCE = OptionMenu(frameFigCElec, colorCEVariable, *list(listaColor.keys()))
        listaColorCE.place(x=0, y=410)
        # Grid: C. Electrico
        longitud_malla = float(inglm.get())
        npuntos_malla = int(ingnm.get())
        if pivar.get() == 1:
            longitud_malla = longitud_malla*numpy.pi
        if necesario_graficar:
            os.makedirs(ruta + "/CElec")
            EMax = numpy.max(numpy.array(E[0], dtype = "double"))
            EMin = numpy.min(numpy.array(E[0], dtype = "double"))
            for i in range(1, len(E)):
                EMaxTemp = numpy.max(numpy.array(E[i], dtype = "double"))
                EMinTemp = numpy.min(numpy.array(E[i], dtype = "double"))
                if EMax < EMaxTemp:
                    EMax = EMaxTemp
                if EMin > EMinTemp:
                    EMin = EMinTemp
            x = numpy.linspace(0, longitud_malla, npuntos_malla + 1)
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0.175, 0.175, 0.8, 0.8])
            ax.set_ylim(EMin, EMax)
            ax.set_xlim(0, x[len(x)-1])
            ax.set_xlabel("x (m)")
            ax.set_ylabel("E (V/m)")
            fig.add_axes(ax)
            ax.grid()
            plt.savefig(ruta + "/CElec/Grid.png", transparent = True)
            for i in range(len(E)):
                ax.cla()
                ax.set_ylim(EMin, EMax)
                ax.set_xlim(x[0], x[len(x)-1])
                ax.set_xlabel("x")
                ax.set_ylabel("E (V/m)")
                barra.step(100/(len(E)+1))
                ax.plot(x, numpy.array(E[i], dtype = "double"), color = "black")
                ax.set_axis_off()
                plt.savefig(ruta + "/CElec/Elec" + str(i).zfill(int(numpy.floor(numpy.log10(len(E)))+1)) + ".png", transparent = True)
                root.update()
            graficadoCE = True

        axCE.imshow(plt.imread(ruta + "/CElec/Grid.png", format = "RGBA"))
        axCE.imshow(plt.imread(ruta + "/CElec/Elec" + str(0).zfill(int(numpy.floor(numpy.log10(len(E)))+1)) + ".png", format = "RGBA"))
        escalaTiempo(1)
        canvasCE.draw()
        labelBarra["text"] = "Gráficas Terminadas: Campo Eléctrico"
        barra.step(0)
        root.update()
        # Diag. Fase
        labelBarra["text"] = "Generando gráficas: Diagrama de Fase..."
        root.update()
        necesario_graficar = True
        if os.path.exists(ruta + "/DiagFase"):
            archivos = [name for name in os.listdir(ruta + "/DiagFase") if os.path.isfile(os.path.join(ruta + "/DiagFase", name))]
            if len(archivos) == 2*(npasos_temporales + 1) + 1:
                necesario_graficar = False
                graficadoDiagFase = True
        with open(ruta + "/posx.csv") as lectorPosx:
            posx = list(csv.reader(lectorPosx))
        Bo = float(ingBo.get())
        if Bo == 0:
            with open(ruta + "/vel.csv") as lectorVel:
                vel = list(csv.reader(lectorVel))
        else:
            with open(ruta + "/velx.csv") as lectorVel:
                vel = list(csv.reader(lectorVel))
        figDiagFase = Figure(figsize = (5, 4), dpi = 110)
        axDiagFase = plt.Axes(figDiagFase, [0., 0., 1., 1.])
        axDiagFase.set_axis_off()
        frameDiagFase = Frame(pestanas, width = 800, height = 800)
        canvasDiagFase = FigureCanvasTkAgg(figDiagFase, master = frameDiagFase)
        canvasDiagFase.get_tk_widget().grid(row = 0, column = 0)
        pestanas.add(frameDiagFase, text = "Diag. Fase")
        sliderTiempo2 = Scale(frameDiagFase, from_ = 0, to = npasos_temporales/10, 
                             resolution = 0.1, orient = HORIZONTAL, length = 440, command = escalaTiempo2)
        sliderTiempo2.place(x=90, y=400)
        figDiagFase.add_axes(axDiagFase)
        
        colorDiagFaseVariableA = StringVar(frameDiagFase)
        colorDiagFaseVariableA.trace("w", cambiocolorDiagFaseA)
        colorDiagFaseVariableA.set(list(listaColor.keys())[2])
        listaColorDiagFaseA = OptionMenu(frameDiagFase, colorDiagFaseVariableA, *list(listaColor.keys()))
        listaColorDiagFaseA.place(x=0, y=380)

        colorDiagFaseVariableB = StringVar(frameDiagFase)
        colorDiagFaseVariableB.trace("w", cambiocolorDiagFaseB)
        colorDiagFaseVariableB.set(list(listaColor.keys())[0])
        listaColorDiagFaseB = OptionMenu(frameDiagFase, colorDiagFaseVariableB, *list(listaColor.keys()))
        listaColorDiagFaseB.place(x=0, y=410)
        # Grid: DiagFase
        if necesario_graficar:
            os.makedirs(ruta + "/DiagFase")
            vMax = numpy.max(numpy.array(vel[0], dtype = "double"))
            vMin = numpy.min(numpy.array(vel[0], dtype = "double"))
            xMax = numpy.max(numpy.array(posx[0], dtype = "double"))
            for i in range(1, npasos_temporales + 1):
                vMaxTemp = numpy.max(numpy.array(vel[i], dtype = "double"))
                vMinTemp = numpy.min(numpy.array(vel[i], dtype = "double"))
                xMaxTemp = numpy.max(numpy.array(posx[i], dtype = "double"))
                if vMax < vMaxTemp:
                    vMax = vMaxTemp
                if vMin > vMinTemp:
                    vMin = vMinTemp
                if xMax < xMaxTemp:
                    xMax = xMaxTemp
            # Grid
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0.175, 0.175, 0.8, 0.8])
            ax.set_ylim(vMin, vMax)
            ax.set_xlim(0, xMax)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("v (m/s)")
            fig.add_axes(ax)
            ax.grid()
            plt.savefig(ruta + "/DiagFase/Grid.png", transparent = True)
            barra.step(0)
            barra.stop()
            barra.start()
            root.update()
            vth = numpy.fromstring(ingvth.get(), dtype="double", sep = ",")
            for i in range(len(posx)):
                v1 = numpy.zeros(len(vel[0]))
                v2 = numpy.zeros(len(vel[0]))
                x1 = numpy.zeros(len(vel[0]))
                x2 = numpy.zeros(len(vel[0]))
                vx_aux = numpy.asarray(vel[i], dtype = 'double')
                x_aux = numpy.asarray(posx[i], dtype = 'double')
                for j in range(len(vel[0])):
                    if (vx_aux[j-1] > vx_aux[j]):
                        v1[j] = vx_aux[j]
                        x1[j] = x_aux[j]
                    elif(vx_aux[j-1] <= vx_aux[j]):
                        v2[j] = vx_aux[j]
                        x2[j] = x_aux[j]
                if vth[0] != 0:
                    ind1 = numpy.where(v1 != 0)
                    ind2 = numpy.where(v2 != 0)
                    v1 = v1[ind1]
                    v2 = v2[ind2]
                    x1 = x1[ind1]
                    x2 = x2[ind2]
                # Superior
                ax.cla()
                ax.set_ylim(vMin, vMax)
                ax.set_xlim(0, xMax)
                ax.set_xlabel("x (m)")
                ax.set_ylabel("v (m/s)")
                barra.step(50/(len(posx)+1))
                ax.scatter(x1, v1, s = 1, color = "black")
                ax.set_axis_off()
                plt.savefig(ruta + "/DiagFase/Sup" + str(i).zfill(int(numpy.floor(numpy.log10(len(vel)))+1)) + ".png", transparent = True)
                root.update()
                # Inferior
                ax.cla()
                ax.set_ylim(vMin, vMax)
                ax.set_xlim(0, xMax)
                ax.set_xlabel("x (m)")
                ax.set_ylabel("v (m/s)")
                barra.step(50/(len(posx)+1))
                ax.scatter(x2, v2, s = 1, color = "black")
                ax.set_axis_off()
                plt.savefig(ruta + "/DiagFase/Inf" + str(i).zfill(int(numpy.floor(numpy.log10(len(vel)))+1)) + ".png", transparent = True)
                labelBarra["text"] = "Diagrama de Fase " + str(i) + " de " + str(len(posx)-1)
                root.update()
            graficadoDiagFase = True

        axDiagFase.imshow(plt.imread(ruta + "/DiagFase/Grid.png", format = "RGBA"))
        axDiagFase.imshow(plt.imread(ruta + "/DiagFase/Inf" + str(0).zfill(int(numpy.floor(numpy.log10(len(vel)))+1)) + ".png", format = "RGBA"))
        axDiagFase.imshow(plt.imread(ruta + "/DiagFase/Sup" + str(0).zfill(int(numpy.floor(numpy.log10(len(vel)))+1)) + ".png", format = "RGBA"))
        escalaTiempo2(1)
        canvasDiagFase.draw()
        barra.step(99.9)
        labelBarra["text"] = "Gráficas Terminadas: Diag. Fase"
        cadena = " "
        root.update()

        lectorPosx.close()
        lectorVel.close()
        
        # Energías
        labelBarra["text"] = "Generando gráficas: Energías..."
        root.update()
        dx = longitud_malla/npuntos_malla
        nparticulas = int(ingnp.get())
        necesario_graficar = True
        if os.path.exists(ruta + "/Energias"):
            archivos = [name for name in os.listdir(ruta + "/Energias") if os.path.isfile(os.path.join(ruta + "/Energias", name))]
            if len(archivos) == 3:
                necesario_graficar = False
                graficadoEnergias = True
        if necesario_graficar:
            os.makedirs(ruta + "/Energias")
            Ekfile = open(ruta + '/Energias/Ek.csv', 'w')
            Epfile = open(ruta + '/Energias/Ep.csv', 'w')
            Etfile = open(ruta + '/Energias/Et.csv', 'w')
            with open(ruta + "/campoelectrico.csv") as lectorE:
                E = list(csv.reader(lectorE))
            if Bo == 0:
                with open(ruta + "/vel.csv") as lectorVel:
                    vel = list(csv.reader(lectorVel))
            else:
                with open(ruta + "/velx.csv") as lectorVel:
                    vel = list(csv.reader(lectorVel))
            Ek = numpy.zeros(npasos_temporales + 1)
            Ep = numpy.zeros(npasos_temporales + 1)
            Et = numpy.zeros(npasos_temporales + 1)
            for i in range(len(vel)):
                Ek[i] = 0.5*longitud_malla*numpy.sum(numpy.asarray(vel[i], dtype = 'double')**2)/nparticulas
                Ep[i] = 0.5*dx*numpy.sum(numpy.asarray(E[i], dtype = 'double')**2)
                Et[i] = Ek[i] + Ep[i]
            numpy.savetxt(Ekfile, [Ek], fmt = '%10.16f', delimiter = ',')
            numpy.savetxt(Epfile, [Ep], fmt = '%10.16f', delimiter = ',')
            numpy.savetxt(Etfile, [Et], fmt = '%10.16f', delimiter = ',')
            Ekfile.close()
            Epfile.close()
            Etfile.close()

        figEnergias = Figure(figsize=(5, 4), dpi = 110)
        axEnergias = plt.Axes(figEnergias, [0.2, 0.15, 0.725, 0.775])
        frameFigEnergias = Frame(pestanas, width = 800, height = 800)
        canvasEnergias = FigureCanvasTkAgg(figEnergias, master = frameFigEnergias)
        canvasEnergias.get_tk_widget().grid(row = 0, column = 0)
        pestanas.add(frameFigEnergias, text = "Energías")
        figEnergias.add_axes(axEnergias)

        colorEtVariable = StringVar(frameFigEnergias)
        colorEtVariable.trace("w", cambiocolorEt)
        colorEtVariable.set(list(listaColor.keys())[6])
        listaColorEt = OptionMenu(frameFigEnergias, colorEtVariable, *list(listaColor.keys()))
        listaColorEt.place(x=0, y=410)

        colorEpVariable = StringVar(frameFigEnergias)
        colorEpVariable.trace("w", cambiocolorEp)
        colorEpVariable.set(list(listaColor.keys())[0])
        listaColorEp = OptionMenu(frameFigEnergias, colorEpVariable, *list(listaColor.keys()))
        listaColorEp.place(x=0, y=380)

        colorEkVariable = StringVar(frameFigEnergias)
        colorEkVariable.trace("w", cambiocolorEk)
        colorEkVariable.set(list(listaColor.keys())[2])
        listaColorEp = OptionMenu(frameFigEnergias, colorEkVariable, *list(listaColor.keys()))
        listaColorEp.place(x=0, y=350)
        cadena1 = colorEkVariable.get()
        cadena2 = colorEpVariable.get()
        cadena3 = colorEtVariable.get()
        
        with open(ruta + '/Energias/Ek.csv') as archivoEk, \
            open(ruta + '/Energias/Ep.csv') as archivoEp, \
            open(ruta + '/Energias/Et.csv') as archivoEt:
            Ek = numpy.asarray(list(csv.reader(archivoEk, delimiter = ',')), dtype = "double")
            Ep = numpy.asarray(list(csv.reader(archivoEp, delimiter = ',')), dtype = "double")
            Et = numpy.asarray(list(csv.reader(archivoEt, delimiter = ',')), dtype = "double")

        t = 0.1*numpy.linspace(0, npasos_temporales, npasos_temporales + 1)
        axEnergias.cla()
        axEnergias.plot(t, Ek[0][:], color = listaColor[colorEkVariable.get()], label = 'E. Cinetica', linewidth = 3)
        axEnergias.plot(t, Ep[0][:], color = listaColor[colorEpVariable.get()], label = 'E. Potencial')
        axEnergias.plot(t, Et[0][:], color = listaColor[colorEtVariable.get()], label = 'E. Total')
        axEnergias.legend(loc = 4)
        axEnergias.grid()
        axEnergias.set_ylabel("Energía (J)")
        axEnergias.set_xlabel("Tiempo (s)")
        canvasEnergias.draw()
        graficadoEnergias = True

        # Rel. Disp.
        labelBarra["text"] = "Generando gráficas: Rel. Disp...."
        root.update()
        necesario_graficar = True
        if os.path.exists(ruta + "/Energias"):
            archivos = [name for name in os.listdir(ruta + "/Energias") if os.path.isfile(os.path.join(ruta + "/Energias", name))]
            if len(archivos) == 1:
                necesario_graficar = False
                graficadoEnergias = True
        omega_min = 2*numpy.pi/(0.1)/2/(npasos_temporales/2)
        omega_max = omega_min*(npasos_temporales/2)
        k_min = 2*numpy.pi/(npuntos_malla)
        k_max = k_min*((npuntos_malla/2)-1)
        k_t = numpy.linspace(0, k_max, nparticulas)
        k_simulada = numpy.linspace(-k_max, k_max, 2**int(numpy.log2(npuntos_malla)+1))
        omega_t = numpy.linspace(0, longitud_malla, npuntos_malla + 1)
        omega_simulada = numpy.linspace(-omega_max, omega_max, 2**int(numpy.log2(npuntos_malla) + 1))
        if necesario_graficar:
            K, W = numpy.meshgrid(k_simulada, omega_simulada)
            if not os.path.exists(ruta + "/RelDisp"):
                os.makedirs(ruta + "/RelDisp")
                n2pmalla = 2**int(numpy.log2(npuntos_malla) + 1)
                with open(ruta + "/RelDisp/Ewk.csv", 'w') as archivoEwk:
                    Ewk = shift(abs(fft2(numpy.array(E, dtype = "double"), 
                                                    (n2pmalla,n2pmalla))/longitud_malla))
                    for i in range(n2pmalla):
                        numpy.savetxt(archivoEwk, 
                                    [Ewk[i]], 
                                    fmt = '%10.16f', delimiter = ',')

        with open(ruta + '/RelDisp/Ewk.csv') as archivoEwk:
                Ewk = numpy.array(numpy.array(list(csv.reader(archivoEwk, 
                                                                delimiter = ','))), 
                                    dtype = 'double')
        
        figRelDisp = Figure(figsize=(5, 4), dpi = 110)
        axRelDisp = plt.Axes(figRelDisp, [0.2, 0.15, 0.725, 0.775])
        frameFigRelDisp = Frame(pestanas, width = 800, height = 800)
        canvasRelDisp = FigureCanvasTkAgg(figRelDisp, master = frameFigRelDisp)
        canvasRelDisp.get_tk_widget().grid(row = 0, column = 0)
        pestanas.add(frameFigRelDisp, text = "Rel. Dispersión")
        figRelDisp.add_axes(axRelDisp)
        lim = numpy.max(numpy.max(Ewk))/8
        limymax = 0
        limymin = len(Ewk[0,:])
        limxmax = 0
        limxmin = limymin
        for i in range(len(Ewk)): # Cuadrado
            ry = Ewk[i,:] > lim
            rx = Ewk[:,i] > lim
            if numpy.sum(ry) != 0:
                if numpy.min(numpy.where(ry)) < limymin:
                    limymin = numpy.min(numpy.where(ry))
                if numpy.max(numpy.where(ry)) > limymax:
                    limymax = numpy.max(numpy.where(ry))
            if numpy.sum(rx) != 0:
                if numpy.min(numpy.where(rx)) < limxmin:
                    limxmin = numpy.min(numpy.where(rx))
                if numpy.max(numpy.where(rx)) > limxmax:
                    limxmax = numpy.max(numpy.where(rx))
        m = axRelDisp.contourf(K, W, Ewk, 8, alpha = .75, cmap = 'jet')
        axRelDisp.set_xlim(k_simulada[limymin], k_simulada[limymax])
        axRelDisp.set_ylim(omega_simulada[limxmin], omega_simulada[limxmax])
        figRelDisp.colorbar(m, ax = axRelDisp)
        axRelDisp.set_xlabel('k')
        axRelDisp.set_ylabel('$\omega$')

        # Ciclos
        if Bo != 0:
            labelBarra["text"] = "Generando gráficas: Ciclos..."
            root.update()
            with open(ruta + "/posy.csv") as lectorPosy:
                posy = list(csv.reader(lectorPosy))
            figCiclos = Figure(figsize=(5, 4), dpi = 110)
            axCiclos = plt.Axes(figCiclos, [0.2, 0.15, 0.725, 0.775])
            frameFigCiclos = Frame(pestanas, width = 800, height = 800)
            canvasCiclos = FigureCanvasTkAgg(figCiclos, master = frameFigCiclos)
            canvasCiclos.get_tk_widget().grid(row = 0, column = 0)
            pestanas.add(frameFigCiclos, text = "Ciclos")
            x1 = numpy.array(posx, dtype = "double")
            y1 = numpy.array(posy, dtype = "double")
            axCiclos.plot(x1[:, int(0.375*nparticulas)], y1[:, int(0.375*nparticulas)])
            axCiclos.plot(x1[:, int(0.50*nparticulas)], y1[:, int(0.50*nparticulas)])
            axCiclos.plot(x1[:, int(0.625*nparticulas)], y1[:, int(0.625*nparticulas)])
            axCiclos.grid()
            figCiclos.add_axes(axCiclos)
            axCiclos.set_xlabel('x (m)')
            axCiclos.set_ylabel('y (m)')
        
        labelBarra["text"] = "Finalizado."
        barra["mode"] = "determinate"
        barra.stop()
        root.update()

        btnSimular["state"] = "disabled"
    return

def cambiocolorFDist(*args):
    actualizar()

def cambiocolorCE(*args):
    escalaTiempo(1)

def escalaTiempo(var):
    global sliderTiempo, graficadoCE, axCE, npasos_temporales, ruta, canvasCE, colorCEVariable, listaColor
    if graficadoCE:
        im = plt.imread(ruta + "/CElec/Elec" + str(int(10*sliderTiempo.get())).zfill(int(numpy.floor(numpy.log10(npasos_temporales))+1)) + ".png", format = "RGBA")
        im[:,:,[0, 1, 2]] = listaColor[colorCEVariable.get()]
        axCE.cla()
        axCE.set_axis_off()
        axCE.imshow(plt.imread(ruta + "/CElec/Grid.png", format = "RGBA"))
        axCE.imshow(im)
        canvasCE.draw()

def cambiocolorDiagFaseA(*args):
    escalaTiempo2(1)

def cambiocolorDiagFaseB(*args):
    global cadena
    cadena = colorDiagFaseVariableB.get()
    escalaTiempo2(1)

def escalaTiempo2(var):
    global sliderTiempo2, graficadoDiagFase, axDiagFase, npasos_temporales, ruta, canvasDiagFase
    global colorDiagFaseVariableA, listaColor, cadena
    if graficadoDiagFase:
        iminf = plt.imread(ruta + "/DiagFase/Inf" + str(int(10*sliderTiempo2.get())).zfill(int(numpy.floor(numpy.log10(npasos_temporales))+1)) + ".png", format = "RGBA")
        iminf[:, :, [0, 1, 2]] = listaColor[colorDiagFaseVariableA.get()]
        imsup = plt.imread(ruta + "/DiagFase/Sup" + str(int(10*sliderTiempo2.get())).zfill(int(numpy.floor(numpy.log10(npasos_temporales))+1)) + ".png", format = "RGBA")
        imsup[:, :, [0, 1, 2]] = listaColor[cadena]
        axDiagFase.cla()
        axDiagFase.set_axis_off()
        axDiagFase.imshow(plt.imread(ruta + "/DiagFase/Grid.png", format = "RGBA"))
        axDiagFase.imshow(imsup)
        axDiagFase.imshow(iminf)
        canvasDiagFase.draw()

def cambiocolorEk(*args):
    global cadena1
    cadena1 = colorEkVariable.get()
    actualizar2()

def cambiocolorEp(*args):
    global cadena2
    cadena2 = colorEpVariable.get()
    actualizar2()

def cambiocolorEt(*args):
    global cadena3
    cadena3 = colorEtVariable.get()
    actualizar2()

def actualizar2():
    global axEnergias, canvasEnergias, cadena1, cadena2, cadena3
    global listaColor, npasos_temporales, graficadoEnergias
    if graficadoEnergias:
        t = 0.1*numpy.linspace(0, npasos_temporales, npasos_temporales + 1)
        axEnergias.cla()
        with open(ruta + '/Energias/Ek.csv') as archivoEk, \
                open(ruta + '/Energias/Ep.csv') as archivoEp, \
                open(ruta + '/Energias/Et.csv') as archivoEt:
            Ek = numpy.asarray(list(csv.reader(archivoEk, delimiter = ',')), dtype = "double")
            Ep = numpy.asarray(list(csv.reader(archivoEp, delimiter = ',')), dtype = "double")
            Et = numpy.asarray(list(csv.reader(archivoEt, delimiter = ',')), dtype = "double")
        axEnergias.plot(t, Ek[0][:], color = listaColor[cadena1], label = 'E. Cinetica', linewidth = 3)
        axEnergias.plot(t, Ep[0][:], color = listaColor[cadena2], label = 'E. Potencial')
        axEnergias.plot(t, Et[0][:], color = listaColor[cadena3], label = 'E. Total')
        axEnergias.set_ylabel("Energía (J)")
        axEnergias.set_xlabel("Tiempo (s)")
        axEnergias.grid()
        axEnergias.legend(loc = 4)
        canvasEnergias.draw()

def CargarEjemplo1():
    ingci.delete(0, END)
    ingci.insert(0, "1")
    ingvh.delete(0, END)
    ingvh.insert(0, "1")
    ingvth.delete(0, END)
    ingvth.insert(0, "0")
    ingnp.delete(0, END)
    ingnp.insert(0, "2048")
    ingnm.delete(0, END)
    ingnm.insert(0, "256")
    ingst.delete(0, END)
    ingst.insert(0, "150")
    inglm.delete(0, END)
    inglm.insert(0, "4")
    pivar.set(1)
    ingBo.delete(0, END)
    ingBo.insert(0, "0.0")
    ingxP.delete(0, END)
    ingxP.insert(0, "0.001")
    ingvP.delete(0, END)
    ingvP.insert(0, "0.0")
    return

def CargarEjemplo2():
    ingci.delete(0, END)
    ingci.insert(0, "0.5,0.5")
    ingvh.delete(0, END)
    ingvh.insert(0, "-2.5,2.5")
    ingvth.delete(0, END)
    ingvth.insert(0, "1,1")
    ingnp.delete(0, END)
    ingnp.insert(0, "20000")
    ingnm.delete(0, END)
    ingnm.insert(0, "1024")
    ingst.delete(0, END)
    ingst.insert(0, "150")
    inglm.delete(0, END)
    inglm.insert(0, "32")
    pivar.set(1)
    ingBo.delete(0, END)
    ingBo.insert(0, "0.0")
    ingxP.delete(0, END)
    ingxP.insert(0, "0.0")
    ingvP.delete(0, END)
    ingvP.insert(0, "0.5")
    return

def CargarEjemplo3():
    ingci.delete(0, END)
    ingci.insert(0, "0.9,0.1")
    ingvh.delete(0, END)
    ingvh.insert(0, "-0.5,4")
    ingvth.delete(0, END)
    ingvth.insert(0, "1.0,1.0")
    ingnp.delete(0, END)
    ingnp.insert(0, "16384")
    ingnm.delete(0, END)
    ingnm.insert(0, "128")
    ingst.delete(0, END)
    ingst.insert(0, "300")
    inglm.delete(0, END)
    inglm.insert(0, "16")
    pivar.set(1)
    ingBo.delete(0, END)
    ingBo.insert(0, "0.0")
    ingxP.delete(0, END)
    ingxP.insert(0, "0.0")
    ingvP.delete(0, END)
    ingvP.insert(0, "0.0")
    return

def CargarEjemplo4():
    ingci.delete(0, END)
    ingci.insert(0, "0.33,0.34,0.33")
    ingvh.delete(0, END)
    ingvh.insert(0, "-5,0,5")
    ingvth.delete(0, END)
    ingvth.insert(0, "1.0,1.0,1.0")
    ingnp.delete(0, END)
    ingnp.insert(0, "50000")
    ingnm.delete(0, END)
    ingnm.insert(0, "1024")
    ingst.delete(0, END)
    ingst.insert(0, "450")
    inglm.delete(0, END)
    inglm.insert(0, "48")
    pivar.set(1)
    ingBo.delete(0, END)
    ingBo.insert(0, "0.0")
    ingxP.delete(0, END)
    ingxP.insert(0, "0.0")
    ingvP.delete(0, END)
    ingvP.insert(0, "0.0")
    return

def CargarEjemplo5():
    ingci.delete(0, END)
    ingci.insert(0, "0.3,0.2,0.2,0.3")
    ingvh.delete(0, END)
    ingvh.insert(0, "-7.5,-2.5,2.5,7.5")
    ingvth.delete(0, END)
    ingvth.insert(0, "1.0,1.0,1.0,1.0")
    ingnp.delete(0, END)
    ingnp.insert(0, "50000")
    ingnm.delete(0, END)
    ingnm.insert(0, "1024")
    ingst.delete(0, END)
    ingst.insert(0, "450")
    inglm.delete(0, END)
    inglm.insert(0, "48")
    pivar.set(1)
    ingBo.delete(0, END)
    ingBo.insert(0, "0.0")
    ingxP.delete(0, END)
    ingxP.insert(0, "0.0")
    ingvP.delete(0, END)
    ingvP.insert(0, "0.0")
    return

def CargarEjemplo6():
    ingci.delete(0, END)
    ingci.insert(0, "0.3,0.15,0.1,0.15,0.3")
    ingvh.delete(0, END)
    ingvh.insert(0, "-10,-5,0,5,10")
    ingvth.delete(0, END)
    ingvth.insert(0, "1,1,1,1,1")
    ingnp.delete(0, END)
    ingnp.insert(0, "50000")
    ingnm.delete(0, END)
    ingnm.insert(0, "1024")
    ingst.delete(0, END)
    ingst.insert(0, "300")
    inglm.delete(0, END)
    inglm.insert(0, "48")
    pivar.set(1)
    ingBo.delete(0, END)
    ingBo.insert(0, "0.0")
    ingxP.delete(0, END)
    ingxP.insert(0, "0.0")
    ingvP.delete(0, END)
    ingvP.insert(0, "0.0")
    return

def CargarEjemplo7():
    ingci.delete(0, END)
    ingci.insert(0, "0.5,0.5")
    ingvh.delete(0, END)
    ingvh.insert(0, "-2.5,2.5")
    ingvth.delete(0, END)
    ingvth.insert(0, "1,1")
    ingnp.delete(0, END)
    ingnp.insert(0, "50000")
    ingnm.delete(0, END)
    ingnm.insert(0, "1024")
    ingst.delete(0, END)
    ingst.insert(0, "300")
    inglm.delete(0, END)
    inglm.insert(0, "48")
    pivar.set(1)
    ingBo.delete(0, END)
    ingBo.insert(0, "5.0")
    ingxP.delete(0, END)
    ingxP.insert(0, "0.0")
    ingvP.delete(0, END)
    ingvP.insert(0, "0.0")
    return

if not os.path.exists("simulaciones"):
    os.makedirs("simulaciones")

##### Ventana Principal #####
root = Tk()
root.title("ESMG1-GPLA")
root.geometry("850x480+150+50")

##### Objetos de la ventana principal #####

# Barra de menús
barramenu = Menu(root)
root.config(menu = barramenu)
opciones1 = Menu(barramenu, tearoff = 0)
opciones1.add_command(label = "Nuevo", command = NuevaSimul)
opciones1.add_command(label = "Abrir...", command = AbrirSimul)
opciones1.add_separator()
opciones1.add_command(label = "Salir", command = cierre)
barramenu.add_cascade(label = "Archivo", menu = opciones1)

ejemplos = Menu(barramenu, tearoff = 0)
ejemplos.add_command(label = "Plasma Frío", command = CargarEjemplo1)
ejemplos.add_command(label = "Two-Stream", command = CargarEjemplo2)
ejemplos.add_command(label = "Bump-On-Tail", command = CargarEjemplo3)
ejemplos.add_command(label = "Tres Flujos", command = CargarEjemplo4)
ejemplos.add_command(label = "Cuatro Flujos", command = CargarEjemplo5)
ejemplos.add_command(label = "Cinco Flujos", command = CargarEjemplo6)
ejemplos.add_command(label = "Bi-Contención", command = CargarEjemplo7)
barramenu.add_cascade(label = "Ejemplos", menu = ejemplos)

frameIzq = Frame(root, width = 50, height = 50)
frameIzq.config(cursor = "")
frameIzq.config(relief = "groove")
frameIzq.config(bd = 5)
frameIzq.grid(row = 0, column = 0)

var1 = StringVar()
var1.trace("w", lambda name, index,mode, var1=var1: callback(var1))
var2 = StringVar()
var2.trace("w", lambda name, index,mode, var2=var2: callback(var2))
var3 = StringVar()
var3.trace("w", lambda name, index,mode, var3=var3: callback(var3))
var4 = StringVar()
var4.trace("w", lambda name, index,mode, var4=var4: nada(var4))
var5 = StringVar()
var5.trace("w", lambda name, index,mode, var5=var5: nada(var5))
var6 = StringVar()
var6.trace("w", lambda name, index,mode, var6=var6: nada(var6))
var7 = StringVar()
var7.trace("w", lambda name, index,mode, var7=var7: nada(var7))
var8 = StringVar()
var8.trace("w", lambda name, index,mode, var8=var8: nada(var8))
var9 = StringVar()
var9.trace("w", lambda name, index,mode, var9=var9: nada(var9))
var10 = StringVar()
var10.trace("w", lambda name, index,mode, var10=var10: nada(var10))

label = Label(frameIzq, text = "Parámetros de simulación", font=("Helvetica", 12, "bold"))
label.grid(row = 0, column = 0, columnspan = 5)

label = Label(frameIzq, text = " ")
label.grid(row = 1, column = 0, columnspan = 5)

label = Label(frameIzq, text = espacio)
label.grid(row = 2, column = 0)
label = Label(frameIzq, text = "Coeficientes", font=("Helvetica", 10))
label.grid(row = 2, column = 1)
ingci = Entry(frameIzq, width = 15, textvariable=var1)
ingci.grid(row = 2, column = 2, columnspan = 2)
label = Label(frameIzq, text = espacio)
label.grid(row = 2, column = 4)

label = Label(frameIzq, text = espacio)
label.grid(row = 3, column = 0)
label = Label(frameIzq, text = "Velocidades Medias", font=("Helvetica", 10))
label.grid(row = 3, column = 1)
ingvh = Entry(frameIzq, width = 15, textvariable=var2)
ingvh.grid(row = 3, column = 2, columnspan = 2)
label = Label(frameIzq, text = espacio)
label.grid(row = 3, column = 4)

label = Label(frameIzq, text = espacio)
label.grid(row = 4, column = 0)
label = Label(frameIzq, text = "Velocidades Térmicas", font=("Helvetica", 10))
label.grid(row = 4, column = 1)
ingvth = Entry(frameIzq, width = 15, textvariable=var3)
ingvth.grid(row = 4, column = 2, columnspan = 2)
label = Label(frameIzq, text = espacio)
label.grid(row = 4, column = 4)

label = Label(frameIzq, text = " ")
label.grid(row = 5, column = 0, columnspan = 5)

label = Label(frameIzq, text = espacio)
label.grid(row = 6, column = 0)
label = Label(frameIzq, text = "Número de Partículas", font=("Helvetica", 10))
label.grid(row = 6, column = 1)
ingnp = Entry(frameIzq, width = 15, textvariable=var4)
ingnp.grid(row = 6, column = 2, columnspan = 2)
label = Label(frameIzq, text = espacio)
label.grid(row = 6, column = 4)

label = Label(frameIzq, text = espacio)
label.grid(row = 7, column = 0)
label = Label(frameIzq, text = "Puntos de Malla", font=("Helvetica", 10))
label.grid(row = 7, column = 1)
ingnm = Entry(frameIzq, width = 15, textvariable=var5)
ingnm.grid(row = 7, column = 2, columnspan = 2)
label = Label(frameIzq, text = espacio)
label.grid(row = 7, column = 4)

label = Label(frameIzq, text = espacio)
label.grid(row = 8, column = 0)
label = Label(frameIzq, text = "Pasos Temporales", font=("Helvetica", 10))
label.grid(row = 8, column = 1)
ingst = Entry(frameIzq, width = 15, textvariable=var6)
ingst.grid(row = 8, column = 2, columnspan = 2)
label = Label(frameIzq, text = espacio)
label.grid(row = 8, column = 4)

label = Label(frameIzq, text = espacio)
label.grid(row = 9, column = 0)
label = Label(frameIzq, text = "Longitud de Malla", font=("Helvetica", 10))
label.grid(row = 9, column = 1)
inglm = Entry(frameIzq, width = 15, textvariable=var7)
inglm.grid(row = 9, column = 2, columnspan = 2)
label = Label(frameIzq, text = espacio)
label.grid(row = 9, column = 4)
pivar = IntVar()
checkpi = Checkbutton(frameIzq, text = "π", variable = pivar)
checkpi.grid(row = 9, column = 4)

label = Label(frameIzq, text = espacio)
label.grid(row = 10, column = 0)
label = Label(frameIzq, text = "Magnitud C. Magnético", font=("Helvetica", 10))
label.grid(row = 10, column = 1)
ingBo = Entry(frameIzq, width = 15, textvariable=var8)
ingBo.grid(row = 10, column = 2, columnspan = 2)
label = Label(frameIzq, text = espacio)
label.grid(row = 10, column = 4)

label = Label(frameIzq, text = espacio)
label.grid(row = 11, column = 0)
label = Label(frameIzq, text = "Amplitud Pert. Pos.", font=("Helvetica", 10))
label.grid(row = 11, column = 1)
ingxP = Entry(frameIzq, width = 15, textvariable=var9)
ingxP.grid(row = 11, column = 2, columnspan = 2)
label = Label(frameIzq, text = espacio)
label.grid(row = 11, column = 4)

label = Label(frameIzq, text = espacio)
label.grid(row = 12, column = 0)
label = Label(frameIzq, text = "Amplitud Pert. Vel.", font=("Helvetica", 10))
label.grid(row = 12, column = 1)
ingvP = Entry(frameIzq, width = 15, textvariable=var10)
ingvP.grid(row = 12, column = 2, columnspan = 2)
label = Label(frameIzq, text = espacio)
label.grid(row = 12, column = 4)

label = Label(frameIzq, text = " ")
label.grid(row = 13, column = 0, columnspan = 5)

btnSimular = Button(frameIzq, text = "Simular...", font=("Helvetica", 10), 
                    width = ancho, height = alto, command = runSimul)
btnSimular.grid(row = 14, column = 0, columnspan = 5)

label = Label(frameIzq, text = " ")
label.grid(row = 15, column = 0, columnspan = 5)

reg = root.register(validar)
ingci.config(validate = "key", validatecommand = (reg, '%P'))
ingvh.config(validate = "key", validatecommand = (reg, '%P'))
ingvth.config(validate = "key", validatecommand = (reg, '%P'))

pestanas = Notebook(root)
figDist = Figure(figsize=(5, 4), dpi = 110)
ax = figDist.add_subplot(111)
ax.set_ylim(0, 1)

frameFigDist = Frame(pestanas, width = 800, height = 800)

canvas = FigureCanvasTkAgg(figDist, master = frameFigDist)
#canvas.show()
canvas.get_tk_widget().grid(row = 0, column = 0)
listaColor = {"Azul": [0, 0, 1], "Verde": [0, 1, 0], "Rojo": [1, 0, 0], 
               "Cyan": [0, 1, 1], "Magenta": [1, 0, 1], "Amarillo": [1, 1, 0], "Negro": [0, 0, 0],
               "Naranja": [1, 0.5, 0], "Venom": [0.5, 0.5, 0], "Lima": [0.5, 1, 0], "Cobalto": [0, 0.5, 1],
               "Aqua": [0, 0.5, 0.5], "Menta": [0, 1, 0.5], "Indigo": [0.5, 0, 1], 
               "Purpura": [0.5, 0, 0.5], "Crimson": [1, 0, 0.5], "Mostaza": [0.913725, 0.741176, 0.058824]}
colorFDistVariable = StringVar(frameFigDist)
colorFDistVariable.trace("w", cambiocolorFDist)
colorFDistVariable.set(list(listaColor.keys())[0])
listaColorFDist = OptionMenu(frameFigDist, colorFDistVariable, *list(listaColor.keys()))
listaColorFDist.place(x=10, y=10)
pestanas.add(frameFigDist, text = "F. Dist. Inicial")
pestanas.grid(row = 0, column = 1, rowspan = 3)

ruta = " "
cadena = "Verde"
cadena1 = "Rojo"
cadena2 = "Azul"
cadena3 = "Negro"
directorio_open = ""
apertura = False
graficadoCE = False
graficadoDiagFase = False
graficadoEnergias = False
labelBarra =Label(root, text = "", font = ("Helvetica", 10))
labelBarra.grid(row = 1, column = 0)
barra = Progressbar(root, length = 200)
barra.grid(row = 2, column = 0)

root.protocol("WM_DELETE_WINDOW", cierre)
root.resizable(False, False)
root.mainloop()
