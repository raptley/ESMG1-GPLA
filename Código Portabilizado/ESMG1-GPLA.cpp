#include <iostream>
#include <math.h>
#include <string.h>
#include <filesystem>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>
#define PI 3.14159265358979323846

// Distribuye las partículas de manera uniforme en la malla comp.
void cargarposicion(int n, double l, double pP, double pos[]) {
    double esp = l / n;
    for (int i = 0; i < n; i++) {
        pos[i] = esp * (i + 0.5);
        // Se aplica la perturbación en la posición.
        pos[i] += pP * cos(pos[i]);
    }
}

// Mezcla las posiciones de las partículas para distribuir las vels.
void comb_pos(int r[], int n) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    for (int i = 0; i < n; i++) r[i] = i;
    std::shuffle(&r[0], &r[n - 1], std::default_random_engine(seed));
}

// Asignación de velocidad usando método de muestreo inverso.
/* rand() tiene fallos... demasiados. no son valores uniformemente 
distribuidos o bajo una funcion de distribución normal. 
Esta es gran fuente de separación de las vels. 
Por ello, se opta por una alternativa un poco más segura. 
En 20k partículas, se nota una rara singularidad: un 0 aleatorio. */
void cargarvelocidad(const int n, double l, int ns,
    double coef[], double vhs[], double vths[],
    double vP, double vel[], int xr[]) {
    std::random_device rd; // Mejores generadores de aleatoriedad
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    int lim_inf = 0, lim_sup = 0;
    double u;
    int signo = 0;
    if (vths[0] != 0) { // Plasma Frío: Delta de Dirac, vth -> 0
        for (int i = 0; i < ns; i++) {
            lim_sup += coef[i] * n;
            for (int j = lim_inf; j < lim_sup; j++) {
                signo = pow(-1, rand() % 2);
                u = 1-dis(gen);
                vel[xr[j]] = signo * vths[i] * sqrt(2 * abs(log(1 / u))) + vhs[i];
                // Se aplica la perturbación de la velocidad
                vel[xr[j]] += vP * sin(2 * PI * vel[xr[j]] / l);
            }
            lim_inf = lim_sup;
        }
    }
}

// Condiciones de frontera: se hacen periódicas
void particula_cf(int n, double dim, double pos[]) {
    for (int i = 0; i < n; i++) {
        if (pos[i] < 0.0)
            pos[i] += dim; // o  |------| -> |---o--|
        if (pos[i] >= dim)
            pos[i] -= dim; // |------| o -> |-o----|
    }
}

// Se calcula la densidad de carga en cada punto de la malla
void densidad(int n, int puntos, double pasox, double qe, 
    double pos[], double dens_e[], int itiempo) {
    double re = qe / pasox;        // Factor de ponderación
    double f1, f2, cuadrante;
    int posicion;
    for (int i = 0; i < n; i++) {       // Mapa de cargas sobre la malla
        cuadrante = pos[i] / pasox;     // xparticula/dx: pos. de la partícula normalizada a n. celdas
        posicion = cuadrante;           // indice de la malla fija
        f2 = cuadrante - posicion;      // factor de ponderación |xmalla - xparticula|/dx
        f1 = 1 - f2;
        dens_e[posicion] += re * f1;
        dens_e[posicion + 1] += re * f2;
    }
    dens_e[0] += dens_e[puntos];
    dens_e[puntos] = dens_e[0];         // Condiciones de frontera periódicas
}

// Campo eléctrico en cada punto de la malla
void campo(int puntos, double dens_t[], double pasox, double E[]) {
    // Algoritmo de integración acumulativa pro teorema del trapezoide
    double sum = 0, prom;
    E[0] = 0;           // E[0] = 0, pues no se da un valor inicial.
    for (int i = 1; i <= puntos; i++) {
        sum += (dens_t[i] + dens_t[i - 1]) * pasox / 2;
        E[i] = sum;     // yi = Sum(i = 0, i) ((yi+y(i-1))*(dx)/2)
    }
    sum = 0;
    for (int i = 0; i <= puntos; i++) {
        sum += E[i];
    }
    prom = sum / (double(puntos) + 1);
    for (int i = 0; i < puntos - 1; i++) {
        E[i] -= prom;   // Remover componente DC.
    }
    E[puntos] = E[0];   // Aparentemente esto no es necesario
}

// Mvto. sin campo magnético.
void mvtoparticulas(int n, double pos[], double vel[], double E[], double pasot, double pasox, double qm) {
    double f1, f2, cuadrante, contribE;
    int posicion;
    for (int i = 0; i < n; i++) {   // Mismos pasos que en densidad
        cuadrante = pos[i] / pasox;
        posicion = cuadrante;
        f2 = cuadrante - posicion;
        f1 = 1 - f2;
        contribE = f1 * E[posicion] + f2 * E[posicion + 1];
        vel[i] += qm * pasot * contribE;
        pos[i] += pasot * vel[i];   // Actualizar pos. (2do paso leap frog)
    }
}

// Mvto. con campo magnético aplicado
void mvtoparticulas(int n, double posx[], double posy[], double velx[],
    double vely[], double E[], double pasot, double pasox, double qm, double Bo) {
    double vmenosx, vmenosy, vprimax, vprimay, vmasx, vmasy;
    double f1, f2, cuadrante, contribE;
    // Originalmente hecho con vectores. Al ver los resultados, ninguna componente en z cambia.
    // Por tanto, se presentan cálculos simplificados
    int posicion;
    double t = qm * Bo * pasot * 0.5;
    double s = 2 * t / (1 + (pow(t, 2)));
    for (int i = 0; i < n; i++) {           // Algoritmo de Boris
        cuadrante = posx[i] / pasox;
        posicion = cuadrante;
        f2 = cuadrante - posicion;
        f1 = 1 - f2;
        contribE = f1 * E[posicion] + f2 * E[posicion + 1];
        vmenosx = velx[i] + qm * pasot * contribE * 0.5;
        vmenosy = vely[i];
        vprimax = vmenosx + vmenosy * t;
        vprimay = vmenosy - vmenosx * t;
        vmasx = vmenosx + vprimay * s;
        vmasy = vmenosy - vprimax * s;
        velx[i] = vmasx + qm * pasot * contribE * 0.5;
        vely[i] = vmasy;
        posx[i] += pasot * velx[i];
        posy[i] += pasot * vely[i];
    }
}

// Programa principal
int main(int argc, char* argv[])
{
    // Reloj de inicio
    auto inicio = std::chrono::system_clock::now();

    // Especies
    const int n_especies = 2;
    double c_n[] = { 0.9, 0.1 };
    double vh[] = { -0.5, 4.0 };
    double vth[] = { 1.0, 1.0 };

    // Parámetros de simulación
    const int nparticulas       = 16384;
    const int npuntos_malla     = 128;
    const int npasos_temporales = 1200;
    double longitud_malla       = 16*PI;
    double Bo                   = 0.0;
    double aP                   = 0.0;
    double vP                   = 0.0;

    double carga_masa = -1.0;
    double dx = longitud_malla / npuntos_malla;
    double dt = 0.1;

    double x[nparticulas];
    double y[nparticulas];
    double v[nparticulas] = {0};
    double vx[nparticulas];
    double vy[nparticulas];
    double Ex[npuntos_malla + 1];
    double rhoe[npuntos_malla + 1] = {0};
    double rhoi = 1.0;
    double rhot[npuntos_malla + 1];

    // Creacion de la carpeta para cada parámetro
    bool CrearCarpeta = false;
    int folder = 1;
    std::string ruta, nombre, directorio;
    directorio = std::filesystem::current_path().string();
    while (!CrearCarpeta) {
        nombre = "Run " + std::to_string(folder);
        ruta = std::filesystem::current_path().string() + "\\" + nombre;
        if (!std::filesystem::exists(std::filesystem::path(ruta))) {
            CrearCarpeta = true;
            std::filesystem::create_directory(std::filesystem::path(ruta));
        }
        folder++;
    }

    // Configuración inicial de la distribucion de partículas y campos:
    cargarposicion(nparticulas, longitud_malla, aP, x);
    int xr[nparticulas];
    comb_pos(xr, nparticulas);
    cargarvelocidad(nparticulas, longitud_malla, n_especies, c_n, vh, vth, vP, v, xr);
    for (int i = 0; i < nparticulas; i++) x[i] += 0.5*dt*v[i];
    particula_cf(nparticulas, longitud_malla, x);

    double q = -rhoi * longitud_malla / nparticulas;
    densidad(nparticulas, npuntos_malla, dx, q, x, rhoe, 0);
    for (int i = 0; i < npuntos_malla + 1; i++) rhot[i] = rhoe[i] + rhoi;
    campo(npuntos_malla, rhot, dx, Ex);

    // Ciclo principal: CIC
    // Se crean los archivos base, estos son los mínimos requeridos
    //std::ofstream archivoX(ruta + "\\posx.csv");
//    std::ofstream archivoEx(ruta + "\\campoelectrico.csv");
//    std::ofstream archivorhoe(ruta + "\\densidadcarga.csv");
//    for (int i = 0; i < nparticulas; i++) {
//        archivoX << std::fixed << x[i];
//        if (i < nparticulas - 1)
//            archivoX << ",";
//        else
//            archivoX << std::endl;
//    }
//    for (int i = 0; i < npuntos_malla + 1; i++) {
//        archivoEx << std::fixed << Ex[i];
//        archivorhoe << std::fixed << rhoe[i];
//        if (i < npuntos_malla) {
//            archivoEx << ",";
//            archivorhoe << ",";
//        }
//        else {
//            archivoEx << std::endl;
//            archivorhoe << std::endl;
//        }
//    }
//    archivoX.close();
//    archivoEx.close();
//    archivorhoe.close();
    // Se reabren en modo append
    //archivoX.open(ruta + "\\posx.csv", std::ios_base::app);
//    archivoEx.open(ruta + "\\campoelectrico.csv", std::ios_base::app);
//    archivorhoe.open(ruta + "\\densidadcarga.csv", std::ios_base::app);

    // for -> if: ineficiente. if -> for: eficiente
    if (Bo == 0) {
        //std::ofstream archivov(ruta + "\\vel.csv");
//        for (int i = 0; i < nparticulas; i++) {
//            archivov << std::fixed << v[i];
//            if (i < nparticulas - 1)
//                archivov << ",";
//            else
//                archivov << std::endl;
//        }
//        archivov.close();
        // Se reabre en modo append:
        //archivov.open(ruta + "\\vel.csv", std::ios_base::app);
        for (int itiempo = 1; itiempo <= npasos_temporales; itiempo++) {
            std::cout << "Paso: " << itiempo << std::endl;
            mvtoparticulas(nparticulas, x, v, Ex, dt, dx, carga_masa);
            particula_cf(nparticulas, longitud_malla, x);
            for (int i = 0; i < npuntos_malla + 1; i++) rhoe[i] = 0;
            densidad(nparticulas, npuntos_malla, dx, q, x, rhoe, itiempo);
            for (int i = 0; i < npuntos_malla + 1; i++) rhot[i] = rhoe[i] + rhoi;
            campo(npuntos_malla, rhot, dx, Ex);
            //for (int i = 0; i < nparticulas; i++) {
//                archivoX << std::fixed << x[i];
//                archivov << std::fixed << v[i];
//                if (i < nparticulas - 1){
//                    archivoX << ",";
//                    archivov << ",";
//                }
//                else {
//                    archivoX << ",";
//                    archivov << std::endl;
//                }
//            }
//            for (int i = 0; i < npuntos_malla + 1; i++) {
//                archivoEx << std::fixed << Ex[i];
//                archivorhoe << std::fixed << rhoe[i];
//                if (i < npuntos_malla) {
//                    archivoEx << ",";
//                    archivorhoe << ",";
//                }
//                else {
//                    archivoEx << std::endl;
//                    archivorhoe << std::endl;
//                }
//            }
        }
//        archivov.close();
    }
    else {
        //std::ofstream archivovx(ruta + "\\velx.csv", std::ios_base::app);
//        std::ofstream archivovy(ruta + "\\vely.csv", std::ios_base::app);
//        std::ofstream archivoY(ruta + "\\posy.csv", std::ios_base::app);
//        for (int i = 0; i < nparticulas; i++) { 
//            vx[i] = v[i] * cos(2 * PI * x[i] / longitud_malla);
//            vy[i] = v[i] * sin(2 * PI * x[i] / longitud_malla);
//            y[i] = dt * vy[i];
//            archivovx << std::fixed << vx[i];
//            archivovy << std::fixed << vy[i];
//            archivoY << std::fixed << y[i];
//            if (i < nparticulas - 1) {
//                archivovx << ",";
//                archivovy << ",";
//                archivoY << ",";
//            }
//            else {
//                archivovx << std::endl;
//                archivovy << std::endl;
//                archivoY << std::endl;
//            }
//        }
//        archivovx.close();
//        archivovy.close();
//        archivoY.close();
        // Se reabren en modo append:
        //archivovx.open(ruta + "\\velx.csv");
//        archivovy.open(ruta + "\\vely.csv");
//        archivoY.open(ruta + "\\posy.csv");
        for (int itiempo = 1; itiempo <= npasos_temporales; itiempo++) {
            std::cout << "Paso: " << itiempo << std::endl;
            mvtoparticulas(nparticulas, x, y, vx, vy, Ex, dt, dx, carga_masa, Bo);
            particula_cf(nparticulas, longitud_malla, x);
            for (int i = 0; i < npuntos_malla + 1; i++) rhoe[i] = 0;
            densidad(nparticulas, npuntos_malla, dx, q, x, rhoe, itiempo);
            for (int i = 0; i < npuntos_malla + 1; i++) rhot[i] = rhoe[i] + rhoi;
            campo(npuntos_malla, rhot, dx, Ex);

            //for (int i = 0; i < nparticulas; i++) {
//                archivovx << std::fixed << vx[i];
//                archivovy << std::fixed << vy[i];
//                archivoX << std::fixed << x[i];
//                archivoY << std::fixed << y[i];
//                if (i < nparticulas - 1) {
//                    archivovx << ",";
//                    archivovy << ",";
//                    archivoX << ",";
//                    archivoY << ",";
//                }
//                else {
//                    archivovx << std::endl;
//                    archivovy << std::endl;
//                    archivoX << std::endl;
//                    archivoY << std::endl;
//                }
//            }
//            for (int i = 0; i < npuntos_malla + 1; i++) {
//                archivoEx << std::fixed << Ex[i];
//                archivorhoe << std::fixed << rhoe[i];
//                if (i < npuntos_malla) {
//                    archivoEx << ",";
//                    archivorhoe << ",";
//                }
//                else {
//                    archivoEx << std::endl;
//                    archivorhoe << std::endl;
//                }
//            }
        }
        //archivovx.close();
//        archivovy.close();
//        archivoY.close();
    }
//    archivoX.close();
//    archivoEx.close();
//    archivorhoe.close();

    auto final = std::chrono::system_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::duration<double>>(final - inicio).count();
    printf("Tiempo total: %f", t);
    std::ofstream archivoInfo(ruta + "\\descripcion.txt");
    archivoInfo << " Descripcion de simulacion \"" + nombre + "\" \n";
    archivoInfo << " - Fenomeno:             ";
    if (vth[0] == 0)
        archivoInfo << "Plasma frio\n";
    else {
        if (n_especies == 1)
            archivoInfo << "Flujo normal (1 especie)\n";
        else {
            if (n_especies == 2) {
                if (c_n[0] == 0.5)
                    archivoInfo << "Two-Stream\n";
                else
                    archivoInfo << "Bump-On-Tail\n";
            }
            else
                archivoInfo << "Personalizado\n";
        }
    }
    archivoInfo << " - Numero de especies:   " << n_especies << "\n";
    archivoInfo << " - Coeficient(es) ci:    ";
    for (int i = 0; i < n_especies; i++)
    {
        archivoInfo << std::fixed << c_n[i];
        if (i < n_especies - 1)
            archivoInfo << ", ";
        else
            archivoInfo << "\n";
    }
    archivoInfo << " - Velocidad(es) vhi:    ";
    for (int i = 0; i < n_especies; i++)
    {
        archivoInfo << std::fixed << vh[i];
        if (i < n_especies - 1)
            archivoInfo << ", ";
        else
            archivoInfo << "\n";
    }
    archivoInfo << " - Ancho(s) v. vthi:     ";
    for (int i = 0; i < n_especies; i++)
    {
        archivoInfo << std::fixed << vth[i];
        if (i < n_especies - 1)
            archivoInfo << ", ";
        else
            archivoInfo << "\n";
    }
    archivoInfo << " ###################################################\n";
    archivoInfo << " - Particulas:           " << nparticulas << "\n";
    archivoInfo << " - Puntos de malla (SP): " << npuntos_malla << "\n";
    archivoInfo << " - Pasos temporales:     " << npasos_temporales << "\n";
    archivoInfo << " - Longitud de malla:    " << std::fixed << longitud_malla << "\n";
    archivoInfo << " - Campo B (Bo, dir z):  " << std::fixed << Bo << "\n";
    archivoInfo << " - Amp. Perturb. Pos.:   " << std::fixed << aP << "\n";
    archivoInfo << " - Amp. Perturb. Vel.:   " << std::fixed << vP << "\n";
    archivoInfo << " ###################################################\n";
    archivoInfo << " - Tiempo de simulacion (seg): " << std::fixed << t << " \n";
    archivoInfo.close();

    return 0;
}
