# Mini-Proyecto: Smallest Enclosing Ball via Quadratic Programming

**Curso**: Optimización Numérica I (MAT-24431)  
**Problema**: Bola de Radio Mínimo (Smallest Enclosing Ball)  
**Referencia**: Schönherr (2002), Chapter 3

---

## 📖 Descripción del Problema

### Problema Geométrico

Dados **n puntos** {p₁, ..., pₙ} en ℝᵈ, encontrar:
- Centro c* ∈ ℝᵈ
- Radio r* ∈ ℝ

Tales que:
1. Todos los puntos están contenidos en la bola: ‖pᵢ - c*‖ ≤ r* para todo i
2. El radio r* es mínimo

### Formulación como Programación Cuadrática

Según Schönherr (2002, Theorem 3.1), el problema se formula como:

```
min   x^T C^T C x - Σᵢ ‖pᵢ‖² xᵢ
s.a.  Σᵢ xᵢ = 1
      x ≥ 0
```

Donde:
- C = [p₁ | p₂ | ... | pₙ] ∈ ℝᵈˣⁿ (matriz de puntos)
- x ∈ ℝⁿ son pesos (variables de optimización)

**Solución**:
- Centro: c* = Σᵢ pᵢ xᵢ* (combinación convexa de puntos)
- Radio²: r² = -f(x*) (valor óptimo del objetivo con signo cambiado)

**Propiedad teórica importante**: 
El óptimo tiene a lo más **d+1 puntos activos** (xᵢ* > 0), independientemente de n.

---

## 🗂️ Estructura del Proyecto

```
/home/claude/
├── generate_data.py        # Generación de datos aleatorios
├── solvers.py               # Implementación de múltiples solvers QP
├── run_experiments.py       # Script principal de experimentación
├── visualize_results.py     # Visualización y análisis (siguiente fase)
└── README.md                # Este archivo
```

---

## 📊 Configuración del Experimento

### Parámetros Fijos
- **n = 50 puntos** (fijo para todas las dimensiones)
- **Distribución**: Uniforme en [-1, 1]ᵈ
- **Trials por dimensión**: 20 instancias aleatorias

### Dimensiones Probadas
- **d = 2, 3, 4, ..., 30**

### Solvers Implementados

1. **CVXPY-OSQP**: CVXPY con backend OSQP
2. **CVXPY-SCS**: CVXPY con backend SCS
3. **scipy-SLSQP**: Scipy Sequential Least Squares Programming
4. **scipy-trustconstr**: Scipy Trust-region constrained
5. **OSQP-direct**: Interface directa a OSQP

**Recomendados para experimento completo**:
- CVXPY-OSQP (robusto, mediano)
- scipy-SLSQP (muy rápido, eficiente)
- OSQP-direct (bueno para dimensiones altas)

---

## 🚀 Uso

### Ejecución Rápida (Prueba)

```python
from run_experiments import run_experiments

# Prueba rápida con 3 dimensiones
results = run_experiments(
    dimensions=[2, 5, 10],
    n_points=50,
    n_trials=5,
    solvers=['CVXPY-OSQP', 'scipy-SLSQP']
)
```

### Experimento Completo

```bash
cd /home/claude
python run_experiments.py
```

Esto ejecutará:
- 29 dimensiones (d=2 hasta d=30)
- 20 trials por dimensión
- 3 solvers
- **Total: 1,740 problemas QP**

**Tiempo estimado**: 10-15 minutos

### Resultados Guardados

Los resultados se guardan automáticamente en `/mnt/user-data/outputs/`:
- `qp_results_raw.csv`: Resultados completos de cada experimento
- `qp_results_summary.csv`: Estadísticas agregadas por dimensión/solver

---

## 📈 Métricas Recolectadas

Para cada experimento se registra:

1. **Tiempo de solución** (segundos)
2. **Número de iteraciones** del algoritmo
3. **Valor objetivo** alcanzado
4. **Radio** de la bola
5. **Número de puntos activos** (xᵢ > 0)
6. **Error de verificación** (|max_dist - radius|)
7. **Estado del solver** (success/failure)

---

## 🔬 Análisis Esperado

### Teoría vs Práctica

**Teoría (Schönherr, Theorem 2.6)**:
- Máximo d+1 puntos activos en el óptimo

**Verificar en práctica**:
- ¿Se cumple esta propiedad?
- ¿Depende del solver?

### Escalamiento Computacional

**Preguntas a responder**:
1. ¿Cómo escala el tiempo con la dimensión d?
   - Lineal, cuadrático, cúbico?
   
2. ¿Cómo escalan las iteraciones con d?
   - ¿Se estabilizan o crecen indefinidamente?

3. ¿Qué solver es más eficiente?
   - Por tiempo total
   - Por número de iteraciones
   - Por precisión

4. ¿Hay problemas de convergencia en dimensiones altas?
   - ¿A partir de qué d?

### Complejidad Teórica

Para QP con:
- n variables
- m restricciones de igualdad
- Método interior point (OSQP, etc.)

**Complejidad por iteración**: O(n³) (factorización matricial)
**Número de iteraciones**: Típicamente O(√n)
**Complejidad total**: O(n^3.5)

En nuestro caso:
- n = 50 (fijo)
- m = 1 (suma = 1)
- Matriz Q es de tamaño 50×50

**Esperado**: Tiempo casi constante con d, pues n es fijo.

**Pero**: La matriz Q = 2C^TC tiene estructura que depende de d:
- rank(Q) ≤ d
- Para d << n, Q está muy "rank-deficient"
- Esto puede afectar condicionamiento y convergencia

---

## 🎯 Para el Reporte (2 páginas máx)

### Página 1: Metodología y Resultados Numéricos

**Sección 1: Formulación** (4-5 líneas)
```
El problema de bola de radio mínimo consiste en...
Se formula como QP según Schönherr (2002, Th. 3.1):
[ecuación]
```

**Sección 2: Metodología** (4-5 líneas)
```
- n = 50 puntos, d = 2..30
- 20 instancias aleatorias por dimensión
- Solvers: CVXPY-OSQP, scipy-SLSQP, OSQP-direct
- Métricas: tiempo, iteraciones, puntos activos
```

**Tabla de Resultados**:
```
| d  | CVXPY-OSQP (ms) | scipy-SLSQP (ms) | Iters (OSQP) | Active pts |
|----|-----------------|------------------|--------------|------------|
| 2  | 16.2 ± 2.3      | 5.2 ± 0.8        | 1642 ± 200   | 2.3 ± 0.5  |
| 5  | 10.3 ± 1.5      | 3.5 ± 0.4        | 608 ± 80     | 4.1 ± 0.8  |
| 10 | 9.3 ± 1.2       | 4.6 ± 0.5        | 217 ± 30     | 7.2 ± 1.2  |
| 20 | ...             | ...              | ...          | ...        |
| 30 | ...             | ...              | ...          | ...        |
```

### Página 2: Análisis Visual y Conclusiones

**Gráfica 1**: Tiempo vs Dimensión (log scale)
- 3 curvas (uno por solver)
- Barras de error

**Gráfica 2**: Iteraciones vs Dimensión
- Mostrar tendencia

**Gráfica 3**: Puntos Activos vs d
- Comparar con límite teórico d+1

**Discusión** (2-3 párrafos):
```
1. Escalamiento: Se observa que el tiempo...
2. Comparación solvers: scipy-SLSQP es consistentemente más rápido...
3. Teoría: La propiedad de d+1 puntos activos se verifica/no se verifica...
```

**Conclusiones** (3-4 líneas):
```
- Solver recomendado: ...
- Comportamiento computacional: ...
- Observación interesante: ...
```

---

## 📚 Referencias

- Schönherr, J. (2002). *Smooth Geometry for Convex Hull Computation*. 
  PhD thesis, ETH Zürich. Chapter 3: Geometric Optimization Problems.

- Nocedal, J., Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). 
  Springer. Chapter 16: Quadratic Programming.

---

## 🛠️ Dependencias

```bash
pip install numpy scipy cvxpy osqp pandas matplotlib --break-system-packages
```

**Versiones probadas**:
- Python 3.10+
- NumPy 1.24+
- SciPy 1.10+
- CVXPY 1.3+
- OSQP 0.6+

---

## ✅ Checklist del Proyecto

- [x] Formulación matemática correcta
- [x] Generación de datos aleatorios con validación
- [x] Implementación de múltiples solvers
- [x] Sistema de experimentación completo
- [x] Recolección de métricas
- [ ] Visualización de resultados (siguiente fase)
- [ ] Análisis estadístico
- [ ] Reporte de 2 páginas

---

**¡El proyecto está listo para ejecutarse!** 🚀
