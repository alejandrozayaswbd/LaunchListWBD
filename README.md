# Pipeline Excel - Limpieza y Actualización de Columnas

Este proyecto es un **pipeline en Python** que automatiza la limpieza y actualización de datos en un archivo de Excel.  
El objetivo principal es **actualizar las columnas del pivot con las nuevas del bridge**, conservar los valores válidos, y exportar una versión limpia de la base.

---

## Características
- Unifica columnas duplicadas usando **diccionarios de mapeo**.
- Conserva siempre el valor válido (`_y` sobre `_x`, si existe).
- Elimina columnas sobrantes (`*_y`).
- Incluye funciones auxiliares para procesar listas de prioridades y cadenas de regiones.
- Exporta un archivo Excel con la fecha del día en el nombre.

---

##  Estructura del Proyecto
## 📦 proyecto-pipeline
*  📜 Launch_List_Pivot_v2.py # Script principal
*  📜 Los input serán el archivo *bridge* en formato **xlsx** y el archivo *pivot* a actalizar.
*  📜 updated_main.csv # Archivo de salida generado
*  📜 README.md # Este archivo
