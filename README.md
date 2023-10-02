## Формирование кадра LR
Предполагая, что каждый элемент выборки поврежден аддитивным шумом, мы можем представить модель наблюдения как
###  $`\ x_L^{(n)} = D  H_n W_n x_H + e_n `$
$D$  – матрица выборки  
$H_n$  – матрица размытия  
$W_n$  – матрица деформации  
$e_n$  – упорядоченный вектор нормально распределенного аддитивного шума  
![image](https://github.com/transitofk/Reconstruction/assets/108511287/4a608f07-76a2-4c26-ab28-fba21bb933aa)
## Реконтрукция
Реконструкция изображения высокого разрешения из одной картины низкого качества будет иметь вид
### $`\ x_H = W_n^T H_n^T D^T x_L `$ 
Но поскольку в роли основы выступает не одно, а некоторый набор изображений низкого качества реализацию данного алгоритма можно осуществить с помощью использования метода градиентного спуска
### $`\ x_H [m+1] = x_H [m] + \alpha \displaystyle\sum_{n=1}^{K} W_n^T H_n^T D^T (x_L^{(n)} - D  H_n W_n x_H [m]) - \beta x_H [m] \bigotimes ker `$ 
$W_n^T$  – матрица, выполняющая обратное аффинное преобразование  
$H_n^T$  – матрица размытия с перевернутым ядром  
$D^T$  – матрица обратного прореживания, реализуемая через заполнения нулям неизвестных пикселей  
$\alpha$  – параметр сходимости  
$\beta$  – параметр, отвечающий за гладкость изображения  
$\bigotimes$  – операция свертки  
$ker$  – ядро двумерного дискретного лапласиана  

![image](https://github.com/transitofk/Reconstruction/assets/108511287/3a3357c2-935e-400a-b358-1b09a0819dc3)