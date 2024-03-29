# Эксперимент

### Условия
Эксперимент проводился на компьютере с ОС ``Windows 10`` и процессором ``11th Gen Intel(R) Core(TM) i5-11400h @ 2.70Hz``


## Краевая задача 

Краевая задача была выбранна указанная в книге

При $`eps = 0.01`$

![](func.png)

## Результаты

* Паралельные версии алгоритмов 11.3 и 11.6 работают быстрее, чем последовательная версия.
* Если же сравнивать паралельные версии между собой, то алгоритм 11.3 окажется быстрее 11.6 при одинаковом числе потоков.
* Ускорение $`a = t(11.1)/t(11.6)`$ увеличивается с увеличением числа потоков. 
Однако эффективность $`a/(число потоков)`$ не всегда растёт пропорционально числу потоков.
В некоторых случаях увеличение числа потоков может привести к уменьшению эффективности, предполагаю что из-за накладных расходов на 
синхронизацию.
* Использование оптимизаций компилятора (03) значительно ускоряет работу.
