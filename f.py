#-------------------------------------------------------------------------------#
#__________________  FUNCTIONS_for_BTC_1H_predict     __________________________#
#-------------------------------------------------------------------------------#
#----------------------------------------------------author: Evstifeev A.I.-----#
#----------------------------------------------------email: Raz1nad@yandex.ru---#
#----------------------------------------------------last_edit: 16.11.2020------#
#!/usr/bin/env python
# coding: utf-8

# import 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pylab
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
import ccxt
import csv

# Функция проверки совпадения длины загруженных ДатаФреймов
def different_DF(data_1, data_2):
    if len(data_1) != len(data_2):
        print ("Кол-во строк не совпадает, нужна оптимизация, "
               "Длина первого ДФ:", len(data_1) ,", Второго:", len(data_2))
        fl_Start_cut_row = True    # Флаг запуска функции выреза строк Cut_row
    else:
        print ("Строки совпадают, оптимизация не требуется. Длина обоих ДФ:", len(data_1))
        fl_Start_cut_row = False
    return(fl_Start_cut_row)

# Функция выреза строк: Перебираем оба датафрейма, если где-то идёт несовпадение времени в строках с одинаковым индексом,
# вырезаем эти строки. (Допустим, в одном ДФ пропущено пару дней, тогда из другого ДФ тоже вырезаем эти дни)
def Cut_row(fl_Start_cut_row, data_1, data_2):
    if fl_Start_cut_row:
        while(True):
            len_for = np.minimum(len(data_1),len(data_2))
            for i in range(len_for):
                if data_1['time'][i] > data_2['time'][i]:
                    data_2.drop([i], inplace = True)
                    data_2.index = range(len(data_2))
                    break
                elif data_1['time'][i] < data_2['time'][i]:
                    data_1.drop([i], inplace = True)
                    data_1.index = range(len(data_1))
                    break
            else:
                break
        if len(data_1) == len(data_2):
            print ('Оптимизация длины исходных ДФ заверешена успешно, итоговая длина ДатаФрейма:', len(data_1))
        else:
            print ('Оптимизация длины исходных ДФ НЕ была успешно завершена. Длина первого:', len(data_1), 'Второго:', len(data_2))
            print ('Вырежем объекты из большего ДФ')
            if  len(data_1) > len(data_2):
                data_1 = data_1[:len(data_2)]
            else:
                data_2 = data_2[:len(data_1)]
            print (' Длина перфого ДФ:', len(data_1), 'Второго:',  len(data_2) )

# Функция формирования признаков. Выделяем целевой признак. Переводим оставшиеся признаки в процентное изменение
def X_Y_form(full_data, ncc1, ncc2):
    x = full_data.drop('time',1)
    x_real = pd.DataFrame(full_data)  #сохраняем датафрейм до перевода в процентные изменения
    yname = ncc1 + 'close'
    y = pd.DataFrame(full_data[yname], columns = [yname])
    for ncc in [ncc1,ncc2]:
        x[ncc+'high'] = x[ncc+'high'].pct_change()
        x[ncc+'low'] = x[ncc+'low'].pct_change()
        x[ncc+'close'] = x[ncc+'close'].pct_change()
        # Добавим отношение тени хая свечи к закрытию и тени лоя свечи к закрытию
        x = x.astype('float64')
        x[ncc + 'upper_shadow'] = (full_data[ncc + 'high'] - full_data[ncc + 'close']) / full_data[ncc + 'close']
        x[ncc + 'lower_shadow'] = (full_data[ncc + 'low'] - full_data[ncc + 'close']) / full_data[ncc + 'close']
    y = y.pct_change().fillna(0) # Переводим целевой признак в процентное изменение
    x = x.fillna(0)
    return (x_real, x, y, full_data, yname)

# Функция рассчёта кол-ва идущих подряд зеленых или красных свеч
def RedGreen(ncc1, ncc2, x):
    x = x.fillna(0)
    x_greens = []
    green = 0
    for i in range(len(x[ncc2+'close'])):
        if x[ncc2+'close'][i]<0:
            green = 0
            green -= 1            # Если свечи красные, считаем их кол-во в отриц.области
        else:
            green += 1            # Считаем кол-во зеленых свеч
        x_greens.append(green)
    x[ncc2+'Greens'] = x_greens
    x_greens.clear()
    for i in range(len(x[ncc1+'close'])):
        if x[ncc1 + 'close'][i] < 0:
            green = 0
            green -= 1  # Если свечи красные, считаем их кол-во в отриц.области
        else:
            green += 1  # Считаем кол-во зеленых свеч
        x_greens.append(green)
    x[ncc1+'Greens'] = x_greens
    x_greens.clear()
    return(x)

# Функция индекса относительной силы. Расcчитывается RSI.
def RSI(x_real, ncc, period_RSI):
    gain, loss, RSI = [],[],[]
    for i in range(len(x_real)):
        if i > period_RSI:
            # Если цена выросла, добавляем в gain величину роста
            if x_real[ncc + 'close'][i] > x_real[ncc + 'close'][i-1]:
                gain.append(x_real[ncc + 'close'][i] - x_real[ncc + 'close'][i-1])
                loss.append(0)
            # Если цена упала, добавляем в loss величину падения
            else:
                loss.append(-x_real[ncc + 'close'][i] + x_real[ncc + 'close'][i-1])
                gain.append(0)
        # Пока не прошли один период, добавляем средние значения
        else:
            gain.append(0.5)
            loss.append(0.5)
        # Считаем индекс относительной силы RSI:
        if np.sum(loss[i-period_RSI+1:i+1])!=0:
            RSI.append(1 - 1/( 1 + np.sum(gain[i - period_RSI + 1 : i + 1])/np.sum(loss[i-period_RSI + 1 : i + 1]) ) )
        # Если потери равны нулю на периоде
        else:
            RSI.append(1 - 1 / (1 + np.sum(gain[i - period_RSI + 1 : i + 1]) / 10**-8))
    x_real[ncc + 'RSI' + str(period_RSI)] = RSI   # Добавляем столбец DataFrame

# Функция EMA. Считаем экспонинциальное скользящее среднее согласно заданного периода.
def EMA(x_real, ncc, period_EMA):
    a = 2/(1+period_EMA)
    EMA = []
    for i in range(len(x_real)):
        Pt = x_real[ncc + 'close'][i]
        if i < 1:
            EMA.append(Pt)
        else:
            EMA.append(a*Pt+(1-a)*EMA[i-1])
    x_real[ncc+'EMA' + str(period_EMA)] = EMA

# Функция построения графика процентного колебания цены на выбранном таймфрейме
def pct_price_change(data, y, x, xname, yname, ncc1, ncc2):
    # Вызываем функцию для определения среднего роста и падения по заданному временному интревалу:
    def mean_pct_change(column):
        x_up=x_down=i_x_up=i_x_down = 0
        for i in range(len(column)):
            if column[i] > 0.0:
                x_up += column[i]
                i_x_up += 1
            else:
                x_down += column[i]
                i_x_down += 1
        mean_x_up = x_up/i_x_up
        mean_x_down = x_down/i_x_down
        return(mean_x_up, mean_x_down)

    # Передаем индексы, как время:
    x_date = data['time']
    y.index = x_date
    name_column = list(y.columns.values)
    # Строим первый график процентного колебания цены:
    y.columns = [ncc1 + '_pct_change']  # Переименуем столбец в лейбл для легенды
    ax = y.plot(figsize=(17, 5),color = 'b', grid = True, linewidth = 0.5, alpha = 0.7)
    y.columns = name_column              # Возвращаем название столбца
    # Устанавливаем границы по оси ординат для комфортного отображения:
    pylab.ylim([-np.max(x[yname])*0.75, np.max(x[yname])*0.75])
    # Строим второй график процентного колебания цены:
    x_plot = x[xname]
    x_plot.index = x_date
    x_plot.plot(figsize=(17, 5),color = 'r', grid = True, label=ncc2+'_pct_change',linewidth = 0.5, alpha = 0.7)
    # Строим линию средних колебаний цен:
    plt.axhline(y=mean_pct_change(x[yname])[0],linestyle='--',linewidth= 1.2, color ='green', alpha = 1.0,label=ncc1+'_average_change')
    plt.axhline(y=mean_pct_change(x[yname])[1],linestyle='--',linewidth= 1.2, color ='green', alpha = 1.0)
    plt.axhline(y=mean_pct_change(x[xname])[0],linestyle='--',linewidth= 1.2, color ='y', alpha = 1.0, label=ncc2+'_average_change')
    plt.axhline(y=mean_pct_change(x[xname])[1],linestyle='--',linewidth= 1.2, color ='y', alpha = 1.0)
    ax.set_xlabel('Дата', fontsize=12)
    pylab.legend(loc="upper left")
    pylab.xticks(rotation=0)
    pylab.title('Процентное изменение цены относительно предыдущей свечи')

    # Выводим на экран средние ценовые колебания:
    print ('Среднее отклонение за интервал ', str(ncc1), ': ',
           np.round((-mean_pct_change(x[yname])[1]+mean_pct_change(x[yname])[0])*100/2, 3),'%','\n',
       'Среднее отклонение за интервал ', str(ncc2), ': ',
           np.round((-mean_pct_change(x[xname])[1]+mean_pct_change(x[xname])[0])*100/2, 3),'%' , sep ='')

# Функция выделения класса целевого признака
# ВОЗВРАЩАЕТ МАССИВ ЦЕЛЕВОГО ПРИЗНАКА (ПОКУПКА, ПРОДАЖА)
# "1" - цена вырастет на закрытии сл.свечи, "0"-упадёт
def class_definition(y, x, yname):
    Y = np.array(range(len(y)))
    step = 1                                           # Шаг
    # Начинаем цикл по всем историческим свечкам
    for i in range(0,(len(y)-1), step):
            x_i = x[i:(i+1+step)]
            next_close = x_i[yname][(i+step)]          # Свеча след.дня
            if next_close > 0:                         # Если след.свеча вырастет, тогда
                y_i = 1                                # 1 клас
            else:                                      # Иначе
                y_i = 0                                # 0 класс(цена упадёт)
            # Формируем список целевых признаков:
            Y[i] = y_i  
    Y[-1:] = 0                                         # Определим последний элемент как класс 0
    return(Y)

# Функция определения корреляции признаков с целевым (ПРОЦЕНТ ЛИНЕЙНОЙ ЗАВИСИМОСТИ)
def correlation(Y, x):
    pylab.rcParams['figure.figsize'] = [14, 5]
    Y_cor = pd.DataFrame(Y[:len(x)], columns=['Next_BTC_close'])
    X_cor = x.join(Y_cor)
    Corrwith_data = pd.DataFrame.corrwith(X_cor, X_cor['Next_BTC_close'])
    Corrwith_data = Corrwith_data.to_frame()
    Corrwith_data.drop('Next_BTC_close', axis=0, inplace=True)

    # Отрисовываем график линейной зависимости к целевому признаку:
    Corrwith_data.plot(kind='bar', width=1, alpha=0.8, edgecolor="k", grid=True, title = ' Корреляция с целевым признаком', legend = False)
    plt.tick_params(labelsize=11)
    plt.xticks(rotation=15)

# Функция деления выборки на обучение и тест
def test_train_split(x, Y, test_size):
    # Перейдём от DF и list к numpy массивам
    X_np, Y_np = np.array(x), Y
    # Разделим нашу выборку на обучение и тест
    (X_train, X_test, 
     y_train, y_test) = train_test_split(X_np, Y_np, test_size=test_size, shuffle = False)
    # Проверим баланс классов:
    print ('В обучающей выборке дней c положительной динамики:', np.sum(y_train == 1), ', отрицательной:', np.sum(y_train == 0), sep='')
    print ('В тестовой выборке дней c положительной динамики:', np.sum(y_test==1), ', отрицательной:',np.sum(y_test==0), sep='')
    return (X_train, X_test, y_train, y_test, X_np, Y_np)

# Функция 1 подбора оптимальных параметров модели по кросс-валидации:
def Forest_params(min_trees, max_trees, step_trees, min_depth, max_depth, step_depth, learning_rate,
                  min_child_weight, X_train, y_train, reg_alpha, reg_lambda):
    import xgboost
    from numpy import asmatrix
    from sklearn import model_selection
    from matplotlib import pylab
    import matplotlib
    # ФУНКЦИЯ ОПРЕДЕЛЯЕТ НАИЛУЧШИЕ ПАРАМЕТРЫ ЛЕСА
    # Задаём диапазон количества деревьев:
    n_trees = [min_trees] + list(range(min_trees+step_trees , max_trees, step_trees ))                    
    scoring = []
    # Задаём диапазон макс.глубины дерева:
    max_depths = [min_depth] + list(range(min_depth+step_depth, max_depth, step_depth))
    for n_tree in n_trees:
        for depth in max_depths:
            estimator = xgboost.XGBClassifier(learning_rate=learning_rate, max_depth=depth, n_estimators=n_tree, min_child_weight=min_child_weight,
                                              reg_alpha=reg_alpha, reg_lambda=reg_lambda)
            score = model_selection.cross_val_score(estimator, X_train, y_train, 
                                                     scoring = 'precision', cv = 3)    
            scoring.append(score)
    scoring = asmatrix(scoring)
    first = int(0)
    last = int((len(scoring.mean(axis = 1))/len(max_depths)))
    step = int(last)
    for i in range(len(max_depths)):
        pylab.plot(n_trees, scoring.mean(axis = 1)[first:last], marker='.', label=max_depths[i])
        first = last           # начальное становится последним
        last += step           # увеличиваем конечное значение на значение шага
        fig = plt.gcf()
        fig.set_size_inches(10, 4, forward=True)
    pylab.grid(True)
    pylab.xlabel('n_trees')
    pylab.ylabel('score')
    pylab.title('Precision score')
    pylab.legend(loc='best', title = 'depth')
    # Определяем лучшее кол-во деревьев:
    scoring_max, n_trees_best, depth_best = 0, 0, 0
    for i in range(len(n_trees)):
        for d in range(len(max_depths)):
            if scoring.mean(axis = 1)[i+d*len(n_trees)] > scoring_max:
                scoring_max = scoring.mean(axis = 1)[i+d*len(n_trees)]
                n_trees_best = n_trees[i]
                depth_best = max_depths[d]
    print ('Лучшее кол-во дереверьв:', ' ',n_trees_best, '\nГлубина:', ' ',depth_best, sep = '')
    return(n_trees_best, depth_best)

# Функция 2 подбора оптимальных параметров модели по кросс-валидации
def Forest_params_LR(n_tree, depth, MIN_mcw,MAX_mcw,step_mcw,step_LR, min_LR, max_LR, X_train, y_train, reg_alpha, reg_lambda):
    import xgboost
    from numpy import asmatrix
    from sklearn import model_selection
    from matplotlib import pylab
    import matplotlib
    # ФУНКЦИЯ ОПРЕДЕЛЯЕТ НАИЛУЧШИЕ ПАРАМЕТРЫ СЛУЧАЙНОГО ЛЕСА
    # Задаём диапазон learning rate:
    min_LR,max_LR, step_LR =int(min_LR*100), int(max_LR*100), int(step_LR*100)
    l_rates = [min_LR] + list(range(min_LR+step_LR, max_LR, step_LR ))
    for i in range(len(l_rates)):
        l_rates[i] = l_rates[i]/100
    scoring = []
    # Задаём диапазон min_child_weight:
    min_child_weights =  [MIN_mcw] + list(range(MIN_mcw+step_mcw, MAX_mcw, step_mcw))
    for min_child_weight in min_child_weights:
        for l_rate in l_rates:
            estimator = xgboost.XGBClassifier(learning_rate=l_rate, min_child_weight=min_child_weight, max_depth=depth, n_estimators=n_tree,
                                              reg_alpha=reg_alpha, reg_lambda=reg_lambda)
            score = model_selection.cross_val_score(estimator, X_train, y_train,
                                                     scoring = 'precision', cv = 3)
            scoring.append(score)
    scoring = asmatrix(scoring)
    first = int(0)
    last = int((len(scoring.mean(axis = 1))/len(min_child_weights)))
    step = int(last)
    for i in range(len(min_child_weights)):
        pylab.plot(l_rates, scoring.mean(axis = 1)[first:last], marker='.', label = min_child_weights[i])
        first = last           # начальное становится последним
        last += step           # увеличиваем конечное значение на значение шага
    fig = plt.gcf()
    fig.set_size_inches(10, 4, forward=True)
    pylab.grid(True)
    pylab.xlabel('l_rates')
    pylab.ylabel('score')
    pylab.title('Precision score')
    pylab.legend(loc='best', title = 'min_child_weight')
    # Определяем лучшее кол-во деревьев:
    scoring_max, l_rate_best, min_child_weight_best = 0, 0, 0
    for i in range(len(l_rates)):
        for d in range(len(min_child_weights)):
            if scoring.mean(axis = 1)[i+d*len(l_rates)] > scoring_max:
                scoring_max = scoring.mean(axis = 1)[i+d*len(l_rates)]
                l_rates_best = l_rates[i]
                min_child_weight_best = min_child_weights[d]
    print ('Лучший уровень обучения:', ' ',l_rates_best, '\nmin_child_weight лучший:', ' ',min_child_weight_best, sep = '')
    return(l_rates_best, min_child_weight_best)

# Функция построения графиков депозита.
def rate_precision_plot(predict,percent,d_ind,y,y_test,cash, cash_start, fl_print, x_date, yname, fl_stats, n, predict_2, percent_2):
    cash_point, index_time_list = [], []
    number  = 0                                                           # счётчик неуспешных прогнозов
    counter = 0                                                           # счётчик успешных прогнозов
    for i in range(len(predict)-1):
        if percent_2 != None:                                             # Если две модели:
            if predict[i]>=percent and predict_2[i]>=percent_2:           # Если вероятность 1-го класса выше percent
                # Анализируем, какой верный ли прогноз
                if y_test[i]==1:                                          # Если в выборке 1 класс и в классификации
                    counter+=1                                            # Увеличиваем число успешных отнесений
                    number+=1
                else:
                    number+=1
                index_time_list.append(x_date[d_ind + i + 1])  # Записываем лист с индексами даты каждой точки
                cash = cash * y[yname][d_ind + i + 1] + cash
                cash_point.append(cash)
        else:
            if predict[i] >= percent:
                # Анализируем верный ли прогноз
                if y_test[i]==1:                                              # Если в выборке 1 класс и в классификации
                    counter+=1                                                # Увеличиваем число успешных отнесений
                    number+=1
                else:
                    number+=1
                index_time_list.append(x_date[d_ind + i + 1])  # Записываем лист с индексами даты каждой точки
                cash = cash * y[yname][d_ind + i + 1] + cash
                cash_point.append(cash)
    # Считаем процент точности по нашим условиям доверия:
    cash_pointDF = pd.DataFrame(cash_point)
    cash_pointDF.index = index_time_list
    if number != 0:
        precision = (counter/number)
    else:
        precision = 0
    if fl_stats:
        print ('Общее количество уверенных прогнозов:', number)
        print ('Успешных:', counter, 'Неуспешных:', number-counter)
        if percent_2 != None:
            print('\033[1m' + 'Общая точность согласно условиям доверия первой модели', np.round(percent * 100, 2), '%',
                  'и второй модели', np.round(percent_2 * 100, 2), '%:', '\n', np.round(precision * 100, 2), '%')
        else:
            print ('\033[1m' + 'Общая точность согласно условиям доверия в', np.round(percent*100, 2),'%:','\n', np.round(precision*100,2), '%')
        print ('Стартовый депозит:', cash_start, '$', 'Конечный депозит:', np.round(cash,0), '$')
    # Смотрим на депозит
    index_time_list.clear()
    cash_point.clear()
    if fl_print:
        if n:
            plt.rcParams['figure.figsize'] = [16, 5.5]
            plt.title('Статистика депозита')
            plt.plot(cash_pointDF, ".", label='Deposit '+n+', pct: '+str(np.round(percent*100,1)))
            plt.legend(loc='upper left')
        else:
            if percent_2 != None:
                cash_pointDF.plot(grid=True, marker="D", color='red', markersize=2, title='XGBoost+KNN deposit')
            else:
                cash_pointDF.plot(grid=True, label="Deposit best try", color = 'red').legend(loc=3)

        # Делим положение меток на оси х, с заданным интервалом:
        #xtick_location = cash_pointDF.index.tolist()[::30]
        #plt.xticks(ticks=xtick_location, rotation=45, fontsize=10, horizontalalignment='center', alpha=.9)
        plt.grid(axis='both', alpha=.9)
        plt.show
    return (cash, precision)


# Функция вывода депозита по линейной модели
def deposit_linear_model(predict,d_ind,y,y_test,cash, cash_start, fl_print, fl_print_trades, x_date, yname, fl_stats, data_plot, X_np, X_train):
    cash_point, index_time_list = [], []
    number  = 0                                                           # счётчик неуспешных прогнозов
    counter = 0                                                           # счётчик успешных прогнозов
    for i in range(len(predict)-1):
        if predict[i] == 1:                                               # Если определили первый класс

            # Анализируем верный ли прогноз
            if y_test[i]==1:                                              # Если в выборке 1 класс и в классификации
                counter+=1                                                # Увеличиваем число успешных отнесений
                number+=1
            else:
                number+=1
            # Считаем отношение цен закрытия двух свеч:
            if fl_print_trades:
                print ('Цена изменилась на:','\033[1m',+ 100*y[yname][d_ind+i+1],'%','\033[0m \n')
            index_time_list.append(x_date[d_ind+i+1])                     # Записываем лист с индексами даты каждой точки
            cash = cash*y[yname][d_ind+i+1]+cash
            cash_point.append(cash)
    cash_pointDF = pd.DataFrame(cash_point)
    cash_pointDF.index = index_time_list
    if fl_stats:
        print ('Успешных прогнозов:', counter, 'Неуспешных:', number-counter)
        print ('Стартовый депозит:', cash_start, '$', 'Конечный депозит:', np.round(cash,0), '$')
    # Смотрим на депозит
    index_time_list.clear()
    cash_point.clear()
    if fl_print:
        cash_pointDF.plot(grid=True, color='red', title = 'График депозита по линейной модели, $')
        plt.grid(axis='both', alpha=.9)
        plt.show
        (data_plot[yname][-(X_np.shape[0] - X_train.shape[0]):]).plot(secondary_y=True, grid=True,
                                                                      label="BTC_PRICE",
                                                                      title='Статистика депозита и ценовое движение').legend(loc=4)

# Функция подбора оптимального доверительного интервала:
def optimum_confidence_interval(data, start_pct, stop_pct, step_pct, d_ind, y, y_test, cash, cash_start, predict, yname):
    best_score = n = 0
    pylab.rcParams['lines.linewidth'] = 1
    y.index = data['time']
    start_pct, stop_pct,  step_pct =  int(start_pct*10), int(stop_pct*10),  int(step_pct*10) # подводим под целочисленные значения
    for percent in range(start_pct, stop_pct, step_pct):
        percent = percent/1000.0;                            # Переводим проценты в дробную десятичную форму записи
        n += 1                                               # чисто итераций
        score_i, acc_i = rate_precision_plot(predict, percent, d_ind, y, y_test, cash, cash_start, fl_print = True,
                                          x_date = data['time'], yname=yname, fl_stats=False, n=str(n), predict_2=None, percent_2=None)
        if best_score < score_i:
            best_score = score_i
            best_percent = percent
    return (best_percent)

# Функция построения графика обучающей выборки:
def plot_test_sample(data, X_np, X_train, yname):
    pylab.rcParams['figure.figsize'] = [15, 5]
    pylab.subplot (2, 1, 2).set_ylabel('$') 
    data_plot = (pd.DataFrame(data))
    data_plot.index = data['time']
    pylab.plot(data_plot[yname][-(X_np.shape[0]-X_train.shape[0]):])
    pylab.grid()
    return (data_plot)

# Функция отрисовки графика Цены, RSI, EMA:
def plot_EMA_RSI(x_real, period_RSI, period_EMA, ncc, length):
    if length > len(x_real):                 # Если ввели слишком большое значение
        length = len(x_real)
    data_plot = (pd.DataFrame(x_real))
    data_plot.index = x_real['time']

    fig, axes = plt.subplots(2, 1, figsize=(15, 5))
    plt.subplots_adjust(wspace=0, hspace=0)  # Убираем отступ между графиками subplots
    # График цены криптовалюты:
    data_plot[ncc+'close'][period_RSI:length+period_RSI].plot(ax=axes[0],linewidth='3', grid=True,color='green')
    data_plot[ncc+'EMA' + str(period_EMA)][period_EMA:length+ period_EMA].plot(ax=axes[0],linewidth='2',grid=True,color='yellow')
    axes[0].set_ylabel('price BTC, $')
    axes[0].legend(([ncc + 'close', 'EMA' + str(period_EMA)]))
    # График осцилятора RSI
    data_plot[ncc + 'RSI' + str(period_RSI)][period_RSI:length + period_RSI].plot(ax=axes[1], sharex = True,
                                                                                 color='red', grid=True, linewidth='2')
    axes[1].set_ylabel('RSI')
    axes[1].legend(['RSI'])
    axes[1].set_xlabel('Дата', fontsize = 12)
    pylab.xticks(rotation=0)

# Функция отрисовки графика цены на тестовой выборке совмещенный с графиком депозита при условной торговле по лучшей модели
def data_deposit_plot(predict, data_plot, d_ind,y, y_test, cash, cash_start, best_percent, data, yname, X_train, X_np):
    # ФУНКЦИЯ ВЫВОДИТ ГРАФИК ЦЕНЫ НА ТЕСТОВОЙ ВРЕМЕННОЙ ВЫБОРКЕ,
    # НАЛОЖЕННЫЙ НА ГРАФИК ИЗМЕНЕНИЯ ДЕПОЗИТА СООТВ.ТОРГОВОЙ СТРАТЕГИИ
    rate_precision_plot(predict, best_percent, d_ind, y, y_test, cash, cash_start,fl_print = True,
                     x_date = data['time'],yname=yname, fl_stats=False, n=None, predict_2=None, percent_2=None)
    (data_plot[yname][-(X_np.shape[0]-X_train.shape[0]):]).plot(secondary_y=True, grid = True,
                                                                label="BTC_PRICE", title = 'Статистика депозита и ценовое движение').legend(loc=4)

# Функция формирования Датафреймов
def data_form(data_1, data_2, ncc1, ncc2):
    data_1.rename(columns={'low': ncc1+'low', 'close': ncc1+'close',
                            'high':ncc1+'high', 'vol':ncc1+'vol', 'open': ncc1+'open'}, inplace=True)
    data_2.rename(columns={'low': ncc2+'low', 'close': ncc2+'close',
                            'high':ncc2+'high', 'vol':ncc2+'vol', 'open': ncc2+'open'}, inplace=True)
    data_2=data_2.drop(data_2.columns[[0]], axis='columns')  # Удаляем у альт.пары дату
    data = data_1.join(data_2)
    return (data)



#_________________________________________________________________________________________#
#--------------------------Candles download from exchanges--------------------------------#
#_________________________________________________________________________________________#

def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise  # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')

def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    earliest_timestamp = exchange.milliseconds()
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    all_ohlcv = []
    while True:
        fetch_since = earliest_timestamp - timedelta
        ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetch_since, limit)
        # if we have reached the beginning of history
        if ohlcv[0][0] >= earliest_timestamp:
            break
        earliest_timestamp = ohlcv[0][0]
        all_ohlcv = ohlcv + all_ohlcv
        # if we have reached the checkpoint
        if fetch_since < since:
            break
    return exchange.filter_by_since_limit(all_ohlcv, since, None, key=0)


def write_to_csv(filename, data):
    with open(filename, mode='w', newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(data)


def scrape_candles_to_csv(filename, exchange_id, max_retries, symbol, timeframe, since, limit):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,  # required by the Manual
    })
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = exchange.parse8601(since)
    # preload all markets from the exchange
    exchange.load_markets()
    # fetch all candles
    ohlcv = scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit)
    # save them to csv file
    write_to_csv(filename, ohlcv)
    print('Saved', len(ohlcv), 'candles from', exchange.iso8601(ohlcv[0][0]), 'to', exchange.iso8601(ohlcv[-1][0]), 'to', filename)




#_________________________________________________________________________________________#
#-----------------------------------Function archive-----------------------------------------#
#_________________________________________________________________________________________#

# Функция добавления 15-ти минутных свеч
##-----------------------------------------------------------
def add_15m_candles(First_Time, nc, vis_step, filename_15m_read, filename_save, data_1):
    import time
    import pandas as pd
    import numpy as np
    if First_Time == True:
        # Проверим на повторяющиеся свечи. Удалим, если загрузили несколько одинаковых свеч↓
        data_15m = pd.read_pickle(filename_15m_read)
        date_pre, hour_pre, summ, minute_pre = 0, 0, 0, 0
        del_rows = []
        print('Повтор на:')
        for i, item_2 in enumerate(data_15m['time']):
            if date_pre == item_2.date() and hour_pre == item_2.time().hour and minute_pre == item_2.time().minute:
                print(i, end='  ')
                summ += 1
                del_rows.append(i)
            date_pre = item_2.date()
            hour_pre = item_2.time().hour
            minute_pre = item_2.time().minute
        print('Элементах \nВсего повторов', summ)
        data_15m.drop(del_rows, inplace=True)  # удаляем лишние строки
        data_15m.index = np.arange(len(data_15m))

        # При добавлении признаков, будем использовать метод append, для ускорения процесса
        start_time = time.time()
        shift_1 = 0  # Инициализация сдвига для ускорения цикла
        shift_2 = 0  # Инициализация сдвига для ускорения цикла
        # Добавим пустые списки для append метода
        data_00m_close, data_15m_close, data_30m_close, data_45m_close = [], [], [], []
        data_00m_high, data_15m_high, data_30m_high, data_45m_high = [], [], [], []
        data_00m_low, data_15m_low, data_30m_low, data_45m_low = [], [], [], []
        data__00_time, data__15_time, data__30_time, data__45_time = [], [], [], []

        print('\nДобавление новых признаков:')
        for id, item_1 in enumerate(data_1['time']):
            if id % vis_step == 0 and id != 0:
                print('Обрабатываю', id, '-ю строку Data из', len(data_1),
                      "- %s seconds -" % (time.time() - start_time))
            for i, item_2 in enumerate(data_15m['time'][shift_2:]):
                if item_1.date() == item_2.date() and item_1.time().hour == item_2.time().hour:
                    if item_2.time().minute == 0:
                        data_00m_close.append(data_15m['close'][i + shift_2])
                        data_00m_high.append(data_15m['high'][i + shift_2])
                        data_00m_low.append(data_15m['low'][i + shift_2])
                        data__00_time.append(item_1)
                    elif item_2.time().minute == 15:
                        data_15m_close.append(data_15m['close'][i + shift_2])
                        data_15m_high.append(data_15m['high'][i + shift_2])
                        data_15m_low.append(data_15m['low'][i + shift_2])
                        data__15_time.append(item_1)
                    elif item_2.time().minute == 30:
                        data_30m_close.append(data_15m['close'][i + shift_2])
                        data_30m_high.append(data_15m['high'][i + shift_2])
                        data_30m_low.append(data_15m['low'][i + shift_2])
                        data__30_time.append(item_1)
                    elif item_2.time().minute == 45:
                        data_45m_close.append(data_15m['close'][i + shift_2])
                        data_45m_high.append(data_15m['high'][i + shift_2])
                        data_45m_low.append(data_15m['low'][i + shift_2])
                        data__45_time.append(item_1)
                # Выходим, если перепрыгнули нужную дату (для ускорения цикла)
                if item_2.date() > item_1.date():
                    shift_1 += 1  # делаем сдвиг для датасета с часовыми свечами (для ускорения цикла)
                    if shift_1 > 30:
                        shift_2 += 4  # делаем сдвиг для датасета с 15 минутными свечами (для ускорения цикла)
                    break
            # Формируем новые датафреймы
            data_new_00, data_new_15, data_new_30, data_new_45 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            # Добавили пустые столбцы
            data_new_00[nc + '00m_close'], data_new_00[nc + '00m_high'], data_new_00[nc + '00m_low'], data_new_00[
                'time'] = data_00m_close, data_00m_high, data_00m_low, data__00_time
            data_new_15[nc + '15m_close'], data_new_15[nc + '15m_high'], data_new_15[nc + '15m_low'], data_new_15[
                'time'] = data_15m_close, data_15m_high, data_15m_low, data__15_time
            data_new_30[nc + '30m_close'], data_new_30[nc + '30m_high'], data_new_30[nc + '30m_low'], data_new_30[
                'time'] = data_30m_close, data_30m_high, data_30m_low, data__30_time
            data_new_45[nc + '45m_close'], data_new_45[nc + '45m_high'], data_new_45[nc + '45m_low'], data_new_45[
                'time'] = data_45m_close, data_45m_high, data_45m_low, data__45_time

        print('data_new_00', len(data_new_00))
        print('data_new_15', len(data_new_15))
        print('data_new_30', len(data_new_30))
        print('data_new_45', len(data_new_45))
        # Cклеиваим датафреймы по ключу тайм
        for data in [data_new_00, data_new_15, data_new_30, data_new_45]:
            data_1 = pd.merge(left=data_1, right=data, left_on='time', right_on='time')

        # Сохраняем в файл
        data_1.to_pickle(filename_save)
        print ('Результат операции сохранён в файл: ', filename_save)
    else:
        data_1 = pd.read_pickle(filename_save)
        print('Произведено чтение из файла: ', filename_save)
    return (data_1)

# Функция перевода 15-ти минутных свеч в процентные изменения
def candles_15m_to_pct(First_Time, nc, filename_save, data_1):
    import time
    import pandas as pd
    import numpy as np
    import sys
    for q in (data_1['btc00m_close'], data_1['btc15m_close'], data_1['btc30m_close'], data_1['btc45m_close']):
        for i in q:
            if i == 0:
                sys.exit() # Останавливаем, если нашли ноль
    print ('Нулевых значений в признаках нет')

    # Получили датасет с вещественными признаками (в add_15m_candles) . Далее перейдём к процентным изменениям признака от объекта к объекту
    # Переходим к процентному изменению 15-ти минуток. Здесь считаем "вручную", т.к. вычисления специфичны
    if First_Time == True:
        # Создаём новые столбцы ДФ для новых признаков 15 минуток в процентном изменении относительно друг друга
        data_1[nc + '00m_close_pct'], data_1[nc + '15m_close_pct'], data_1[nc + '30m_close_pct'], data_1[
            nc + '45m_close_pct'] = 0., 0., 0., 0.
        data_1[nc + '00m_upper_shadow'], data_1[nc + '15m_upper_shadow'], data_1[nc + '30m_upper_shadow'], data_1[
            nc + '45m_upper_shadow'] = 0., 0., 0., 0.
        data_1[nc + '00m_lower_shadow'], data_1[nc + '15m_lower_shadow'], data_1[nc + '30m_lower_shadow'], data_1[
            nc + '45m_lower_shadow'] = 0., 0., 0., 0.
        # Создаём новые пустые списки для заполнения 15 минуток в процентном изменении
        data_00m_close_pct, data_15m_close_pct, data_30m_close_pct, data_45m_close_pct = [], [], [], []
        data_00m_upper_shadow, data_15m_upper_shadow, data_30m_upper_shadow, data_45m_upper_shadow = [], [], [], []
        data_00m_lower_shadow, data_15m_lower_shadow, data_30m_lower_shadow, data_45m_lower_shadow = [], [], [], []
        for i, item in enumerate(data_1['time']):
            if i > 0:
                data_00m_close_pct.append((data_1[nc + 'close'][i - 1] - data_1[nc + '00m_close'][i]) / data_1[nc + '00m_close'][i])
            else:
                data_00m_close_pct.append(0.0)

            data_15m_close_pct.append((data_1[nc + '15m_close'][i] - data_1[nc + '00m_close'][i]) / data_1[nc + '00m_close'][i])
            data_00m_upper_shadow.append((data_1[nc + '00m_high'][i] - data_1[nc + '00m_close'][i]) / data_1[nc + '00m_close'][i])
            data_00m_lower_shadow.append((data_1[nc + '00m_low'][i] - data_1[nc + '00m_close'][i]) / data_1[nc + '00m_close'][i])

            data_30m_close_pct.append((data_1[nc + '30m_close'][i] - data_1[nc + '15m_close'][i]) / data_1[nc + '15m_close'][i])
            data_15m_upper_shadow.append((data_1[nc + '15m_high'][i] - data_1[nc + '15m_close'][i]) / data_1[nc + '15m_close'][i])
            data_15m_lower_shadow.append((data_1[nc + '15m_low'][i] - data_1[nc + '15m_close'][i]) / data_1[nc + '15m_close'][i])

            data_45m_close_pct.append((data_1[nc + '45m_close'][i] - data_1[nc + '30m_close'][i]) / data_1[nc + '30m_close'][i])
            data_30m_upper_shadow.append((data_1[nc + '30m_high'][i] - data_1[nc + '30m_close'][i]) / data_1[nc + '30m_close'][i])
            data_30m_lower_shadow.append((data_1[nc + '30m_low'][i] - data_1[nc + '30m_close'][i]) / data_1[nc + '30m_close'][i])

            data_45m_upper_shadow.append((data_1[nc + '45m_high'][i] - data_1[nc + '45m_close'][i]) / data_1[nc + '45m_close'][i])
            data_45m_lower_shadow.append((data_1[nc + '45m_low'][i] - data_1[nc + '45m_close'][i]) / data_1[nc + '45m_close'][i])

            # Заполняем признаки в ДФ полученными сверху списками
        data_1[nc + '00m_close_pct'], data_1[nc + '15m_close_pct'], data_1[nc + '30m_close_pct'], data_1[
            nc + '45m_close_pct'] = data_00m_close_pct, data_15m_close_pct, data_30m_close_pct, data_45m_close_pct
        data_1[nc + '00m_upper_shadow'], data_1[nc + '15m_upper_shadow'], data_1[nc + '30m_upper_shadow'], data_1[
            nc + '45m_upper_shadow'] = data_00m_upper_shadow, data_15m_upper_shadow, data_30m_upper_shadow, data_45m_upper_shadow
        data_1[nc + '00m_lower_shadow'], data_1[nc + '15m_lower_shadow'], data_1[nc + '30m_lower_shadow'], data_1[
            nc + '45m_lower_shadow'] = data_00m_lower_shadow, data_15m_lower_shadow, data_30m_lower_shadow, data_45m_lower_shadow
        # Сохраняем в файл
        data_1.to_pickle(filename_save)
        print('Результат перевода 15-ти минутных свеч в процентные измненения сохранён в: ', filename_save)
    else:
        data_1 = pd.read_pickle(filename_save)
        print ('Данные прочитаны из: ', filename_save)
    # Удаляем из ДФ лишние признаки, до преобразования в процентные изменения
    for time_per in [nc + '00m_', nc + '15m_', nc + '30m_', nc + '45m_']:
        for sign in ['close', 'high', 'low']:
            data_1.drop(time_per + sign, axis=1, inplace=True)
    return (data_1)


# Функция : SMA. Считаем простое скользящее среднее согласно заданному периоду.
def SMA(x_real, x, ncc, period_SMA,fl_CloseSMA, fl_VolSMA):
    from numpy import mean
    SMA = []
    if fl_CloseSMA==True:
        for i in range(len(x_real)):
            if i < period_SMA:
                SMA.append(x_real[ncc + 'close'][i])
            else:
                SMA.append(mean(x_real[ncc + 'close'][i-period_SMA+1:i+1]))
        x_real[ncc+'SMA' + str(period_SMA)] = SMA
        x[ncc+'SMA_pct' + str(period_SMA)] = SMA
        x[ncc+'SMA_pct' + str(period_SMA)] = x[ncc+'SMA_pct' + str(period_SMA)].pct_change().fillna(0).astype('float64')
    SMA = []
    if fl_VolSMA==True:
        for i in range(len(x_real)):
            if i < period_SMA:
                SMA.append(x_real[ncc + 'vol'][i])
            else:
                SMA.append(mean(x_real[ncc + 'vol'][i-period_SMA+1:i+1]))
        x_real[ncc+'volSMA' + str(period_SMA)] = SMA
        x[ncc+'volSMA_pct' + str(period_SMA)] = SMA
        x[ncc+'volSMA_pct' + str(period_SMA)] = x[ncc+'volSMA_pct' + str(period_SMA)].pct_change().fillna(0).astype('float64')
    return(x_real, x)