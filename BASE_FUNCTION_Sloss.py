#!/usr/bin/env python
# coding: utf-8

def name_coins(name_1, name_2, symbols_after_logo):
    from pandas import read_csv, to_datetime
    ncc1 = name_1[(-symbols_after_logo-3)]+name_1[(-symbols_after_logo-2)]+name_1[(-symbols_after_logo-1)]
    ncc2 = name_2[(-symbols_after_logo-3)]+name_2[(-symbols_after_logo-2)]+name_2[(-symbols_after_logo-1)]
    data_1 = read_csv(name_1)
    data_2 = read_csv(name_2)
    data_1['time'] = to_datetime(data_1['time'], format='%Y.%m.%d')
    data_2['time'] = to_datetime(data_2['time'], format='%Y.%m.%d')
    return(ncc1, ncc2, data_1, data_2)

# проверяем совпадают ли размерности загруженных ДатаФреймов
def different_DF(data_1, data_2):
    if len(data_1) != len(data_2):
        print ("Строки не совпадают, нужна подгонка, Длина первого ДФ:", len(data_1) ,"Второго:", len(data_2))
        fl_Start_cut_row = True
    else:
        print ("Строки совпадают, подгонка не нужна. Длина ДФ:", len(data_1))
        fl_Start_cut_row = False
    return(fl_Start_cut_row)

# Функция : Перебираем циклы по всей длинне наших массивов. Если где-то идёт несовпадение времени,
# вырезаем "лишние" строки. (допустим, в одном ДФ пропущено пару дней, тогда из другого ДФ тоже вырезаем эти дни)
def Cut_row(fl_Start_cut_row, data_1, data_2):
    from numpy import minimum
    if fl_Start_cut_row:
        while(True):
            len_for = minimum(len(data_1),len(data_2))
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
            print ('В итоге длина первого ДФ:', len(data_1), 'Второго:',  len(data_2) )
                
# Добавляем кол-во идущих подряд зеленых или красных свеч
def EMA_RedGreen(x_real, ncc1, ncc2, x):   
    # Расчитаем разницу между EMA10 и EMA18
    for i in range(len(x_real)):
        x[ncc1+'diffEMA'] = (x_real[ncc1+'MA']-x_real[ncc1+'MA.1'])/x_real[ncc1+'MA.1']
    # Добавим кол-во свечей подряд зеленых или красных
    x = x.fillna(0)
    x_greens, x_reds = [],[]
    green,red = 0, 0
    for i in range(len(x[ncc2+'close'])):
        if x[ncc2+'close'][i]<0:
            red += 1
            green = 0
        else:
            green += 1
            red = 0
        x_greens.append(green)
        x_reds.append(red)
    x[ncc2+'Greens'] = x_greens
    x[ncc2+'Reds'] = x_reds
    x_greens.clear()
    x_reds.clear()
    for i in range(len(x[ncc1+'close'])):
        if x[ncc1+'close'][i]<0:
            red += 1
            green = 0
        else:
            green += 1
            red = 0
        x_greens.append(green)
        x_reds.append(red)
    x[ncc1+'Greens'] = x_greens
    x[ncc1+'Reds'] = x_reds
    x_greens.clear()
    x_reds.clear()
    return(x)

# Визуализируем ценовые колебания валют
def pch_price_change_2CC(time_start, time_stop, data, y, x, xname, yname, ncc1, ncc2):
    from pandas import Timestamp, to_datetime
    from numpy import asarray, linspace, round
    from matplotlib import pylab
    # Определим переменные старта и конца временного интервала: 
    l  = Timestamp(time_start).value
    ll = Timestamp(time_stop).value
    # Массив времени:
    u = asarray(to_datetime(linspace(l,ll,10)))                    
    # Вызываем вложенную функцию для определения среднего роста и падения по заданному временному интревалу: 
    def mean_pct_change(column):
        mean_x_up, mean_x_down, x_up, x_down, i_x_up, i_x_down = 0, 0, 0, 0, 0, 0
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
    
    pylab.rcParams['figure.figsize'] = [17, 3]
    # ДФ даты:
    x_date = data['time']
    # Передаем индексы, как время:
    y.index = x_date  
    # Определяем ДФ, которые пойдут на печать
    y.plot(figsize=(17, 5),color = 'b', grid = True, legend = False, linewidth=1)
    x_plot = x[xname]
    x_plot.index = x_date
    x_plot.plot(figsize=(17, 5),color = 'r', grid = True,legend = False, linewidth=1)
    pylab.title('Процентное изменение цены относительно предыдушего дня '+ncc1+' (blue), '+ ncc2+' (red)')
    pylab.plot(u, linspace(mean_pct_change(x[xname])[0], mean_pct_change(x[xname])[0], 10), color = 'green',linewidth=1.5)
    pylab.plot(u, linspace(mean_pct_change(x[xname])[1], mean_pct_change(x[xname])[1], 10), color = 'green',linewidth=1.5)
    pylab.plot(u, linspace(mean_pct_change(x[yname])[0], mean_pct_change(x[yname])[0], 10), color = 'yellow',linewidth=1.5)
    pylab.plot(u, linspace(mean_pct_change(x[yname])[1], mean_pct_change(x[yname])[1], 10), color = 'yellow',linewidth=1.5)
    # Выводим на экран средние ценовые колебания:
    print ('Среднее часовое колебание цены предсказываемой валюты:',
           round((mean_pct_change(x[yname])[0]*100 - mean_pct_change(x[yname])[1]*100)/2, 3),'%','\n',
           'Среднее часовое колебание цены альтернативной валюты:',
           round((mean_pct_change(x[xname])[0]*100 - mean_pct_change(x[xname])[1]*100)/2, 3),'%' , sep ='')

# Определяем класс для формирования целевого признака
def class_definition(y, x, yname):
    from numpy import array
    # ФУНКЦИЯ ВОЗВРАЩАЕТ МАССИВ ЦЕЛЕВОГО ПРИЗНАКА (ПОКУПКА, ПРОДАЖА)
    Y = array(range(len(y)))
    step = 1
    # Начинаем цикл по всем историческим свечкам
    for i in range(0,(len(y)-1), step):
            x_i = x[i:(i+1+step)]                      # добавляем массив длинной WINDOW всех признаков
            next_close = x_i[yname][(i+step)]          # свеча след.дня
            if next_close > 0:                         # если след.свеча вырастет, тогда
                y_i = 1                                # 1 клас
            else:                                      # иначе 
                y_i = 0                                # 0 класс(цена упадёт)
            # Формируем список целевых признаков
            Y[i] = y_i  
    Y[-1:] = 0                                         # определим последний класс как 0
    return(Y)

# Смотрим на корреляцию к целевому признаку (изменение цены на след свече)
def correlation(Y, x):
    from pandas import DataFrame
    # ФУНКЦИЯ ПОКАЗЫВАЕТ КОРРЕЛЯЦИЮ ПРИЗНАКОВ К ЦЕЛЕВОМУ (ПРОЦЕНТ ЛИНЕЙНОЙ ЗАВИСИМОСТИ)
    Y_cor = DataFrame(Y[:len(x)],columns = ['Y(yname)'])
    X_cor = x.join(Y_cor)
    Corrwith_data = DataFrame.corrwith(X_cor, X_cor['Y(yname)'])
    print ('Корреляция признаков с целевым \n',+ Corrwith_data , sep='')

# Функция деления выборки на обучение и тест
def test_train_split(x, Y, test_size):
    from numpy import array, sum
    from sklearn.model_selection import train_test_split
    # Перейдём от DF и list к NP массивам
    X_np, Y_np = array(x), Y
    # Разделим нашу выборку на обучение и тест
    (X_train, X_test, 
     y_train, y_test) = train_test_split(X_np, Y_np, test_size=test_size, shuffle = False) 
    print ('В тестовой выборке дней c положительной динамики:', sum(y_test==1), ', отрицательной:',sum(y_test==0), sep='')
    return (X_train, X_test, y_train, y_test, X_np, Y_np)

# Функция определения наилучших параметров модели
def Forest_params(min_trees, max_trees, step_trees, min_depth, max_depth, learning_rate, min_child_weight, X_train, y_train):
    import xgboost
    from numpy import asmatrix
    from sklearn import model_selection
    from matplotlib import pylab
    # ФУНКЦИЯ ОПРЕДЕЛЯЕТ НАИЛУЧШИЕ ПАРАМЕТРЫ СЛУЧАЙНОГО ЛЕСА
    # Задаём диапазон количества деревьев:
    n_trees = [min_trees] + list(range(min_trees+step_trees , max_trees, step_trees ))                    
    scoring = []
    # Задаём диапазон макс.глубины дерева:
    max_depths = [min_depth] + list(range(3, max_depth, 1))                  
    for n_tree in n_trees:
        for depth in max_depths:
            estimator = xgboost.XGBClassifier(learning_rate=learning_rate, max_depth=depth, n_estimators=n_tree, min_child_weight=min_child_weight)
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
    pylab.grid(True)
    pylab.xlabel('n_trees')
    pylab.ylabel('score')
    pylab.title('Precision score')
    pylab.legend(loc='lower right')
    # Определяем лучшее кол-во деревьев:
    scoring_max, n_trees_best, depth_best = 0, 0, 0
    for i in range(len(n_trees)):
        for d in range(len(max_depths)):
            if scoring.mean(axis = 1)[i+d*len(n_trees)] > scoring_max:
                scoring_max = scoring.mean(axis = 1)[i+d*len(n_trees)]
                n_trees_best = n_trees[i]
                deph_best = max_depths[d]
    print ('Лучшее кол-во дереверьв:', ' ',n_trees_best, '\nГлубина:', ' ',deph_best, sep = '')
    return(n_trees_best, deph_best)

# Строим графики депозита с наилучшими параметрами
def predict_accuracy(predict,percent,d_ind,y,y_test,cash, cash_start, fl_print, fl_print_trades, x_date, yname, fl_stats, n):
    from matplotlib import pylab
    from pandas import DataFrame
    from numpy import round
    cash_point, index_time_list = [], []
    number  = 0                                                           # счётчик неуспешных прогнозов
    counter = 0                                                           # счётчик успешных прогнозов
    for i in range(len(predict)-1):
        if predict[i]>=percent:                                           # Если вероятность 1-го класса выше percent
            if fl_print_trades:
                if y_test[i] == 1:                                        # Определяем верный ли прогноз
                    plus_min = '+'
                else:
                    plus_min = '-'
                print('Строка',i+d_ind, 'Вероятность 1-го класса:',       # Отображаем эти элементы
                      '\033[1m', + predict[i], plus_min, '\033[0m')                     
                print ('Дата:', '\033[1m', 
                       (x_date[d_ind+i]), '\033[0m')
                print ('Закрытие предыдущей и анализируемой свечи:', '\033[1m',+ 
                       x_real[yname][d_ind+i], x_real[yname][d_ind+i+1],  '\033[0m')
            # Анализируем, какой класс спрогнозирован, какой реально 
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
    # Считаем процент точности по нашим условиям доверия:
    cash_pointDF = DataFrame(cash_point)
    cash_pointDF.index = index_time_list
    if number != 0:
        accuracy = (counter/number)
    else:
        accuracy = 0
    if fl_stats:
        print ('Общее количество уверенных прогнозов:', number)
        print ('Успешных:', counter, 'Неуспешных:', number-counter)
        print ('\033[1m' + 'Общая точность согласно условиям доверия в', round(percent*100, 2),'%:','\n', round(accuracy*100,2), '%')
        print ('Стартовый депозит:', cash_start, '$', 'Конечный депозит:', round(cash,0), '$')
    # Смотрим на депозит
    index_time_list.clear()
    cash_point.clear()
    if fl_print:
        pylab.rcParams['figure.figsize'] = [16, 5.5]
        pylab.title('Статистика депозита')
        if n:
            pylab.plot(cash_pointDF, ".", label='Deposit '+n)
            pylab.legend(loc='upper left')
        else:
            pylab.plot(cash_pointDF, ".", label='Deposit best try', color = 'green')
            pylab.legend(loc='upper left')
        pylab.grid()
    return (cash, accuracy)


# Подбираем оптимальный доверительный интервал

def optimum_confidence_interval(data, start_pct, stop_pct, step_pct, d_ind, y, y_test, cash, cash_start, predict, yname):
    from matplotlib import pylab
    best_score, score_i, acc_i, n = 0, 0, 0, 0
    pylab.rcParams['lines.linewidth'] = 1
    y.index = data['time']
    for percent in range(start_pct, stop_pct, step_pct):
        percent = percent/1000.0;
        n += 1                                               # чисто итераций
        score_i, acc_i = predict_accuracy(predict, percent, d_ind, y, y_test, cash, cash_start, fl_print = True,
                                          fl_print_trades = False, x_date = data['time'], yname=yname, fl_stats=False, n=str(n))
        if best_score < score_i:
            best_score = score_i
            best_percent = percent
    pylab.rcParams['lines.linewidth'] = 4
    predict_accuracy(predict, best_percent, d_ind, y, y_test, cash, cash_start, fl_print = True, fl_print_trades = False, x_date = data['time'], yname=yname, fl_stats=True, n=None)
    return (best_percent)


# Посмотрим на нашу тестовую выборку
def plot_test_sample(data, X_np, X_train, yname,Fl_print):
    from matplotlib import pylab
    from pandas import DataFrame
    data_plot = (DataFrame(data))
    data_plot.index = data['time']
    if Fl_print:
        pylab.rcParams['figure.figsize'] = [15, 5]
        pylab.subplot(2, 1, 2).set_ylabel('$')
        pylab.plot(data_plot[yname][-(X_np.shape[0]-X_train.shape[0]):])
    return (data_plot)


def data_deposit_plot(predict, data_plot, d_ind,y, y_test, cash, cash_start, best_percent, data, yname, X_train, X_np):
    # ФУНКЦИЯ ВЫВОДИТ ГРАФИК ЦЕНЫ В ТЕСТОВОЙ ВРЕМЕННОЙ ВЫБОРКЕ, 
    # НАЛОЖЕННЫЙ НА ГРАФИК ИЗМЕНЕНИЯ ДЕПОЗИТА СООТВ.ТОРГОВОЙ СТРАТЕГИИ
    predict_accuracy(predict, best_percent, d_ind, y, y_test, cash, cash_start,fl_print = True, fl_print_trades = False, x_date = data['time'],yname=yname, fl_stats=False, n=None)
    (data_plot[yname][-(X_np.shape[0]-X_train.shape[0]):]).plot(secondary_y=True, grid = True, label="BTC_PRICE").legend(bbox_to_anchor=(1, 0))

# Формируем ДФ
def data_form(data_1, data_2, ncc1, ncc2):
    from pandas import DataFrame
    data_1.rename(columns={'low': ncc1+'low', 'close': ncc1+'close', 'RSI': ncc1+'RSI', 
                            'high':ncc1+'high', 'MA':ncc1+'MA', 'MA.1':ncc1+'MA.1'}, inplace=True)
    data_2.rename(columns={'low': ncc2+'low', 'close': ncc2+'close', 'RSI': ncc2+'RSI', 
                            'high':ncc2+'high', 'MA':ncc2+'MA', 'MA.1':ncc2+'MA.1'}, inplace=True)
    data_1=data_1.drop(data_1.columns[[1]], axis='columns')
    data_2=data_2.drop(data_2.columns[[0, 1]], axis='columns')
    data = data_1.join(data_2)
    return (data)


def data_changes(full_data, ncc1, ncc2):
    from pandas import DataFrame
    x = full_data.drop('time',1)
    x_real = DataFrame(full_data)
    yname = ncc1 + 'close'
    xname = ncc2 + 'close'
    y = DataFrame(full_data[yname], columns = [yname])
    x = x.pct_change().fillna(0)
    x = x.astype('float64')
    y = y.pct_change().fillna(0)     
    # Добавим отношение тени хая свечи к закрытию и тени лоуа свечи к закрытию
    x[ncc1+'high_close'] = (x_real[ncc1+'high'] - x_real[ncc1+'close']) / x_real[ncc1+'close']
    x[ncc1+'low_close'] = (x_real[ncc1+'low'] - x_real[ncc1+'close']) / x_real[ncc1+'close']
    x[ncc2+'high_close'] = (x_real[ncc2+'high'] - x_real[ncc2+'close']) / x_real[ncc2+'close']
    x[ncc2+'low_close'] = (x_real[ncc2+'low'] - x_real[ncc2+'close']) / x_real[ncc2+'close']
    # Добавим реальное значение RSI
    x[ncc1+'RSI_val'] = x_real[ncc1+'RSI']
    x[ncc2+'RSI_val'] = x_real[ncc2+'RSI']
    x = x.fillna(0)
    return (x, y, full_data, yname, xname, x_real)    

# Вывод графика депозита, основываясь на 2х предикторах
def summary_prediction(predict_1, predict_2, percent_1, percent_2,
                       d_ind, y, y_test,cash, cash_start, fl_print, fl_print_trades, x_date, yname):
    from matplotlib import pylab
    from pandas import DataFrame
    from numpy import round
    cash_point, index_time_list = [], []
    number  = 0                                                           # счётчик неуспешных прогнозов
    counter = 0                                                           # счётчик успешных прогнозов
    for i in range(len(predict_1)-1):
        if predict_1[i]>=percent_1 and predict_2[i]>=percent_2:           # Если вероятность 1-го класса выше percent
            if fl_print_trades:
                if y_test[i] == 1:                                        # Определяем верный ли прогноз
                    plus_min = '+'
                else:
                    plus_min = '-'
                print('Строка',i+d_ind, 'Вероятность 1-го класса:',       # Отображаем эти элементы
                      '\033[1m', + predict_1[i], 'и', predict_2[i], plus_min, '\033[0m')                     
                print ('Дата:', '\033[1m', 
                       (x_date[d_ind+i]), '\033[0m')
                print ('Закрытие предыдущей и анализируемой свечи:', '\033[1m',+ 
                       x_real[yname][d_ind+i], x_real[yname][d_ind+i+1],  '\033[0m')
            # Анализируем, какой класс спрогнозирован, какой реально 
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
    # Считаем процент точности по нашим условиям доверия:
    cash_pointDF = DataFrame(cash_point)
    cash_pointDF.index = index_time_list
    if number != 0:
        accuracy = (counter/number)
    else:
        accuracy = 0
    if fl_print:
        print ('Общее количество уверенных прогнозов:', number)
        print ('Успешных:', counter, 'Неуспешных:', number-counter)
        print ('\033[1m' + 'Общая точность согласно условиям доверия в', round(percent_1*100, 2),'%',
               'и',round(percent_2*100, 2), '%:','\n', round(accuracy*100,2), '%') 
        print ('Стартовый депозит:', cash_start, '$', 'Конечный депозит:', round(cash,0), '$')
    # Смотрим на депозит
    pylab.rcParams['figure.figsize'] = [16, 5.5]
    pylab.title('Статистика депозита')
    index_time_list.clear()
    cash_point.clear()
    pylab.plot(cash_pointDF, "-r", marker="D", markersize=2, label = 'Deposit both predictors')
    pylab.legend(loc='upper left')
    pylab.grid()
    return (cash, accuracy)

# Вывод графика депозита, основываясь на 2х предикторах с учётом стоп-лосса
def summary_prediction_SL(predict_1, predict_2, percent_1, percent_2,
                       d_ind, y, y_test,cash, cash_start, fl_print, fl_print_trades, x_date, ncc1, percent_sl, x_real):
    from matplotlib import pylab
    from pandas import DataFrame
    from numpy import round
    cash_point, index_time_list, SL_point = [], [], []
    num_sl = 0                                                            # счётчик закрытия сделок по Stop Loss
    number  = 0                                                           # счётчик неуспешных прогнозов
    counter = 0                                                           # счётчик успешных прогнозов
    for i in range(len(predict_1)-1):
        if predict_1[i]>=percent_1 and predict_2[i]>=percent_2:           # Если вероятность 1-го класса выше percent
            if fl_print_trades:
                if y_test[i] == 1:                                        # Определяем верный ли прогноз
                    plus_min = '+'
                else:
                    plus_min = '-'
                print('Строка',i+d_ind, 'Вероятность 1-го класса:',       # Отображаем эти элементы
                      '\033[1m', + predict_1[i], 'и', predict_2[i], plus_min, '\033[0m')
                print ('Дата:', '\033[1m',
                       (x_date[d_ind+i]), '\033[0m')
                print ('Закрытие предыдущей и анализируемой свечи:', '\033[1m',+
                       x_real[ncc1+'close'][d_ind+i], x_real[ncc1+'close'][d_ind+i+1],  '\033[0m')
            # Анализируем, какой класс спрогнозирован, какой реально
            if y_test[i]==1:                                              # Если в выборке 1 класс и в классификации
                counter+=1                                                # Увеличиваем число успешных отнесений
                number+=1
            else:
                number+=1
            # Считаем отношение цен закрытия двух свеч:
            if fl_print_trades:
                print ('Цена изменилась на:','\033[1m',+ 100*y[ncc1+'close'][d_ind+i+1],'%','\033[0m \n')
            index_time_list.append(x_date[d_ind+i+1])                     # Записываем лист с индексами даты каждой точки
            # Смотрим тейк профит и стоп лосс:

            # Если лоу цена свечи ниже, чем цена её открытия на больший процент, чем стоп лосс
            # закрываем сделку по SL
            if ((x_real[ncc1 + 'low'][d_ind + i + 1] - x_real[ncc1 + 'close'][d_ind + i]) / x_real[ncc1 + 'close'][
                d_ind + i]) <= percent_sl:
                cash = cash * percent_sl + cash
                num_sl += 1
                SL_point.append(cash)
            # иначе считаем кэш по закрытию текущей свечи
            else:
                cash = cash * y[ncc1 + 'close'][d_ind + i + 1] + cash
                SL_point.append(None)
            # Добавляем в список цен
            cash_point.append(cash)

    # Считаем процент точности по нашим условиям доверия:
    SL_point_DF = DataFrame(SL_point)
    SL_point_DF.index = index_time_list

    cash_pointDF = DataFrame(cash_point)
    cash_pointDF.index = index_time_list
    if number != 0:
        accuracy = (counter/number)
    else:
        accuracy = 0
    if fl_print:
        print ('Общее количество уверенных прогнозов:', number)
        print ('Успешных:', counter, 'Неуспешных:', number-counter)
        print('Вышли по SL:', num_sl, ' раз(а)')
        print ('\033[1m' + 'Общая точность согласно условиям доверия в', round(percent_1*100, 2),'%',
               'и',round(percent_2*100, 2), '%:','\n', round(accuracy*100,2), '%')
        print ('Стартовый депозит:', cash_start, '$', 'Конечный депозит:', round(cash,0), '$')
    # Смотрим на депозит
        pylab.rcParams['figure.figsize'] = [16, 5.5]
        pylab.title('Статистика депозита')
        pylab.plot(cash_pointDF, "-r", marker="D", markersize=2, label = 'Статистика депозита')
        pylab.plot(SL_point_DF, "-g", marker="x", markersize=20, label = 'Выход по SL')
        pylab.legend(loc='upper left')
        #pylab.ylim(bottom = 800)  # отображаем график с 800, верхний предел - автоматически
        pylab.grid()
    index_time_list.clear()
    cash_point.clear()
    SL_point.clear()
    return (cash, accuracy)

# Определяем лучший процент стоп-лосса
def best_SL_percent (predict_1, predict_2, percent_1, percent_2,d_ind, y, y_test,cash, cash_start, x_date, ncc1, x_real,
                     start_pct, stop_pct):
    best_score, score_i, acc_i = 0, 0, 0
    for percent in range(start_pct, stop_pct, 1):
        percent = percent/10000
        score_i, acc_i = summary_prediction_SL(predict_1, predict_2, percent_1, percent_2,
                       d_ind, y, y_test,cash, cash_start, False, False, x_date, ncc1, percent, x_real)
        if best_score < score_i:
            best_score = score_i
            best_percent = percent
    summary_prediction_SL(predict_1, predict_2, percent_1, percent_2,
                       d_ind, y, y_test,cash, cash_start, True, False, x_date, ncc1, best_percent, x_real)
    print ('Лучший Stop Loss:',best_percent*100, '%')
    return (best_percent)