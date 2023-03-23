import yfinance as yf
import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn import linear_model
import time
import pickle
import datetime
import mysql.connector
import smtplib
import itertools
import statistics
import matplotlib as plt
import seaborn as sns

'automated callculate_range still has to be tested when retraining the model'"!!!!!!!!"


#had to adjust because the server is in a -2h timezone
#target time is 22:00 my timezone so it's 20:00 in the servers timezone
#the second time check is in the morning at 10 am before

'''remember to build the architecture so that only the main loop calls functions '''

#######################################################################################################################
# preset data

testnow = ('00', '02', '04', '06', '08', '10', '12', '14', '16', '18', '20', '22')
send_now = ('16')
#train_period = '25wk' is automated now
test_range_days = 3

time_sat_ness = False
MySQL_enabled = False
sendmail_enabled = False
retrain_at_first_run = True

crash_restart = 1200

script_duration = 0 #global parameter

counter = itertools.count()
retrain_after_prediction_periods = 12

#######################################################################################################################
#database information

set_database = "172.105.78.117"
name_database = 'btc-analysis-db'
user = "petersservers"
password = "Strat1998*"
auth_plugin = 'mysql_native_password'

#######################################################################################################################
#input conversion to match script logic

test_times = len(testnow)
pauseperiod = (24/test_times)*(60**2)

if retrain_at_first_run is True:
    retrain_at_first_run = 0
else:
    retrain_at_first_run = 1

########################################################################################################################


#######################################################################################################################

def model_retrain():

    try:

        time_range = callculate_range()

    except:

        time_range = '25wk'
    # data time range
    # the longer the more accurate

    # donwload and rename relevant columns
    BTC = yf.download(tickers='BTC-USD', period=time_range, interval='1d')
    BTC.rename(columns={'Adj Close': 'BTC Closing'}, inplace=True)

    SaP = yf.download('SPY', period=time_range, interval="1d")
    SaP.rename(columns={'Adj Close': 'S&P Closing'}, inplace=True)

    Gold = yf.download('GC=F', period=time_range, interval="1d")
    Gold.rename(columns={'Adj Close': 'Gold Closing'}, inplace=True)

    Bonds = yf.download('MXBIX', period=time_range, interval="1d")
    Bonds.rename(columns={'Adj Close': 'Bonds Closing'}, inplace=True)

    BTC.to_pickle("BTC.pkl")
    SaP.to_pickle("SaP.pkl")
    Gold.to_pickle("Gold.pkl")
    Bonds.to_pickle('Bonds.pkl')

    btc_closing = BTC['BTC Closing'].pct_change()
    btc_closing = pd.Series(btc_closing, index=SaP.index)
    gold_closing = Gold['Gold Closing'].pct_change()
    gold_closing = pd.Series(gold_closing, index=SaP.index)
    sap_closing = SaP['S&P Closing'].pct_change()
    bond_closing = Bonds['Bonds Closing'].pct.change()
    bond_closing = pd.Series(bond_closing, index=SaP.index)

    full_data = pd.concat([gold_closing, sap_closing, bond_closing, btc_closing], axis=1)

    full_data = full_data.fillna(full_data.mean())

    X = np.array(full_data.drop(['BTC Closing'], 1))
    y = np.array(full_data['BTC Closing'])

    actual_test_value = 0
    test_lenght = 100000

    for i in range(test_lenght):

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

        linear = linear_model.LinearRegression()

        linear.fit(x_train, y_train)

        # test the accuracy of the model
        accuracy_of_model = linear.score(x_test, y_test)

        if accuracy_of_model > actual_test_value:
            actual_test_value = accuracy_of_model
            print(f"Model increased accuracy {actual_test_value}")
            with open("BTCpredictionmodel.pickle", "wb") as f:
                pickle.dump(linear, f)

        if i == (test_lenght-1): #I have to use -1 because python i 0 indexed, meaning that the last i
            #in a range of 10000 is 9999

            information = (f"BTC APIML1 model has just been retrained with data from in the range of {time_range}, "
                           f"final accuracy reached {actual_test_value}")

            return information

#######################################################################################################################

def load_model():

    pickle_in = open("BTCpredictionmodel.pickle", "rb")
    linear = pickle.load(pickle_in)
    return linear

#######################################################################################################################


def check_time(): #checktime is 4 pm eastern time US (when the S&P normally closes, but BTC is still open)

    now = f"{datetime.datetime.now():%H}"

    if now in testnow:
        print(f'the time conditions was satisfied with {now}, as the condition is {testnow}')
        return True

    else:
        return False

def check_send():  # checktime is 4 pm eastern time US (when the S&P normally closes, but BTC is still open)

    now = f"{datetime.datetime.now():%H}"
    if now in send_now:
        print(f'the time conditions was satisfied with {now}, as the condition is {send_now}')
        return True
    else:
        return False

#######################################################################################################################

def start_predicition(linear):

    global test_range_days

    prediction_period = f"{test_range_days}d"
    # chose to take 2 days instead of one,
    # that way the programm still works even if the market is not in the state
    # where the daily closing prices are already published

    BTC_ = yf.download(tickers='BTC-USD', period=prediction_period, interval='1d')
    BTC_.rename(columns={'Adj Close': 'BTC Closing'}, inplace=True)

    SaP_ = yf.download('SPY', period=prediction_period)
    SaP_.rename(columns={'Adj Close': 'S&P Closing'}, inplace=True)

    Gold_ = yf.download('GC=F', period=prediction_period, interval="1d")
    Gold_.rename(columns={'Adj Close': 'Gold Closing'}, inplace=True)

    Bonds_ = yf.download('MXBIX', period=prediction_period, interval="1d")
    Bonds_.rename(columns={'Adj Close': 'Bonds Closing'}, inplace=True)

    BTC_ = BTC_['BTC Closing'].pct_change()
    SaP_ = SaP_['S&P Closing'].pct_change()
    Gold_ = Gold_['Gold Closing'].pct_change()
    Bonds_ = Bonds_['Bonds Closing'].pct_change()

    sap_closing_test = SaP_['S&P Closing']
    btc_last_price = BTC_['BTC Closing'][-1]

    btc_closing_test = pd.Series(BTC_['BTC Closing'], index=SaP_.index)
    gold_closing_test = pd.Series(Gold_['Gold Closing'], index=SaP_.index)
    bonds_closing_test = pd.Series(Bonds_['Bonds Closing'], index=SaP_.index)

    test_data = pd.concat([gold_closing_test, sap_closing_test, bonds_closing_test], axis=1)

    test_data = test_data.fillna(test_data.mean())

    if test_data.isnull().values.any():
        print(f'Null values in dataframe with test_day_range{test_range_days}')

    else:
        print(f'no null values in dataframe')

    #fill all the NaN values with a the average of the dataframe of the last 3 days
    #this implements redundency and guarantees that the downloading of the model is still going to work
    #with missing values

    test_data = test_data.tail(1)

    test_data_array = np.array(test_data)

    day_of_test = str(test_data.index[0])
    day_of_test = day_of_test.split(" ", 1)
    day_of_test = day_of_test[0]

    prediction_results = float(linear.predict(test_data_array))
    gold_store_value = float(test_data_array[0][0])
    sap_store_value = float(test_data_array[0][1])
    bonds_store_value = float(test_data_array[0][2])
    btc_store_value_last_price = float(btc_last_price)
    actual_difference = prediction_results - btc_store_value_last_price

    if MySQL_enabled is True:

        store_values(day_of_test, gold_store_value, sap_store_value, bonds_store_value, prediction_results, btc_store_value_last_price, actual_difference)

        print('values stored')

    operations = True

    inhalt1 = (f'last cycle of prediction successfull = {str(operations)}, all information of {day_of_test} was available and is stored in db {set_database}, actuall difference amounts to {actual_difference}')

    sendcheck = check_send()

    if sendcheck is True:

        sendmail(inhalt1)

    return (operations, prediction_results, btc_last_price, actual_difference)



#######################################################################################################################

def store_values(day_of_test, gold_close, sap_close, bonds_close, btc_est, btc_actual, actual_difference):

    mydb = mysql.connector.connect(host=set_database, user=user, password=password, database=name_database, auth_plugin=auth_plugin)

    c = mydb.cursor()

    c.execute("INSERT INTO `ML1`(`day_of_test`, `gold_close`, `sap_close`, `bonds_close`, `btc_est`, `btc_actual`,`actual_difference`) "
              "VALUES (%s, %s, %s, %s, %s, %s, %s);",(day_of_test, gold_close, sap_close, bonds_close, btc_est, btc_actual, actual_difference))

    mydb.commit()

#######################################################################################################################

def sendmail(inhalt):

    if sendmail_enabled is True:

        try:

            sender_email = "Peterservers@yahoo.com"
            reciever_email = "Peter_pichler@yahoo.com"
            password = ("xxxxxxxx")
            inhalt = inhalt
            message = 'Subject: {}\n\n{}'.format("API&ML1 Notification", inhalt)
            server = smtplib.SMTP("smtp.mail.yahoo.com", 587)
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, reciever_email, message)
            print(f"mail sent to {reciever_email}")

        except:

            print('mail could not be sent, check smtplib authentication')

#######################################################################################################################

def main():

    global script_duration

    while True:

        right_state = check_time()

        if right_state is True or time_sat_ness is False:

             start = time.time()

             reps = int(next(counter)+retrain_at_first_run) #this is necessary so that the model doesn't retrain immidiately

             print('condition satisfied')

             if reps %retrain_after_prediction_periods == 0:

                 model_information = model_retrain()

                 inhalt = (str(model_information))

                 if sendmail_enabled is True:

                     sendmail(inhalt)

             linear = load_model()

             operations = start_predicition(linear)

             if operations[0] is True:

                print(f'calculable is {operations[0]}, prediction results are {operations[1]}, '
                      f'actual price of btc is {operations[2]}, acutal difference amounts to {operations[3]}. sleep time active.')

                stop = time.time()

                script_duration = float(format(stop-start))

                calculated_pause = (pauseperiod-script_duration+0.01)

                #this part ensures that the script has the perfect sleep time
                #the concept works only if the times that the script runs are y = 24/x
                # = 24 h are equally divided

                print(f'script duration amounts to {script_duration}')

                print(f"the calculated pause is {calculated_pause}")

                time.sleep(calculated_pause)


                continue

             else:

                print(f'calculable is {operations[0]}, problem is {operations[1]}, sleep time active')

                stop = time.time()

                script_duration = float(format(stop - start))

                calculated_pause = (pauseperiod - script_duration + 0.01)

                time.sleep(calculated_pause)

                continue

        else:

            #this should actually never really happen as the sleep time should work perfectly
            print('time condition not satisfied')
            time.sleep(100)


#test period calculations

'''how do i find the most accurate value in the 3 distributions????'''

plt.style.use('seaborn-darkgrid')

correlation_snp = []
correlation_gold = []
correlation_bonds = []

def callculate_range():

    correlation_snp.clear()
    correlation_gold.clear()
    correlation_bonds.clear()

    starting_point = 9

    for i in range(starting_point, 52):

        time_range = f'{i}wk'

        BTC = yf.download(tickers='BTC-USD', period=time_range, interval='1d')
        BTC.rename(columns={'Adj Close': 'BTC Closing'}, inplace=True)

        SaP = yf.download('SPY', period=time_range, interval="1d")
        SaP.rename(columns={'Adj Close': 'S&P Closing'}, inplace=True)

        Gold = yf.download('GC=F', period=time_range, interval="1d")
        Gold.rename(columns={'Adj Close': 'Gold Closing'}, inplace=True)

        Bonds = yf.download('MXBIX', period=time_range, interval="1d")
        Bonds.rename(columns={'Adj Close': 'Bonds Closing'}, inplace=True)


        btc_closing = BTC['BTC Closing']
        btc_closing = pd.Series(btc_closing, index=SaP.index)
        gold_closing = Gold['Gold Closing']
        gold_closing = pd.Series(gold_closing, index=SaP.index)
        sap_closing = SaP['S&P Closing']
        bond_closing = Bonds['Bonds Closing']
        bond_closing = pd.Series(bond_closing, index=SaP.index)

        correlation_snp.append(btc_closing.corr(sap_closing))
        correlation_gold.append(btc_closing.corr(gold_closing))
        correlation_bonds.append(btc_closing.corr(bond_closing))

    correlation_snp_median = float(statistics.median(correlation_snp))
    correlation_gold_median = float(statistics.median(correlation_gold))
    correlation_bonds_median = float(statistics.median(correlation_bonds))

    print(f'the median correlation between BTC and the S&P500 (SPY) amounts to: {correlation_snp_median}, '
          f'it can be found in week {correlation_snp.index(correlation_snp_median)}')

    print(f'the correlation between BTC and Gold (GC=F) amounts to: {correlation_gold_median}, '
          f'it can be found in week {correlation_gold.index(correlation_gold_median)}')

    print(f'the correlation between BTC and western bonds (MXBIX) amounts to: {correlation_bonds_median}, '
          f'it can be found in week {correlation_bonds.index(correlation_bonds_median)}')

    average_median_assets = int((correlation_snp.index(correlation_snp_median) + \
                            correlation_gold.index(correlation_gold_median) + \
                            correlation_bonds.index(correlation_bonds_median))/3)

    print(f'the average median of the assets can be found in week {average_median_assets}')

    return f'{average_median_assets}wk'

#######################################################################################################################

'''this does work as the while True loop continues to check time also if the function won't execute immidiately'''
'''to align the process perfectly is problematic, as i would need to callculate the runtime of the script in the main
loop, and if a function returns something, the function breaks'''
'''I had to declare the parameter as 0 in the beginning of the script and use it as global parameter so that it gets
overwritten every time that the main loop runs'''
'''by adding crash restart - the global parameter of script duration, the sleeptime before restarting the script 
amounts to 20 minutes - script duration. The following pauseperiod after the script is executed again then can
only take the normal sleeptime - the crash restart, as the script duration was already callculated in in the first
time.sleep period'''

if __name__ == '__main__':
    errors = []
    while True:
        try:
            main()
            errors.clear() #the errors only get cleared if the main function is executed properly
            test_range_days = 3 #reset the test range days to 3
            #this architecture is necessary so that the script is as accurate as possible

        except Exception as e:
            Error = f'ERROR OCCURED: {e} !script will be run again in {crash_restart-script_duration} seconds!'
            print(Error)
            errors.append(Error)
            sendmail(Error)

            if len(errors) > 3:
                test_range_days+=1
            #the script is run again after 20 seconds and after 3 error codes the script adds a day to the testdays
            #and averages it out
            time.sleep(crash_restart-script_duration)
            main() #if this execution fails than the try starts again after 20 minutes as the finally statement guarantees
            #that the next part of the script will be executed
            time.sleep(pauseperiod-crash_restart)
        finally:
            continue

            #test

            '''logic of this script explained'''
            '''sitation x 8:00 : fail 1 20 min fail 2 20 min fail 3 20 min
            testdays +=1. 9:00 script works but won't execute because the time doesn't check out
            the time.sleep(200) is active till it's ten. then the script is exit success, the errors get cleared
            and the test range days are 3 days again, should the script not work with 4 days of test range data, 
            then the test range days get increased to 5 in the next term, and if it doesn't work even then, the script 
            continuously increases the number of test range days till it works, and then sets them back to 3 again whenever the testing '''

#######################################################################################################################

'''the script is designed to run between the closing price of the S&P/Gold and BTC'''
'''S&P closes at 6pm eastern time'''
'''it should be able to predict how much the price is going to rise till the end of the day'''

# does this project make sense?
# does the math make sense?

# can i predict the closing price BTC with the closing prices of S&P and Gold?

