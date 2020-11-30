import pandas as pd
import numpy as np

#data = pd.read_csv('Micro_Climate_Data.csv')   
data = pd.read_csv('Ambient_Data.csv')    #  Reads in weather data


str1 = 'Results.csv'

data.drop_duplicates(subset=["date", "hour"], keep='first', inplace=True)   # Keeps one data point per hour
data = data.reset_index(drop=True)  # resets the index of data after duplicate removal

periods = 24    # periods in a day
daysAvg = 7     # number of days used for daily temperature average
AvgTempPeriod = periods*daysAvg

runTime = len(data.index)  # number of data points in the data can be automated to find count

# Labels columns for data
X = pd.DataFrame(columns=['Eggs', 'Larvae', 'Pupae', 'Emerged Adults', 'Blood-feeding Adults', \
                          'Gestating Adults', 'Ovipositing Adults'])

# Initial starting population after running the learning data
X.at[0, 'Eggs'] = 227469982
X.at[0, 'Larvae'] = 746497337
X.at[0, 'Pupae'] = 188382830
X.at[0, 'Emerged Adults'] = 5288614
X.at[0, 'Blood-feeding Adults'] = 9884535
X.at[0, 'Gestating Adults'] = 7754654
X.at[0, 'Ovipositing Adults'] = 6119789

X.at[0, 'Egg Hatching Rate'] = 0
X.at[0, 'Egg Mort Rate'] = 0
X.at[0, 'Larvae Develop Rate'] = 0
X.at[0, 'Larvae Mort Rate'] = 0
X.at[0, 'Pupae Develop Rate'] = 0
X.at[0, 'Pupae Mort Rate'] = 0
X.at[0, 'Emerged Develop Rate'] = 0
X.at[0, 'Blood-feeding Rate'] = 0
X.at[0, 'Ovipositing Rate'] = 0
X.at[0, 'Gestating Rate'] = 0
X.at[0, 'Beta'] = 0
X.at[0, 'Adult Mort Rate'] = 0

for t in range(1, runTime):
   
###############################################################################
    # Determines fluctuation in 24 hour period
    
    if t < periods-1:
        Fluct = 11.111
    else:
        data['Fluctuation'] = data['temp'].rolling(window=24).max() - data['temp'].rolling(window=24).min()
        Fluct = data.at[t, 'Fluctuation']
      
    if t < AvgTempPeriod-1:
        Temp = 28.189
    else:
        data['meanPeriodTemp'] = data['temp'].rolling(window=AvgTempPeriod).mean()
        Temp = data.meanPeriodTemp[t] # Captures temperature at each hour
    
###############################################################################
    # Establishes aquatic development rates
    
    eggHatchRatePerDay = 0.5070*np.exp(-(((Temp - 30.85)/12.82)**2))
    eggHatchRate = 1- (1- eggHatchRatePerDay)**(1/periods)
    X.at[t, 'Egg Hatching Rate'] = eggHatchRate
    
    larvaeDevelopRatePerDay = 0.1727*np.exp(-(((Temp - 28.40)/10.20)**2))
    larvaeDevelopRate = 1 - (1 - larvaeDevelopRatePerDay)**(1/periods)
    
    pupaeDevelopRatePerDay = 0.6020*np.exp(-(((Temp - 34.29)/15.07)**2))
    pupaeDevelopRate = 1 - (1 - pupaeDevelopRatePerDay)**(1/periods)   

    if Temp < 10 or Temp > 40:  # If the temperature is too low or too high, aquatic development stops
        eggHatchRate = 0
        larvaeDevelopRatePerDay = 0
        pupaeDevelopRatePerDay = 0
    
    X.at[t, 'Larvae Develop Rate'] = larvaeDevelopRate
    X.at[t, 'Pupae Develop Rate'] = pupaeDevelopRate    
###############################################################################
    # Establishes aquatic mortality rates
    
    eggMortRatePerDay = 0.05
    eggMortRate = 1 - (1-eggMortRatePerDay)**(1/periods)
    X.at[t, 'Egg Mort Rate'] = eggMortRate
    
    larvaeMortRatePerDay = 1
    if Temp < 36 and Temp > 0:   # anything above 36 and below 0 will not survive
        larvaeMortRatePerDay = np.minimum( (1/np.abs(-0.1305*Temp**2 + 3.868*Temp + 30.83)), 1)    
    
#    larvaeMortRatePerDay = np.exp(-Temp/2) + 0.08
    larvaeMortRate = 1 - (1-larvaeMortRatePerDay)**(1/periods)
    X.at[t, 'Larvae Mort Rate'] = larvaeMortRate
  
    pupaeMortRatePerDay = 1
    if Temp < 34.35 and Temp > 0: # anything above 34.35 and below 0 will not survive
        pupaeMortRatePerDay = np.minimum( (1/np.abs(-0.1502*Temp**2 + 5.057*Temp + 3.517)), 1)

#    pupaeMortRatePerDay = np.exp(-Temp/2) + 0.03
    pupaeMortRate = 1 - (1-pupaeMortRatePerDay)**(1/periods)
    X.at[t, 'Pupae Mort Rate'] = pupaeMortRate
    
  
############################################################################### 	
    # Establishes adult development rates
    
    AeDevelopDailyRate = 0.4
    AeDevelopRate = (1+AeDevelopDailyRate)**(1/periods) - 1
    
    AbDevelopDailyRate = 0.2
    AbDevelopRate = (1+AbDevelopDailyRate)**(1/periods) - 1
    
    AoDevelopDailyRate = 0.2
    AoDevelopRate = (1+AoDevelopDailyRate)**(1/periods) - 1
    
    if Temp < 9.5:
        AeDevelopRate = 0
        AbDevelopRate = 0
        AoDevelopRate = 0
    X.at[t, 'Emerged Develop Rate'] = AeDevelopRate
    X.at[t, 'Blood-feeding Rate'] = AbDevelopRate
    X.at[t, 'Ovipositing Rate'] =   AoDevelopRate
    
    AgDevelopRate = 0    
    TDD_Ag = 77
    TempAg = 10    
    
    if Temp > TempAg:
        AgDevelopRate = (Temp - TempAg)/TDD_Ag
        AgDevelopRate = (1+AgDevelopRate)**(1/periods) - 1
    X.at[t, 'Gestating Rate'] = AgDevelopRate
    
    # Egg laying rate
    betaPerDay = np.maximum(-15.837 + 1.2897*Temp - 0.0163*(Temp**2), 0)
    
    if Temp > 36.55:
        betaPerDay = 0
    
    if betaPerDay < 1:
        beta = (1+betaPerDay)**(1/periods) - 1
    else:
        beta = betaPerDay**(1/periods)
    X.at[t, 'Beta'] = beta     
###############################################################################
	# Establishes adult mortality rates
    
    adultMortRate = 1
    if 3.03 < Temp and Temp < 39.37:
        adultMortRate = np.minimum( (1/np.abs(-0.1921*Temp**2 + 8.147*Temp - 22.98)), 1)
    adultMortRate = 1 - (1-adultMortRate)**(1/periods)
    X.at[t, 'Adult Mort Rate'] = adultMortRate    
    
    EmAdultMortDailyRate = 0.1
    EmAdultMortRate = 1 - (1-EmAdultMortDailyRate)**(1/periods)
    
    RiskMortDaily = 0.08
    RiskMort = 1 - (1-RiskMortDaily)**(1/periods)
    
    if Temp < 9.5:
        RiskMort = 0
    
    sigma = 0.5 # Proportion female
	
###############################################################################
    # Environmental carrying capacity
    
    larvaeCarryingCapacity = 250000
    pupaeCarryingCapacity = 250000
    
###############################################################################
    # Steps through the population model
    
    X.at[t, 'Eggs'] = X.at[t-1, 'Eggs'] \
                        + beta*X.at[t-1, 'Ovipositing Adults'] \
                        - eggMortRate*X.at[t-1, 'Eggs'] \
                        - eggHatchRate*X.at[t-1, 'Eggs']
                          
    X.at[t, 'Larvae'] = X.at[t-1, 'Larvae'] \
                          + eggHatchRate*X.at[t-1, 'Eggs'] \
                          - larvaeMortRate*(X.at[t-1, 'Larvae'] + X.at[t-1, 'Larvae']/larvaeCarryingCapacity) \
                          - larvaeDevelopRate*X.at[t-1, 'Larvae']
                          
    X.at[t, 'Pupae'] = X.at[t-1, 'Pupae'] \
                         + larvaeDevelopRate*X.at[t-1, 'Larvae'] \
                         - pupaeMortRate*X.at[t-1, 'Pupae'] \
                         - pupaeDevelopRate*X.at[t-1, 'Pupae']
                         
    X.at[t, 'Emerged Adults'] = X.at[t-1, 'Emerged Adults'] \
                                  + X.at[t-1, 'Pupae']*pupaeDevelopRate*sigma*np.exp(-EmAdultMortRate*(1 + X.at[t-1, 'Pupae']/pupaeCarryingCapacity)) \
                                  - (adultMortRate+RiskMort)*X.at[t-1, 'Emerged Adults'] \
                                  - AeDevelopRate*X.at[t-1, 'Emerged Adults']
                                  
    X.at[t, 'Blood-feeding Adults'] = X.at[t-1, 'Blood-feeding Adults'] \
                                        + AeDevelopRate*X.at[t-1, 'Emerged Adults'] \
                                        + AoDevelopRate*X.at[t-1, 'Ovipositing Adults'] \
                                        - (adultMortRate + RiskMort)*X.at[t-1, 'Blood-feeding Adults'] \
                                        - AbDevelopRate*X.at[t-1, 'Blood-feeding Adults']
                                        
    X.at[t, 'Gestating Adults'] = X.at[t-1, 'Gestating Adults'] \
                                    + AbDevelopRate*X.at[t-1, 'Blood-feeding Adults'] \
                                    - adultMortRate*X.at[t-1, 'Gestating Adults'] \
                                    - AgDevelopRate*X.at[t-1, 'Gestating Adults']
                                    
    X.at[t, 'Ovipositing Adults'] = X.at[t-1, 'Ovipositing Adults'] \
                                      + AgDevelopRate*X.at[t-1, 'Gestating Adults'] \
                                      - (adultMortRate + RiskMort)*X.at[t-1, 'Ovipositing Adults'] \
                                      - AoDevelopRate*X.at[t-1, 'Ovipositing Adults']
        
X.to_csv(str1)









import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.ticker import MaxNLocator

###############################################################################
# Plotting the results                                      

    
ft = 5

x = np.linspace(1,t+1,t+1)

fig = plt.figure(figsize=(4,2), dpi=300)
ax = fig.add_subplot(1, 1, 1)

ax.plot(x, X['Eggs'], color='blue', linestyle = '-')
ax.plot(x, X['Larvae'], color='green')
ax.plot(x, X['Pupae'], color='gold')

#ax.plot(x, X['Emerged Adults'], color='orange', linestyle = '-')
#ax.plot(x, X['Blood-feeding Adults'], color='red')
#ax.plot(x, X['Gestating Adults'], color='deeppink')
#ax.plot(x, X['Ovipositing Adults'], color='orchid')
ax.plot(x, X['Emerged Adults'] + X['Blood-feeding Adults'] + X['Gestating Adults'] + X['Ovipositing Adults'], color='red')

ax.set_xlim([0, t])
ax.set_ylim(0, 1.5e9)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.set_ylabel('Number of Individuals (x $10^8$)',fontsize=6, weight='bold')
ax.set_xlabel('Time', fontsize=6, weight='bold')

#ax.set_title('line plot with data points')

ax.tick_params(axis='both', which='major', direction="out", labelsize=5)

my_xticks = np.array(['Jun.','Jul.','Aug.','Sept.','Oct.','Nov',
                      'Dec.','Jan.','Feb.','March'])
frequency = 720
plt.xticks(x[::frequency], my_xticks)



#ax.legend(["Eggs", 'Larvae', 'Pupae', 'Adults'], loc = 'upper left', prop={'size':5}, frameon=False)
ax.legend(["Emerged Adults", 'Blood-feeding Adults', 
           'Gestating Adults', 'Ovipositing Adults'], loc = 'upper left', prop={'size':5}, frameon=False)

plt.show()      

