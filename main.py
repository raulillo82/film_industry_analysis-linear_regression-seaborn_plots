import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:,.2f}'.format
import seaborn as sns
from sklearn.linear_model import LinearRegression
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

data = pd.read_csv('cost_revenue_dirty.csv')

#Explore and Clean the Data"
#CHALLENGE 1:
print("**Challenge**: Answer these questions about the dataset:")
print("1. How many rows and columns does the dataset contain?")
print("2. Are there any NaN values present?")
print("3. Are there any duplicate rows?")
print("4. What are the data types of the columns?")
print("")

#Answers
print(f"1: {data.shape}")
print(f"1: {data.head()}")
print(f"1: {data.tail()}")
print(f"1: {data.sample()}")
print("")
print("Info about the dataframe:")
print(data.info())

print(f"2: NaN values? {data.isna().values.any()}")
print(f"3: Duplicates? {data.duplicated().values.any()}")
print("")

#CHALLENGE 2:

print("Convert USD_Production_Budget, USD_Worldwide_Gross and USD_Domestic_Gross columns to a numeric format by removing $ signs and ,")
print("Note that domestic in this context refers to the United States.")

chars_to_remove = [',', '$']
columns_to_clean = ['USD_Production_Budget',
                    'USD_Worldwide_Gross',
                    'USD_Domestic_Gross']

for column in columns_to_clean:
    for char in chars_to_remove:
        #Remove undesired chars
        data[column] = data[column].astype(str).str.replace(char, "")
    #Convert to integer
    data[column] = pd.to_numeric(data[column])

print(data.head())

#CHALLENGE 3

print("Convert the Release_Date column to a Pandas Datetime type.")
data.Release_Date = pd.to_datetime(data.Release_Date)
print(data.head())
print("")
print("Info about the dataframe (processed):")
print(data.info())

print("Investigate the Films that had Zero Revenue")
#Challenge 1

print("1. What is the average production budget of the films in the data set?")
print("2. What is the average worldwide gross revenue of films?")
print("3. What were the minimums for worldwide and domestic revenue?")
print("4. Are the bottom 25% of films actually profitable or do they lose money?")
print("5. What are the highest production budget and highest worldwide gross revenue of any film?")
print("6. How much revenue did the lowest and highest budget films make?")

print("Printing some info, answering question 1-5:")
print(data.describe())

print("Answer for question 6:")
print(data[data.USD_Production_Budget == 1100])

#Challenge 2

print("How many films grossed $0 domestically (i.e., in the United States)?")
zero_domestic = data[data.USD_Domestic_Gross == 0]
print(f'Number of films that grossed $0 domestically: {len(zero_domestic)}')
print("What were the highest budget films that grossed nothing?")
print(zero_domestic.sort_values('USD_Production_Budget', ascending=False))


#Challenge 3

print("How many films grossed $0 worldwide?")
zero_worldwide = data[data.USD_Worldwide_Gross == 0]
print(f'Number of films that grossed $0 worldwide: {len(zero_worldwide)}')
print("What are the highest budget films that had no revenue internationally (i.e., the biggest flops)?")
print(zero_worldwide.sort_values('USD_Production_Budget', ascending=False))

#Filter on Multiple Conditions: International Films
print("Number of films without domestic revenue but with worldwide revenue:")
#international_releases = data.loc[(data.USD_Domestic_Gross == 0) &
#                                  (data.USD_Worldwide_Gross != 0)]
#Another option with .query() function:
international_releases = data.query('USD_Domestic_Gross == 0 and USD_Worldwide_Gross != 0')
#print(international_releases.head())
print(len(international_releases))

#Unreleased Films
#Challenge
#USD_Production_Budget  USD_Worldwide_Gross  USD_Domestic_Gross
print("Identify which films were not released yet as of the time of data collection (May 1st, 2018).")
scrape_date = pd.Timestamp("2018-5-1")
future_releases = data.loc[(data.Release_Date >= scrape_date)]
print(future_releases)
#Remove those films from future questions:
print("How many films are included in the dataset that have not yet had a chance to be screened in the box office?")
print(len(future_releases))
print("Create another DataFrame called data_clean that does not include these films.")
data_clean = data.drop(future_releases.index)
print("")

#Films that lost money
print("Having removed the unreleased films entirely can you calculate the percentage of films that did not break even at the box office?")
print("We already saw that more than the bottom quartile of movies appears to lose money")
print("However, what is the true percentage of films where the costs exceed the worldwide gross revenue?")
money_losing = data_clean.query('USD_Production_Budget > USD_Worldwide_Gross')
print(f"{float('{:.2f}'.format(len(money_losing) * 100 / len(data_clean)))}%")

#Seaborn data visualization: bubble charts
plt.figure(figsize=(8, 4), dpi=200)
# set styling on a single chart
with sns.axes_style('darkgrid'):
    #other options are: whitegrid, dark, ticks...

    ax = sns.scatterplot(data=data_clean,
                         x='USD_Production_Budget',
                         y='USD_Worldwide_Gross',
                         hue='USD_Worldwide_Gross',
                         size='USD_Worldwide_Gross')

    ax.set(ylim=(0, 3000000000),
           xlim=(0, 450000000),
           ylabel='Revenue in $ billions',
           xlabel='Budget in $100 millions')

#Plot
plt.show()

with sns.axes_style('darkgrid'):
    ax = sns.scatterplot(data=data_clean,
                         x='Release_Date',
                         y='USD_Production_Budget',
                         hue='USD_Worldwide_Gross',
                         size='USD_Worldwide_Gross')
    ax.set(ylim=(0, 450000000),
           xlim=(data_clean.Release_Date.min(),
                 data_clean.Release_Date.max()),
           ylabel='Budget in $100 millions',
           xlabel='Year')
plt.show()

#Create a decade column:
dt_index = pd.DatetimeIndex(data_clean.Release_Date)
years = dt_index.year
decades = years // 10 * 10
data_clean['Decade'] = decades
#print(data_clean.sample())

#Challenge:

print("Create two new DataFrames: old_films and new_films")
print("old_films should include all the films before 1970 (up to and including 1969)")
print("new_films should include all the films from 1970 onwards")
old_films = data_clean.loc[(data_clean.Decade < 1970)]
new_films = data_clean.loc[(data_clean.Decade >= 1970)]
print("How many of our films were released prior to 1970?")
print(len(old_films))
print("What was the most expensive film made prior to 1970?")
print(old_films.sort_values('USD_Production_Budget', ascending=False).head())

with sns.axes_style("whitegrid"):
    ax = sns.regplot(data=old_films,
                     x='USD_Production_Budget',
                     y='USD_Worldwide_Gross',
                     scatter_kws = {'alpha': 0.4},
                     line_kws = {'color': 'black'})
    ax.set(ylabel='Revenue in $100 millions',
           xlabel='Budget in $10 millions')

plt.show()

with sns.axes_style("darkgrid"):
    ax = sns.regplot(data=new_films,
                     x='USD_Production_Budget',
                     y='USD_Worldwide_Gross',
                     color='#2f4b7c',
                     scatter_kws = {'alpha': 0.3},
                     line_kws = {'color': '#ff7c43'})

    ax.set(ylim=(0, 3000000000),
           xlim=(0, 450000000),
           ylabel='Revenue in $ billions',
           xlabel='Budget in $100 millions')

plt.show()

#scikit-learn for own regressions
def scikit_learn_calcs(films):
    regression = LinearRegression()
    # Explanatory Variable(s) or Feature(s)
    X = pd.DataFrame(films, columns=['USD_Production_Budget'])
    # Response Variable or Target
    y = pd.DataFrame(films, columns=['USD_Worldwide_Gross'])

    # Find the best-fit line
    regression.fit(X, y)

    print("Print intercept: a movie with $0 budget would make:")
    print(regression.intercept_)

    print("Slope means that for every $1 the budget is increased, the film makes:")
    print(regression.coef_)

    # R-squared
    print(f"r-squared parameter of the model: {float('{:.2f}'.format(regression.score(X, y)*100))}%")

    budget = 350000000
    revenue_estimate = regression.intercept_[0] + regression.coef_[0,0]*budget
    revenue_estimate = round(revenue_estimate, -6)
    print(f'The estimated revenue for a $350 million film, according to this model, is around ${revenue_estimate:.10}.')

print("Some data from scikit-learn library about new films:")
scikit_learn_calcs(new_films)
print("")
print("Some data from scikit-learn library about old films:")
scikit_learn_calcs(old_films)
