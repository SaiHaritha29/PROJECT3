import pandas as pd
import bar_chart_race as bcr
df=pd.read_csv("https://raw.githubusercontent.com/dexplo/bar_chart_race/master/data/urban_pop.csv")
bcr.bar_chart_race(df=df,title="populationrace")
