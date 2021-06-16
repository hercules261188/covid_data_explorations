import glob
import json
import requests
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import StrMethodFormatter, MultipleLocator
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, RidgeCV
from IPython.display import display


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

DAYS_PER_MONTH = [np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]),
                  np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
                  ]

### Functions for getting deaths and population by age
def get_sweden_deaths():
    ages = "0-64", "65-79", "80-89", "90+"
    df = (
        pd.read_excel("data/Sweden/deaths_current.xlsx", sheet_name=2, header=[5, 6, 7])
        .dropna(how="all")
        .iloc[:, :13]
    )
    df.columns = [
        "year",
        "date",
        "total",
        "M total",
        "M 0-64",
        "M 65-79",
        "M 80-89",
        "M 90+",
        "F total",
        "F 0-64",
        "F 65-79",
        "F 80-89",
        "F 90+",
    ]

    df_other = (
        df.query('date == "Okänd dödsdag "')
        .drop(columns=["date"])
        .astype("int")
        .set_index("year")
        .sort_index()
    )
    df_known = df.query('date != "Okänd dödsdag "').astype({"year": "int"})
    df_known_total = df_known.groupby("year").sum(numeric_only=True).astype("int")
    df_total = df_known_total + df_other
    df_deaths = pd.DataFrame(df_total.iloc[:, 2:6].values + df_total.iloc[:, 7:].values)
    df_deaths.index = df_total.index
    df_deaths.columns = ages

    month = {
        "december": "December",
        "november": "November",
        "oktober": "October",
        "september": "September",
        "augusti": "August",
        "juli": "July",
        "juni": "June",
        "maj": "May",
        "april": "April",
        "mars": "March",
        "februari": "February",
        "januari": "January",
    }

    df_known["month"] = df_known["date"].str.split().str[1].replace(month)
    for age in ages:
        df_known[age] = df_known[f"M {age}"] + df_known[f"F {age}"]
        df_other[age] = df_other[f"M {age}"] + df_other[f"F {age}"]
    df_known = df_known[["0-64", "65-79", "80-89", "90+"]][::-1]
    df_known["date"] = pd.date_range("2015-01-01", periods=len(df_known))
    df_known["year"] = df_known["date"].dt.year
    df_known.reset_index(drop=True, inplace=True)
    df_other = df_other[["0-64", "65-79", "80-89", "90+"]]
    s = df_known.groupby("year").size()
    daily_other = df_other.div(s, axis=0)
    df_all = df_known.merge(daily_other, on="year")
    for age in ages:
        df_all[age] = df_all[f"{age}_x"] + df_all[f"{age}_y"]
    df_all = df_all[["date", "0-64", "65-79", "80-89", "90+"]].set_index("date")
    # daily deaths are "age during the year" different deaths than
    return df_all


def get_sweeden_historical_data():
    def get_data(kind, years):
        years = list(years)
        d = json.load(open('data/Sweden/api_query.json'))
        url = d['url'][kind]
        filt = d['filter'][kind]
        query = d['query']
        max_age = d['max_age'][kind]
        
        ages = list(map(str, range(max_age))) + [f'{max_age}+']
        query['query'][1]['selection']['values'] = ages
        query['query'][2]['selection']['values'] = years
        query['query'][1]['selection']['filter'] = filt
        
        if kind in ('end', 'deaths'):
            del query['query'][0]
            
        r = requests.post(url, json=query)
        data = StringIO(r.text)
        df = pd.read_csv(data)
        df = df.drop(columns='age')
        df.index = ages
        if kind == "start":
            df = df.drop(columns=['region'])
        df = df.T

        if kind == 'start':
            df.index = [2021]
        else:
            df.index = years
        if max_age > 100:
            s = df.loc[:, '100':].sum(axis=1)
            df = df.loc[:, :'99']
            df['100+'] = s

        df.columns = range(101)
        return df

    df_deaths1 = pd.read_csv('data/Sweden/deaths.csv', index_col='year') # smaller
    # df_deaths2 = pd.read_csv('data/Sweden/deaths_year_of_birth.csv', index_col='year') # larger
    df_age = pd.read_csv('data/Sweden/age.csv', index_col='year')
    df_deaths = df_deaths1

    df_age = df_age + df_deaths
    df_age_start = get_data('start', ['2021M01'])
    df_age.loc[2021] = df_age_start.loc[2021].values
    df_age.iloc[-1, 0] = df_age.iloc[-2, 0]
    
    # df_age_start = get_data('start', ['2021M01'])
    # df_age.loc[2021] = df_age_start.loc[2021].values - df_deaths.iloc[-1].values

    df_age.columns = df_age.columns.astype('int64')
    df_deaths.columns = df_deaths.columns.astype('int64')
    return df_deaths, df_age


def get_nz_deaths():
    # from http://infoshare.stats.govt.nz/
    # use "as at" for quarter ending in March
    file = sorted(glob.glob("data/New Zealand/2021*"))[-1]
    cols = ["indicator_name", "category", "sub_series_name", "parameter", "value"]
    ages = ["Under 30", "30 to 59", "60 to 79", "80 and over", "Total"]
    query = 'indicator_name == "Weekly deaths by age"'
    df = (
        pd.read_csv(file, usecols=cols, low_memory=False)
        .query(query)
        .rename(columns={"sub_series_name": "age", "parameter": "date", "value": "deaths"})
        .astype({"date": "datetime64[ns]"})
        .pivot(index="date", columns="age", values="deaths")
        .rename_axis(index=None, columns=None)
    )
    df = df[ages]
    df.iloc[-2] *= 1.01
    df.iloc[-1] *= 1.04
    df = df.astype("int")
    df_daily = df.reindex(pd.date_range("2011", "2021-12-31")).fillna(method="bfill").dropna() / 7
    df_daily = df_daily.iloc[:, :-1]
    df_daily.columns = ["0-29", "30-59", "60-79", "80+"]
    return df_daily


def get_norway_deaths():
    q = json.load(open("data/Norway/api_query.json"))
    headers = {"Content-type": "application/json", "Accept": "*/*"}
    r = requests.post("https://data.ssb.no/api/v0/en/table/07995", json=q)
    data = r.json()

    dfs = []
    for i in range(106):
        df = pd.DataFrame(
            np.reshape(data["value"][i * 424 : 424 * (i + 1)], (-1, 8)),
            columns=range(2014, 2022),
        )
        df = df.stack().reset_index()
        df.columns = ["week", "year", "deaths"]
        df["age"] = i
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.pivot(index=("year", "week"), columns="age", values="deaths").fillna(0)
    df = df[df.any(axis=1)].reset_index(drop=True)
    df.loc[:, 100] = df.loc[:, 100:].sum(axis=1)
    df = df.loc[:, :100]
    df.index = pd.date_range(start="2014-1-5", periods=len(df), freq="W")
    df = df.reindex(pd.date_range(start="2014-1-1", end=df.index[-1]), method="bfill") / 7
    bins = [0, 5, 20, 35, 50, 65, 80, 90, 130]
    g = pd.cut(df.columns, bins, right=False, precision=0)
    df = df.groupby(g, axis=1).sum()
    names = []
    for col in df.columns:
        names.append(f'{col.left}-{col.right - 1}')
    names[-1] = '90+'
    df.columns = names
    return df


def get_denmark_deaths():
    data = json.load(open("data/Denmark/api_query.json"))
    url = "https://api.statbank.dk/v1/data"
    r = requests.post(url, json=data)
    s = r.content.decode().replace(";", ",")
    with open("data/Denmark/deaths_weekly.csv", "w") as f:
        f.write(s)

    df = pd.read_csv("data/Denmark/deaths_weekly.csv", names=["date", "age", "deaths"], skiprows=1)
    df = df.query('age != "Total"').copy()
    df["date"] = pd.to_datetime(df["date"] + "D0", format="%YU%UD%w")
    df["age"] = df["age"].str.split().str[0]
    ages = df["age"].unique()
    n = len(ages)
    df.iloc[n:, 0] += pd.Timedelta("7D")
    df = df.pivot(index="date", columns="age", values="deaths")[ages]
    df = df.reindex(pd.date_range("2021-01-01", df.index[-1]), method="bfill") / 7
    df1 = pd.read_csv("data/Denmark/daily07-20.csv", parse_dates=["date"], index_col="date" )
    df = pd.concat([df1, df])
    df["90-94"] += df["95-99"] + df["100"]
    df = df.loc[:, :"90-94"].rename(columns={"90-94": "90+"})

    df_final = df[['0-4']].copy()
    left = ['5-9', '20-24', '35-39', '50-54', '65-69', '80-84']
    right = ['15-19', '30-34', '45-49', '60-64', '75-79', '85-89']
    last = df.columns[-1]
    for col_left, col_right in zip(left, right):
        new_left = col_left.split('-')[0]
        new_right = col_right.split('-')[1]
        new_col = f'{new_left}-{new_right}'
        df_final[new_col] = df.loc[:, col_left:col_right].sum(axis=1)

    df_final[last] = df[last]
    return df_final


def get_finland_deaths():
    url = "https://pxnet2.stat.fi:443/PXWeb/api/v1/en/Kokeelliset_tilastot/vamuu_koke/koeti_vamuu_pxt_12ng.px"
    data = json.load(open("data/Finland/api_query.json"))
    last_val = data["query"][1]["selection"]["values"][-1]
    next_week = int(last_val[-2:]) + 1
    new_last = f"{last_val[:5]}{next_week:02}"
    data["query"][1]["selection"]["values"].append(new_last)
    r = requests.post(url, json=data)
    if not r.ok:
        data["query"][1]["selection"]["values"].remove(new_last)
        r = requests.post(url, json=data)
    df = pd.read_csv(BytesIO(r.content)).drop(columns=["Area", "Week"])
    df.columns = df.columns.str[:-13]
    df = df.rename(columns={"90 -": "90+"})
    df.index = pd.date_range("2015-1-4", periods=len(df), freq="W")
    df = df.reindex(pd.date_range("2015-1-1", df.index[-1]), method="bfill") / 7
    df.columns = df.columns.str.replace(' ', '')


    df_final = df[['0-4']].copy()
    left = ['5-9', '20-24', '35-39', '50-54', '65-69', '80-84']
    right = ['15-19', '30-34', '45-49', '60-64', '75-79', '85-89']
    last = df.columns[-1]
    for col_left, col_right in zip(left, right):
        new_left = col_left.split('-')[0]
        new_right = col_right.split('-')[1]
        new_col = f'{new_left}-{new_right}'
        df_final[new_col] = df.loc[:, col_left:col_right].sum(axis=1)

    df_final[last] = df[last]
    return df_final


def get_uk_deaths():
    df = pd.read_excel("data/UK/deaths_202117.xlsx", 4, header=5).iloc[11:31, 1:-1]
    df = df.set_index(df.columns[0]).T
    df.index = pd.to_datetime(df.index)
    df = df.dropna().astype("int")
    df = df.rename_axis(columns=None)
    df = df.rename(columns={"<1": "Under 1 year"})

    df["01-14"] = df.iloc[:, 1:4].sum(axis=1)
    df["15-44"] = df.loc[:, "15-19":"40-44"].sum(axis=1)
    df["45-64"] = df.loc[:, "45-49":"60-64"].sum(axis=1)
    df["65-74"] = df.loc[:, "65-69":"70-74"].sum(axis=1)
    df["75-84"] = df.loc[:, "75-79":"80-84"].sum(axis=1)
    df["85+"] = df.loc[:, "85-89":].sum(axis=1)
    cols = df.columns[0:1].tolist() + df.columns[-6:].tolist()
    df = df[cols].rename_axis(index="date", columns=None)
    return df


####### End getting country data

country_dict = {
    "Sweden": {"max_age": 110, "func": get_sweden_deaths},
    "Norway": {"max_age": 110, "func": get_norway_deaths},
    "New Zealand": {"max_age": 105, "func": get_nz_deaths},
    "Denmark": {"max_age": 105, "func": get_denmark_deaths},
    "Finland": {"max_age": 110, "func": get_finland_deaths},
}


def convert_age_to_slice(age):
    if age.endswith("+"):
        age = int(age[:-1])
        return slice(age, None)
    elif '-' not in age:
        return slice(None)
    start, stop = age.split("-")
    start = int(start)
    stop = int(stop)
    return slice(start, stop)


def add_line_name(names, ax):
    for name, line in zip(names, ax.lines):
        arr = line.get_ydata()
        if hasattr(arr, "mask"):
            val = arr.data[~arr.mask][-1]
        else:
            val = arr[-1]
        x = line.get_xdata()[-1].to_timestamp() + pd.Timedelta("5D")
        ax.text(x, val, name, color=line.get_color(), size=8, va="center")


def plot_excess_bars(title, countries, year='2019', age='total'):
    """
    Bar plot of each countries excess deaths for each year in 2019-2021
    """
    names = []
    vals = []
    for c in countries:
        vals.append(c.excess_deaths_daily.loc[year:, age].resample('Y', kind='period').sum())
        names.append(c.country)

    df = pd.concat(vals, axis=1, keys=names).tail(3)
    df.loc["{year}-21 Total"] = df.sum()
    ax = df.plot(kind="bar", rot=0)
    ax.set_title(title, size=10)


def plot_cumulative_excess(start, end, countries, age='total', ax=None, **kwargs):
    start, end = str(start), str(end)
    names = []
    vals = []
    for country in countries:
        names.append(country.country)
        s = country.excess_deaths_daily.loc[start:end, age]
        vals.append(s)
        
    df = pd.concat(vals, axis=1, keys=names).cumsum()

    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    df.plot(legend=False, ax=ax)
    add_line_name(names, ax)
    ax.axhline(color="black", ls="--")

    year_string = f"{end}" if start == end else f"{start}-{end}"
    ax.set_title(f"{year_string} Cumulative Excess Deaths", size=10)
    ax.text(
        0.1,
        0.02,
        "Created by Teddy Petrou (@TedPetrou)",
        color="black",
        size=8,
        ha="left",
        transform=ax.transAxes,
        style="italic",
    )
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    return df


def plot_grid_expected(start, end, countries, age=0):
    fig = plt.figure(constrained_layout=True, figsize=(8, 8))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 0.05, 1, 1])
    ax_top = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(gs[3, 0])
    ax4 = fig.add_subplot(gs[3, 1])
    axes = ax1, ax2, ax3, ax4

    df = plot_cumulative_excess(start, end, countries, age, ax_top)

    for country, ax in zip(countries, axes):
        handles = country.plot_trend_line(2021, age, ax=ax, legend=False, auto_label=False, return_data=True)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    fig.legend(
        handles,
        ["Model", "Current Pace", "Expected"],
        ncol=3,
        bbox_to_anchor=(0.5, 0.27),
        loc="center",
        framealpha=0,
        columnspacing=0.5,
        handletextpad=0.2,
    )

    year_string = f"{end}" if start == end else f"{start}-{end}"
    ax_top.set_title(f"{year_string} Cumulative Excess Deaths", size=10)
    fig.text(
        0.5,
        0.57,
        "Model for Deaths Standardized to 2021 Age-Stratified Population",
        ha="center",
        size=11,
    )
    fig.suptitle("Nordic Countries Expected vs Actual Deaths", size=15, y=1.035)
    fig.savefig("images/nordic_grid.png", facecolor="white", bbox_inches="tight")


def plot_excess_baseline_other(countries, baseline, ax=None, **kwargs):
    names = []
    dfs = []
    labels = []
    baseline_name = baseline.country
    for c in countries:
        exp = c.age_grouped * baseline.rate_pred
        act = c.age_grouped * baseline.rate_grouped
        ex = act.sum(axis=1) - exp.sum(axis=1)
        dfs.append(ex.loc[2019:2020])
        names.append(c.country)
        labels.append(f"{baseline_name} with {c.country}'s population")
    df = pd.concat(dfs, axis=1, keys=names)

    if ax is None:
        if not kwargs:
            kwargs['figsize'] = (7, 3)
        fig, ax = plt.subplots(**kwargs)

    df.plot(kind="bar", rot=0, ax=ax)
    ax.legend(labels=labels)
    ax.set_ylabel("Excess Deaths")
    ax.set_title(f"Excess Deaths if {baseline.country} had Other Country's Age-Stratified Population", size=10,)


def plot_perc_over_age(age, year, countries):
    over = []
    names = []
    for c in countries:
        p = c.age.loc[:, age:].sum(axis=1) / c.age.sum(axis=1) * 100
        names.append(c.country)
        over.append(p)

    df = pd.concat(over, keys=names, axis=1)
    s = df.loc[year].sort_values()

    ax = s.plot(kind="bar", rot=0)
    ax.legend(bbox_to_anchor=(1, 1))
    ax.set_title(f"{year} Percent of Population {age} and over", size=10)
    return s


def plot_perc_mort_change(countries, age=70, year=2019, **kwargs):
    changes = []
    names = []
    for c in countries:
        change = c.rate_grouped.loc[year, age:] / c.rate_pred.loc[year, age:] - 1
        changes.append(change)
        names.append(c.country)
        
    df = pd.concat(changes, axis=1, keys=names) * 100
    idx = [f"{i.left}-{i.right - 1}" for i in df.index[:-1]]
    idx.append(f"{df.index[-1].left}+")
    df.index = idx
    df.index.name = "age"
    fig, ax = plt.subplots(**kwargs)
    df.plot(kind="bar", rot=0, ax=ax)
    ax.set_title(f"Percentage Deviation from Mortality Rate Trend Line from {year - 1} to {year}", size=10)
    ax.set_ylabel("% change")
    ax.legend(bbox_to_anchor=(1, 1))


class Excess:
    def __init__(self, country, start_year, last_training_year, pred_year, skip, group_daily=False):
        cur_dict = country_dict[country]
        self.country = country
        self.max_age = cur_dict["max_age"]
        self.start_year = start_year
        self.skip = skip
        self.get_country_func = cur_dict["func"]
        self.last_training_year = last_training_year
        self.pred_year = pred_year
        self.group_daily = group_daily

        self.get_historical_data()
        self.get_age_deaths_rate()
        self.cur_year_deaths = self.get_current_deaths()
        self.last_daily_date = self.daily_deaths.index[-1]
        self.month_mult = self.compute_month_mult()
        self.get_expected_month()

    def get_historical_data(self):
        if self.country == "Sweden":
            deaths, age = get_sweeden_historical_data()
        else:
            age = pd.read_csv(f"data/{self.country}/age.csv", index_col="year")
            age = age.loc[self.start_year:]
            age.columns = age.columns.astype("int")
            deaths = pd.read_csv(f"data/{self.country}/deaths.csv", index_col="year")
            deaths = deaths.loc[self.start_year :]
            deaths.columns = deaths.columns.astype("int")

            last = age.loc[2021]
            age += deaths
            age.loc[2021] = last + deaths.loc[2020]
            

        self.age = age.loc[self.start_year:]
        self.deaths = deaths.loc[self.start_year:]

    def get_age_deaths_rate(self):
        other_ages = list(range(45, 100, 5)) + [130]
        bins = [0, 5, 10, 15, 30, 40] + other_ages
        # bins = [0, 15, 30, 40, 50, 60, 65, 70, 80, 90, 95, 130]
        self.age_grouper = pd.cut(self.age.columns, bins, right=False, precision=0)

        deaths_grouped = self.deaths.groupby(self.age_grouper, axis=1).sum()
        age_grouped = self.age.groupby(self.age_grouper, axis=1).sum()
        rate_grouped = (deaths_grouped / age_grouped).dropna()

        rate_pred = rate_grouped.apply(self.predict_deaths)
        expected_deaths = rate_pred * age_grouped

        self.deaths_grouped = deaths_grouped
        self.age_grouped = age_grouped
        self.rate_grouped = rate_grouped
        self.rate_pred = rate_pred
        self.expected_deaths = expected_deaths
        self.excess_age = self.deaths_grouped - self.expected_deaths
        self.excess = self.excess_age.sum(axis=1)

    def get_current_deaths(self):
        self.daily_deaths = self.get_country_func()
        if self.skip > 0:
            self.daily_deaths = self.daily_deaths.iloc[: -self.skip]

        self.daily_deaths["total"] = self.daily_deaths.sum(axis=1)
        return (
            self.daily_deaths.loc["2021"]
            .groupby(lambda x: x.month_name(), sort=False)
            .sum()
        )

    def predict_deaths(self, s):
        s_train = s.loc[:self.last_training_year]
        num_predict = self.pred_year - self.last_training_year
        n_train = len(s_train)
        n = n_train + num_predict # predict into the future
        y = s_train.values
        x = np.arange(n)
        X = x.reshape(-1, 1)
        X2 = np.column_stack((x, x ** 2, x ** 3))
        
        X_train = X[:n_train]
        X2_train = X2[:n_train]

        # train model using ridge regression on box-cox transformed values
        ttr = TransformedTargetRegressor(regressor=Ridge(alpha=10), 
                                         transformer=PowerTransformer('box-cox'))
        ttr.fit(X_train, y)
        yp1 = ttr.predict(X)

        if s.name.left < 90:
            ttr.fit(X2_train, y)
            yp2 = ttr.predict(X2)
            
            # average predictions
            yp = yp1 * .9 + yp2 * .1
        else:
            yp = yp1

        index = range(s.index[0], self.pred_year + 1)
        sp = pd.Series(yp, index=index)
        return sp

    def compute_month_mult(self):
        deaths = self.daily_deaths.loc[:f"{self.last_training_year}"]
        daily_month = deaths.resample("M").mean()
        daily_year = deaths.resample("Y").mean().reindex(daily_month.index, method="bfill")
        month_mult = daily_month / daily_year
        month_mult = month_mult.groupby(lambda x: x.month_name(), sort=False).mean()
        return month_mult

    def compute_expected(self, age_group, year):
        if not isinstance(age_group, slice):
            age_slice = convert_age_to_slice(age_group)
        else:
            age_slice = age_group
        deaths = self.expected_deaths.loc[year, age_slice].sum()
        is_leap_year = year % 4 == 0
        days = DAYS_PER_MONTH[is_leap_year].sum()
        return self.month_mult[age_group] * deaths / days     

    def get_expected_month(self):
        dfs = []
        first_daily_year = self.daily_deaths.index[0].year
        years = range(first_daily_year, 2022)
        for year in years:
            df = pd.DataFrame()
            for age_group in self.daily_deaths.columns:
                df[age_group] = self.compute_expected(age_group, year)
            dfs.append(df)
        expected_deaths_per_month = pd.concat(dfs, keys=years)
        flat_index = expected_deaths_per_month.index.to_flat_index()
        index = pd.PeriodIndex([pd.Period(f'{val[1]}{val[0]}') for val in flat_index])
        expected_deaths_per_month.index = index

        self.expected_deaths_per_month = expected_deaths_per_month
        self.actual_deaths_per_month = self.daily_deaths.resample('M', kind='period').mean()
        self.excess_deaths_per_month = (self.actual_deaths_per_month - self.expected_deaths_per_month).dropna()

        idx = pd.date_range(f'{first_daily_year}', self.last_daily_date)
        self.expected_deaths_daily = expected_deaths_per_month.to_timestamp().reindex(idx, method='ffill')
        self.excess_deaths_daily = self.daily_deaths - self.expected_deaths_daily
        self.excess.iloc[-1] = self.excess_deaths_daily.loc['2021', 'total'].sum()

    def plot_month_exp_vs_actual(self, age, year, how="daily", kind="line", ax=None, **kwargs):
        year = str(year)
        if ax is None:
            fig, ax = plt.subplots(**kwargs)
        actual = self.actual_deaths_per_month.loc[year:, age].to_timestamp()
        expected = self.expected_deaths_per_month.loc[year:, age].to_timestamp()
        actual  = actual.asfreq('D').interpolate()
        expected  = expected.asfreq('D').interpolate()
        expected.plot(label='expected', ax=ax, color='black', ls='-', lw=2)
        actual.plot(label='actual', ax=ax, color='black', lw=2, ls='--')
        
        n = len(actual)
        ax.fill_between(actual.index, actual.values, expected[:n], 
                        actual.values > expected.values[:n], color='red', label='Excess')
        ax.fill_between(actual.index, actual.values, expected[:n], 
                        actual.values < expected.values[:n], color='green', label='Deficit')
        ax.legend()
        if age == "total":
            age = ' '
        else:
            age = ' ' + age + ' '
        ax.set_title(f'{self.country}{age}Actual vs Expected Daily Deaths by Month', size=12)

    def plot_trend_line(self, year, age=None, ax=None, legend=True, auto_label=True, 
                        return_data=False, return_vals=False, **kwargs):
        if age:
            if isinstance(age, str):
                age_slice = convert_age_to_slice(age)

            expected = self.expected_deaths.loc[year, age_slice].sum()
            excess = self.excess_deaths_daily.loc['2021', age].sum()
            actual_all = (self.rate_grouped.loc[:, age_slice] * self.age_grouped.loc[year, age_slice]).sum(axis=1)
            expected_all = (self.rate_pred.loc[:, age_slice] * self.age_grouped.loc[year, age_slice]).sum(axis=1)
        else:
            excess = self.excess.values[-1]
            expected = self.expected_deaths.sum(axis=1).loc[year]
            actual_all = (self.rate_grouped * self.age_grouped.loc[year]).sum(axis=1)
            expected_all = (self.rate_pred * self.age_grouped.loc[year]).sum(axis=1)
        current_pace = expected + excess

        if ax is None:
            fig, ax = plt.subplots(**kwargs)
        else:
            ax.set_title(f"{self.country}", size=10)

        if auto_label:
            ax.set_ylabel(
                f"Standardized Deaths",
                size=11,
            )
            ax.set_title(f"{self.country} {age} Expected Deaths vs Current Pace - {year}", size=12)

        ax.scatter(actual_all.index, actual_all.values, s=10)
        model = ax.plot(expected_all.index, expected_all.values, label="Model")
        if year == 2021:
            exp_year = expected_all.values[-1]
            point_pace = ax.scatter([2021], [current_pace], label="Current Pace", zorder=4)
            point_expected = ax.scatter([2021], [exp_year], label="Expected")

        if legend:
            ax.legend()
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        if return_data:
            return model[0], point_pace, point_expected
        if return_vals:
            return actual_all, expected_all

    def plot_single_cumulative_excess(self, age, start, end, **kwargs):
        s = self.excess_deaths_daily.loc[start:end, age].cumsum()
        fig, ax = plt.subplots(**kwargs)
        s.plot(ax=ax)
        phrase = 'Cumulative Excess Deaths'
        if age != 'total':
            phrase = age + ' ' + phrase
            

        ax.set_title(f'{self.country} {phrase}', size=10)

    def plot_rate(self, age, freq="W"):
        if age.endswith("+"):
            start = int(age.strip("+"))
            end = None
        elif age == "total":
            start = end = None
        else:
            ages = age.split("-")
            start = int(ages[0])
            end = int(ages[1])

        sl = slice(start, end)
        d_total_2020 = self.daily_deaths.loc["2020", age].sum()
        d_total_2021 = self.daily_deaths.loc["2021", age].sum()
        d_daily_2021 = self.daily_deaths.loc["2021", age]

        n_2021 = len(d_daily_2021)
        d_cumsum_2021 = d_daily_2021.cumsum()
        pop80_2020 = self.age_grouped.loc[:, sl].sum(axis=1)[2020] / 1000
        pop80_2021 = self.age_grouped.loc[:, sl].sum(axis=1)[2021] / 1000
        rate_2020 = (d_total_2020 / 366) * 7 / pop80_2020
        rate_2021 = (d_total_2021 / n_2021) * 7 / pop80_2021

        days = np.arange(1, n_2021 + 1)
        rate_cum_2021 = (d_cumsum_2021 / days) * 7 / pop80_2021
        rate_cum_2021 = rate_cum_2021.rolling(28, 14, center=True).mean()

        fig, ax = plt.subplots()
        weekly_rate_2021 = d_daily_2021.resample(freq).mean() * 7 / pop80_2021

        ax = weekly_rate_2021.plot(marker=".")
        ax.axhline(
            rate_2020, color="black", ls="--", label="2020 overall weekly deaths per 1000"
        )
        ax.axhline(rate_2021, ls="--", label="2021 overall weekly deaths per 1000")
        ax.legend()

    def plot_excess_and_trend(self, age, start=2018, end=2021, figsize=(10, 7)):
        fig = plt.Figure(figsize=figsize, tight_layout=True)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        self.plot_month_exp_vs_actual(age, start, ax=ax1, figsize=(10, 4))
        self.plot_trend_line(end, age, ax=ax2)
        ax1.set_xlabel(None)
        ax2.set_xticks(range(self.start_year, 2022, 2))
        ax2.text(0.05, -.12, 'Deaths Standardized to 2021 Age-Stratified Population', 
                transform=ax2.transAxes, style='italic')
        return fig
