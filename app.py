import streamlit as st

st.markdown("""
Python programming Project
Dataset - the movie dataset
Contains data on more than 40,000 films and their ratings
First, let's understand what our data consists of.""")


import kagglehub
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
main_path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")
print(main_path)
st.write(os.listdir(main_path))
st.markdown("Look at files in dataset")
movies_metadata = pd.read_csv(main_path + "/movies_metadata.csv", low_memory=False)

st.table(movies_metadata.head().T)

st.markdown('The first criterion of the dataset is that the table below contains the average values, medians and standard deviations for at least three fields')
st.table(movies_metadata.describe())

st.write("Clearing the data - let's look at the number of empty values in the columns)")

st.write(len(movies_metadata))
st.write(movies_metadata.isnull().sum())
st.markdown("""Let's remove those films that lack imdb_id, original_language, popularity, revenue, runtime, vote_average, vote_count
These are the main columns and we don't need empty values in them. The other columns are important, but the absence of values in them is not so important.""")
movies_metadata = movies_metadata.dropna(subset=
                                         ['imdb_id', 'original_language', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count'])

st.markdown("Let's look at the distribution of ratings - read the table and build a graph")
ratings = pd.read_csv(main_path + "/ratings.csv", low_memory=False)
st.table(ratings.head())
rating_distribution = ratings.groupby('rating')['rating'].count()

plt.figure(figsize=(10, 6))
plt.bar(
    rating_distribution.index,
    rating_distribution.values,
    color='blue',
    width=0.4
)

ticks = np.arange(0.5, 5.5, 0.5)
plt.xticks(ticks)
plt.title("Rate frequency")
plt.xlabel("Rate")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

st.pyplot(plt)
st.markdown("""It can be seen that most often users give a rating of 4. In general, viewers tend to give whole ratings(5, 4, 3, 2, 1), than not whole (4.5, 3.5, 2.5, 1.5, 0.5)
For the next graph, enter the column - profit (revenue minus budget)""")
movies_metadata['revenue'] = pd.to_numeric(movies_metadata['revenue'], errors='coerce')
movies_metadata['budget'] = pd.to_numeric(movies_metadata['budget'], errors='coerce')
movies_metadata['profit'] = movies_metadata['revenue'] - movies_metadata['budget']
top_profit_movies = movies_metadata[['title', 'profit']].sort_values(by='profit', ascending=False).head(10)

st.markdown("""Plotting the current of 10 profitable films""")
plt.figure(figsize=(12, 6))
sns.barplot(x=top_profit_movies['profit'], y=top_profit_movies['title'], palette='rocket')
plt.title("Top 10 most profitable films")
plt.xlabel("Profit")
plt.ylabel("Film name")
st.pyplot(plt)
st.markdown("Another graph is the distribution of the duration of films")
filtered_runtime = movies_metadata[movies_metadata['runtime'].between(30, 300)]['runtime']

plt.figure(figsize=(12, 8))
sns.histplot(filtered_runtime, bins=50, color='green', orientation='horizontal')
plt.title("The distribution of the duration of films", fontsize=14)
plt.xlabel("Duration (in minutes)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(plt)
st.markdown("""Let's introduce 3 more columns - the month of release, the day of release and the year""")
movies_metadata['date'] = pd.to_datetime(movies_metadata['release_date'], format='%Y-%m-%d')
movies_metadata['month'] = movies_metadata['date'].dt.month_name()
movies_metadata['weekday'] = movies_metadata['date'].dt.day_name()
movies_metadata['r_year'] = movies_metadata['date'].dt.year
st.markdown("We are building a graph - the distribution of ratings by month")

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

movies_metadata['month'] = pd.Categorical(movies_metadata['month'],
                                          categories=month_order,
                                          ordered=True)
plt.figure(figsize=(12, 6))
sns.boxplot(data=movies_metadata, x='month', y='vote_average', order=month_order, palette='Set3')

plt.title("Rating by months", fontsize=16)
plt.xlabel("Months", fontsize=14)
plt.ylabel("Rating", fontsize=14)

plt.xticks(rotation=45)
st.pyplot(plt)
st.markdown("Next, let's look at the profit by month and by day of the week")
average_revenue_by_weekday = movies_metadata.groupby('weekday')['profit'].mean()

plt.figure(figsize=(10, 6))
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

average_revenue_by_weekday = average_revenue_by_weekday.reindex(days_order)
average_revenue_by_weekday.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Mean profit by weekday', fontsize=16)
plt.xlabel('Weekday', fontsize=14)
plt.ylabel('Mean profit', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
st.pyplot(plt)

average_revenue_by_month = movies_metadata.groupby('month')['revenue'].mean()

plt.figure(figsize=(10, 6))
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

average_revenue_by_month = average_revenue_by_month.reindex(month_order)
average_revenue_by_month.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title('Mean profit by months', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Mean profit', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

st.pyplot(plt)

st.markdown("""The profit of the films that were released on Wednesday is really higher. 
Let's put forward a hypothesis - films released on Wednesday make a big profit. 
Let's put forward a null hypothesis - the profit of films released on Wednesday and on other days does not differ. 
Let's check this using p-value""")
wednesday_profit = movies_metadata[movies_metadata['weekday'] == 'Wednesday']['profit']
other_days_profit = movies_metadata[movies_metadata['weekday'] != 'Wednesday']['profit']
from scipy.stats import ttest_ind
stat, p_value = ttest_ind(wednesday_profit, other_days_profit, equal_var=False)
st.write(p_value)
st.markdown("""Therefore, the null hypothesis is rejected and our main hypothesis is confirmed.""")
st.markdown("""This conclusion is very useful for the film business. 
After all, by releasing the film on Wednesday, we can expect a big profit. 
Perhaps this may be due to the fact that it is convenient for people to go to a movie after work, 
because Wednesday is the middle of the week and by the end of the week people have more and more working hours.""")
st.markdown("Let's show restapi")

st.session_state["main_df"] = ratings
response_area = st.container()


def get_function(data_filter=None, page_filter=None,):
    main_df = st.session_state["main_df"]
    filtered_data = main_df.copy()

    if data_filter:
        filtered_data = filtered_data.sort_values(by=data_filter)
    if page_filter is not None:
        filtered_data = filtered_data[10*(page_filter-1):10*page_filter]

    return filtered_data


def post_function(new_data):
    try:
        userId = new_data.get("userId")
        movieId = new_data.get("movieId")
        rating = new_data.get("rating")
        timestamp = new_data.get("timestamp")
        if not userId or not movieId or not rating or not timestamp:
            return {"error": "not all fields are full"}
        new_row = {"userId": userId, "movieId": movieId, "rating": rating, "timestamp": timestamp}
        st.session_state["main_df"] = pd.concat([st.session_state["main_df"], new_row], ignore_index=True)
        return {"message": "Well done"}
    except Exception as e:
        return {"error": str(e)}


st.subheader("GET Filter")
data_filter = st.text_input("Filtered by 'userId', 'movieId', 'rating', 'timestamp':")
page_filter = st.number_input("page_number:", value=None, step=1)

if st.button("Send GET"):
    response = get_function(data_filter, page_filter)
    with response_area:
        st.write("GET response:")
        st.table(response)

st.subheader("POST - make new row to raitings.csv")
d_1 = st.number_input("userId:", step=1)
d_2 = st.number_input("movieId:",step=1)
d_3 = st.number_input("rating:", step=0.5)
d_4 = st.number_input("timestamp:", step=1)

if st.button("Send POST"):
    new_data = {"userId": int(d_1), "movieId": int(d_2), "rating": int(d_3), "timestamp": int(d_4)}
    response = post_function(new_data)
    with response_area:
        st.write("РOST result")
        st.json(response)