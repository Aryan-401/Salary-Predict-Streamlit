import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')


def show_explore_page():
    st.header("Explore the Data")
    st.subheader("Stack Overflow Developer Survey 2022")

    st.divider()
    st.subheader("Original Data (Sample)")
    st.code('''import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv('survey_results_public.csv')
    ''', language='python')
    df = pd.read_csv('survey_results_public.csv')
    st.dataframe(df.head())

    st.divider()
    st.subheader("Only selecting the columns we need")
    st.code('''
        df = df[['Country', 'Age' , 'EdLevel', 'YearsCodePro', 'Employment', 'ConvertedCompYearly']]
        print(df.shape)
        df.head()
        df = df.rename({'ConvertedCompYearly': 'Salary'}, axis=1)
    ''', language='python')
    df = df[['Country', 'Age', 'EdLevel', 'YearsCodePro', 'Employment', 'ConvertedCompYearly']]
    st.dataframe(df.head())
    st.write(df.shape)
    df = df.rename({'ConvertedCompYearly': 'Salary'}, axis=1)

    st.divider()
    st.subheader("Removing rows with missing values")
    st.code('''
        df = df[df['Salary'].notnull()]
        df = df.dropna()
        print(df.shape)
        df.head()
        ''', language='python')
    df = df[df['Salary'].notnull()]
    df = df.dropna()
    st.write(df.shape)
    st.dataframe(df.head())

    st.divider()
    st.subheader("Removing rows with null values")
    st.code('''
    df = df[df['Employment'] != 'I prefer not to say']
    print(df.shape)
    ''', language='python')
    df = df[df['Employment'] != 'I prefer not to say']
    st.write(df.shape)

    st.divider()
    st.subheader("Keeping only columns where developers are employed full-time")
    st.code('''
    df = df[df['Employment'].str.contains('Employed, full-time')]
    print(df.shape)
    df.head()
''', language='python')
    df = df[df['Employment'].str.contains('Employed, full-time')]
    st.write(df.shape)
    st.dataframe(df.head())

    st.divider()
    st.subheader("Removing the Employment column, since we don't need it anymore")
    st.code('''
    df.drop('Employment', axis=1, inplace=True)
    ''', language='python')
    df.drop('Employment', axis=1, inplace=True)

    st.divider()
    st.subheader("Creating a function to lower the number of categories in the Country column")
    st.code('''
    def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map
    ''', language='python')

    def shorten_categories(categories, cutoff):
        categorical_map = {}
        for i in range(len(categories)):
            if categories.values[i] >= cutoff:
                categorical_map[categories.index[i]] = categories.index[i]
            else:
                categorical_map[categories.index[i]] = 'Other'
        return categorical_map

    st.divider()
    st.code('''
    map_country = shorten_categories(df.Country.value_counts(), 600)
df['Country'] = df['Country'].map(map_country)
''', language='python')
    map_country = shorten_categories(df.Country.value_counts(), 600)
    df['Country'] = df['Country'].map(map_country)

    st.divider()
    st.subheader("Using a boxplot to visualize the distribution of salaries by country")
    st.code('''
    fig, ax = plt.subplots(1,1, figsize=(12,7))
df.boxplot('Salary', 'Country', ax=ax)
plt.suptitle('USD Salary Distributions by Country')
plt.title('')
plt.yscale('log')
plt.ylabel('Salary in USD')
plt.xticks(rotation=90)
plt.show()
''', language='python')
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    df.boxplot('Salary', 'Country', ax=ax)
    plt.suptitle('USD Salary Distributions by Country')
    plt.title('')
    plt.yscale('log')
    plt.ylabel('Salary in USD')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.divider()
    st.subheader("Removing outliers")
    st.code('''
    df = df[df['Salary'] < 300000]
    df = df[df['Salary'] > 1000]
    print(df.shape)
''', language='python')
    df = df[df['Salary'] < 300000]
    df = df[df['Salary'] > 1000]
    st.write(df.shape)

    st.divider()
    st.subheader("Replacing the long strings in the EdLevel column with shorter, integer, ones")
    st.code('''
    df['YearsCodePro'].replace('Less than 1 year', 0, inplace=True)
    df['YearsCodePro'].replace('More than 50 years', 51, inplace=True)
    df['YearsCodePro'] = df['YearsCodePro'].astype(float)
    df.head()
''', language='python')
    df['YearsCodePro'].replace('Less than 1 year', 0, inplace=True)
    df['YearsCodePro'].replace('More than 50 years', 51, inplace=True)
    df['YearsCodePro'] = df['YearsCodePro'].astype(float)
    st.dataframe(df.head())

    st.divider()
    st.subheader("Removing rows with null values")
    st.code('''
    df = df[df['Age'] != 'Prefer not to say']
    print(df.shape)
    df.head()
''', language='python')
    df = df[df['Age'] != 'Prefer not to say']
    st.write(df.shape)
    st.dataframe(df.head())

    st.divider()
    st.subheader("Replacing the long strings in the Age column with shorter, integer, ones")
    st.code('''
    df.replace('25-34 years old', 30, inplace=True)
df.replace('35-44 years old', 40, inplace=True)
df.replace('18-24 years old', 22, inplace=True)
df.replace('45-54 years old', 50, inplace=True)
df.replace('55-64 years old', 60, inplace=True)
df.replace('Under 18 years old', 17, inplace=True)
df.replace('65 years or older', 65, inplace=True)
df['Age'] = df['Age'].astype(float)
df.head()
''', language='python')
    df.replace('25-34 years old', 30, inplace=True)
    df.replace('35-44 years old', 40, inplace=True)
    df.replace('18-24 years old', 22, inplace=True)
    df.replace('45-54 years old', 50, inplace=True)
    df.replace('55-64 years old', 60, inplace=True)
    df.replace('Under 18 years old', 17, inplace=True)
    df.replace('65 years or older', 65, inplace=True)
    df['Age'] = df['Age'].astype(float)
    st.dataframe(df.head())

    st.divider()
    st.subheader("Replacing the long strings in the Education column with shorter, integer, ones")
    st.code('''
    def clean_education(x):
        if 'Bachelor’s degree' in x:
            return 'Bachelors'
        elif 'Master’s degree' in x:
            return 'Masters'
        elif 'Professional degree' in x or 'Other doctoral' in x:
            return 'Post Graduation'
        else:
            return 'Less than a Bachelors'
        df['EdLevel'] = df['EdLevel'].apply(clean_education)
        print(df.shape)
        df.head()
    ''', language='python')

    def clean_education(x):
        if 'Bachelor’s degree' in x:
            return 'Bachelors'
        elif 'Master’s degree' in x:
            return 'Masters'
        elif 'Professional degree' in x or 'Other doctoral' in x:
            return 'Post Graduation'
        else:
            return 'Less than a Bachelors'

    df['EdLevel'] = df['EdLevel'].apply(clean_education)
    st.write(df.shape)
    st.dataframe(df.head())

    st.divider()
    st.subheader("Using LabelEncoder to convert the categorical columns to numerical ones")
    st.code('''
    from sklearn.preprocessing import LabelEncoder
    from numpy import ravel
    
    encoder_ed = LabelEncoder()
    df['EdLevel'] = encoder_ed.fit_transform(ravel(df[['EdLevel']].values))
        ''', language='python')
    from sklearn.preprocessing import LabelEncoder
    from numpy import ravel

    encoder_ed = LabelEncoder()
    df['EdLevel'] = encoder_ed.fit_transform(ravel(df[['EdLevel']].values))

    st.divider()
    st.subheader("Using LabelEncoder to convert the categorical columns to numerical ones")
    st.code('''
    encoder_country = LabelEncoder()
    df['Country'] = encoder_country.fit_transform(ravel(df[['Country']].values))
''', language='python')
    encoder_country = LabelEncoder()
    df['Country'] = encoder_country.fit_transform(ravel(df[['Country']].values))

    st.divider()
    st.subheader("Saving the cleaned data to a new csv file")
    st.code('''
    df.to_csv('clean_servey_results_public.csv', index=False)
''', language='python')
    df.to_csv('clean_servey_results_public.csv', index=False)


    







