################################### RULE BASED CLASSIFICATION  ########################################


# A game company wants to create level-based new customer definitions (personas) by using some
# features ( Country, Source, Age, Sex) of its customers, and to create segments according to these new customer
# definitions and to estimate how much profit can be generated from  the new customers according to these segments.

# In this study, how to do rule-based classification and customer-based revenue calculation
# have been discussed step by step.


########################## Importing Libraries ##########################

print(14 * " >", "\t\t n.B.a. \t", "< " * 14, "\n\n\n")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 15)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', True)


########################## Load The Data ##########################

def load_dataset(datasetname):
    return pd.read_csv(f"{datasetname}.csv")


df = load_dataset("/persona")
df.head()


########################### Describe The Data ######################
def check_df(dataframe):
    print(f"""
        ##################### Shape #####################\n\n\t{dataframe.shape}\n\n
        ##################### Types #####################\n\n{dataframe.dtypes}\n\n
        ##################### Head #####################\n\n{dataframe.head(3)}\n\n
        ##################### NA #####################\n\n{dataframe.isnull().sum()}\n\n
        ##################### Quantiles #####################\n\n{dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T}\n\n""")


check_df(df)


######################## Selection of Categorical and Numerical Variables ########################

# Let's define a function to perform the selection of numeric and categorical variables in the data set in a parametric way.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


######################## General Exploration for Categorical Data ########################


def cat_summary(dataframe, plot=False):
    # cat_cols = grab_col_names(dataframe)["Categorical_Data"]
    for col_name in cat_cols:
        print("############## Unique Observations of Categorical Data ###############")
        print("The unique number of " + col_name + ": " + str(dataframe[col_name].nunique()))

        print("############## Frequency of Categorical Data ########################")
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": dataframe[col_name].value_counts() / len(dataframe)}))
        if plot:  # plot == True (Default)
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show()


cat_summary(df, plot=True)


######################## General Exploration for Numerical Data ########################

def num_summary(dataframe, plot=False):
    numerical_col = ['PRICE', 'AGE']  ##or grab_col_names(dataframe)["Numerical_Data"]
    quantiles = [0.25, 0.50, 0.75, 1]
    for col_name in numerical_col:
        print("########## Summary Statistics of " + col_name + " ############")
        print(dataframe[numerical_col].describe(quantiles).T)

        if plot:
            sns.histplot(data=dataframe, x=col_name)
            plt.xlabel(col_name)
            plt.title("The distribution of " + col_name)
            plt.grid(True)
            plt.show(block=True)


num_summary(df, plot=True)


######################## Data Analysis  ########################

def data_analysis(dataframe):
    # Unique Values of Source:
    print("Unique Values of Source:\n", dataframe[["SOURCE"]].nunique())

    # Frequency of Source:
    print("Frequency of Source:\n", dataframe[["SOURCE"]].value_counts())

    # Unique Values of Price:
    print("Unique Values of Price:\n", dataframe[["PRICE"]].nunique())

    #  Number of product sales by sales price:
    print("Number of product sales by sales price:\n", dataframe[["PRICE"]].value_counts())

    # Number of product sales by country:
    print("Number of product sales by country:\n", dataframe["COUNTRY"].value_counts(ascending=False, normalize=True))

    # Total & average amount of sales by country
    print("Total & average amount of sales by country:\n", dataframe.groupby("COUNTRY").agg({"PRICE": ["mean", "sum"]}))

    # Average amount of sales by source:
    print("Average amount of sales by source:\n", dataframe.groupby("SOURCE").agg({"PRICE": "mean"}))

    # Average amount of sales by source and country:
    print("Average amount of sales by source and country:\n", dataframe.pivot_table(values=['PRICE'],
                                                                                    index=['COUNTRY'],
                                                                                    columns=["SOURCE"],
                                                                                    aggfunc=["mean"]))


data_analysis(df)


######################## Defining Personas ########################

# Let's define new level-based customers (personas) by using Country, Source, Age and Sex.
# But, firstly we need to convert age variable to categorical data.

def define_persona(dataframe):
    # Let's define new level-based customers (personas) by using Country, Source, Age and Sex.
    # But, firstly we need to convert age variable to categorical data.

    bins = [dataframe["AGE"].min(), 18, 23, 35, 45, dataframe["AGE"].max()]
    labels = [str(dataframe["AGE"].min()) + '_18', '19_23', '24_35', '36_45', '46_' + str(dataframe["AGE"].max())]

    dataframe["AGE_CAT"] = pd.cut(dataframe["AGE"], bins, labels=labels)
    dataframe.groupby("AGE_CAT").agg({"AGE": ["min", "max", "count"]})

    # For creating personas, we group all the features in the dataset:
    df_summary = dataframe.groupby(["COUNTRY", "SOURCE", "SEX", "AGE_CAT"])[["PRICE"]].sum().reset_index()
    df_summary["CUSTOMERS_LEVEL_BASED"] = pd.DataFrame(["_".join(row).upper() for row in df_summary.values[:, 0:4]])

    # Calculating average amount of personas:
    df_persona = df_summary.groupby("CUSTOMERS_LEVEL_BASED").agg({"PRICE": "mean"})
    df_persona = df_persona.reset_index()

    return df_persona


define_persona(df)


######################## Creating Segments based on Personas ########################

def create_segments(dataframe):
    # When we list the price in descending order, we want to express the best segment as the A segment and to define 4 segments.

    df_persona = define_persona(dataframe)

    segment_labels = ["D", "C", "B", "A"]
    df_persona["SEGMENT"] = pd.qcut(df_persona["PRICE"], 4, labels=segment_labels)
    # df_segment = df_persona.groupby("SEGMENT").agg({"PRICE": "mean"})

    # Demonstrating segments as bars on a chart, where the length of each bar varies based on the value of the customer profile
    # plot = sns.barplot(x="SEGMENT", y="PRICE", data=df_segment)
    # for bar in plot.patches:
    #     plot.annotate(format(bar.get_height(), '.2f'),
    #                   (bar.get_x() + bar.get_width() / 2,
    #                    bar.get_height()), ha='center', va='center',
    #                   size=8, xytext=(0, 8),
    #                   textcoords='offset points')
    return df_persona


create_segments(df)


######################## Prediction ########################

def ruled_based_classification(dataframe):
    df_segment = create_segments(dataframe)

    def AGE_CAT(age):
        if age <= 18:
            AGE_CAT = "15_18"
            return AGE_CAT
        elif (age > 18 and age <= 23):
            AGE_CAT = "19_23"
            return AGE_CAT
        elif (age > 23 and age <= 35):
            AGE_CAT = "24_35"
            return AGE_CAT
        elif (age > 35 and age <= 45):
            AGE_CAT = "36_45"
            return AGE_CAT
        elif (age > 45 and age <= 66):
            AGE_CAT = "46_66"
            return AGE_CAT

    COUNTRY = input("Enter a country name (USA/EUR/BRA/DEU/TUR/FRA):")
    SOURCE = input("Enter the operating system of phone (IOS/ANDROID):")
    SEX = input("Enter the gender (FEMALE/MALE):")
    AGE = int(input("Enter the age:"))
    AGE_SEG = AGE_CAT(AGE)
    new_user = COUNTRY.upper() + '_' + SOURCE.upper() + '_' + SEX.upper() + '_' + AGE_SEG

    print(new_user)
    print("Segment:" + df_segment[df_segment["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:, "SEGMENT"].values[0])
    print("Price:" + str(df_segment[df_segment["CUSTOMERS_LEVEL_BASED"] == new_user].loc[:, "PRICE"].values[0]))

    return new_user


ruled_based_classification(df)
