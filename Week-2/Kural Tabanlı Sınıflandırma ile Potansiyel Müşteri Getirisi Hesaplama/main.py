import numpy as np
import pandas as pd
import seaborn as sns

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)

#####################################################

# Görev 1: Aşağıdaki Soruları Yanıtlayınız.

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

df = pd.read_csv(r"C:\Users\SERKAN\OneDrive\Masaüstü\MİUUL READMİ\persona.csv")
df.head(5)
df.info()

def check_df(dataframe, head=5):

    print("################# Shape #####################")
    print(dataframe.shape)
    print("################# Types #####################")
    print(dataframe.dtypes)
    print("################# Head ####################")
    print(dataframe.head(head))
    print("################# Tail #####################")
    print(dataframe.tail(head))
    print("################# NA #####################")
    print(dataframe.isnull().sum())
    print("################# Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

print(df["SOURCE"].nunique())
print(df["SOURCE"].value_counts())

# Soru 3: Kaç unique PRICE vardır?

print(df["PRICE"].nunique())

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

print(df["PRICE"].value_counts())

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

print(df.groupby("COUNTRY")["PRICE"].count())

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

print(df.groupby("COUNTRY")["PRICE"].sum())

# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?

print(df["SOURCE"].value_counts())

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

print(df.groupby("COUNTRY")["PRICE"].mean())

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

print(df.groupby("SOURCE")["PRICE"].mean())

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

print(df.groupby(["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"}))

#####################################################

# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

print(df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}))

#####################################################

# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.

# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

agg_df = df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
print(agg_df.head())

#####################################################

# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.

# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.

agg_df = agg_df.reset_index()
print(agg_df.head())

#####################################################

# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.

# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70] ,labels=['0_18', '19_23', '24_30', '31_40', '41_70'])
print(agg_df.head())

#####################################################

# GÖREV 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.

# Yeni eklenecek değişkenin adı: customers_level_based
# Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek
# customers_level_based değişkenini oluşturmanız gerekmektedir.

agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_"
                                   + row[2].upper() + "_" + row[5].upper()
                                   for row in agg_df.values]
agg_df = agg_df[["customers_level_based", "PRICE"]]
print(agg_df.head())

#####################################################

# GÖREV 7: Yeni müşterileri (personaları) segmentlere ayırınız.

# Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# Segmentleri betimleyiniz. (Segmentlere göre group by yapıp price mean, max, sum’larını alınız.)

SEGMENT = pd.qcut(agg_df["PRICE"], 4, labels= ["D", "C", "B", "A"])
agg_df["SEGMENT"] = SEGMENT
print(agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]}))

#####################################################

# GÖREV 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.

# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir
# ve ortalama ne kadar gelir kazandırması beklenir?

new_person = "TUR_ANDROID_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_person])
print(agg_df[agg_df["customers_level_based"] == new_person]["PRICE"].mean())

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve
# ortalama ne kadar gelir kazandırması beklenir?

new_person2 = "FRA_IOS_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_person2])
print(agg_df[agg_df["customers_level_based"] == new_person2]["PRICE"].mean())
